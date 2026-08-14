# PRISMA ConWeave：RL Core v8 里程碑与重大重构前版本对比

日期：2026-07-14  
重构前只读基线：`/home/cuiyiqin/Cyq_RL/PRISMA_conweave_backup7.13`  
当前主版本：`/home/cuiyiqin/Cyq_RL/PRISMA_conweave`  
里程碑档案：`/home/cuiyiqin/Cyq_RL/milestones/rl_core_v8_beats_ecmp_20260714/PRISMA_conweave_rl_core_v8_beats_ecmp_20260714.tar.gz`

## 1. 结论摘要

这次工作不是一次普通调参，而是修复了从 C++ 状态生产、Python 预处理、
`env.step()` transition、训练调度、Q 网络归纳偏置、flowlet 路由所有权，
一直到 checkpoint 保存加载和冻结评估的整条闭环。

重构前版本可以产生看似正常的训练曲线，但存在以下结构性问题：

1. destination ID 被除以 64 后再转成整数 one-hot，当前 8 个 overlay destination
   基本全部坍缩为类别 0；
2. `lb=rl` 时却读取只在 CONGA 路径维护的 OneHopMetrics，多个拥塞特征实际为
   常数或陈旧值；
3. replay transition 通过 `pending` 延迟到下一次 `env.step()` 才闭环，使
   action、reward 和 next state 错位一拍；
4. 20 us flowlet 边界外又叠加 1.5 ms dwell，且 continuation packet 经过另一个
   TTL 不一致的 cache，导致绝大多数新 flowlet 没有新决策，并存在静默 ECMP
   fallback；
5. 普通 MLP 容易记忆端口身份和“快链路容量大”的捷径，冻结部署时即使快链路
   已饱和，仍会持续选择快链路；
6. 训练更新量由 wall-clock trainer thread 的唤醒时机决定，checkpoint 保存前
   可能仍有未完成更新，训练不可复现；
7. 旧 checkpoint 没有 observation/reward/network contract manifest，加载异常
   还可能被吞掉，无法证明训练和评估使用同一份语义。

v8 将上述问题改为显式、可测试的契约。最终在相同非对称拓扑、相同 50 ms
流量文件、netload 70、seed 100 下，冻结模型 `174791074` 以 `epsilon=0`
完成 6660/6660 条流，并在绝大多数 FCT 指标上超过配对 ECMP `401470137`。

## 2. 核心机制对比

| 机制 | backup7.13 | 当前 rl-core-v8 | 直接影响 |
|---|---|---|---|
| Observation layout | `[dst,lastAction]+N*5`，语义依赖 CONGA OneHopMetrics | `[dst,lastAction]+N*6`：瞬时队列、队列 EMA、RL-native DRE、队列趋势、容量感知 headroom、相对带宽 | 状态来自 `lb=rl` 真正维护的数据 |
| Destination 编码 | `dst/64` 后再 cast 为整数 one-hot | 保留原始整数 destination 后 one-hot | 不再把不同目的节点压成同一类别 |
| 连续特征缩放 | 通用 `normalize_obs`，环境变量可改变缩放；缺乏维度/范围校验 | C++ 固定量化到 `[0,5000]`，Python 只除一次；维度、范围、NaN、lastAction 整数性全部 fail-fast | 训练和加载评估使用同一 observation contract |
| 状态源 | RL 模式读取 dormant CONGA 状态，实测多项为常数 | MacTx 队列采样、RL-native DRE、idle-port decay、真实链路容量 | Agent 能看到拥塞与异速链路差异 |
| Transition | 当前 action 先放入 `pending`，下一次 step 才配 reward/next state | `env.step(a_t)` 返回后立即写 `(s_t,a_t,r_{t+1},s_{t+1},done)` | 消除一拍错位和错误 credit assignment |
| Flowlet 决策 | gap=20 us，额外 dwell=1.5 ms | gap=20 us，额外 dwell=0 | 每个真实新 flowlet 都请求新动作 |
| Continuation 路由 | 经过 SwitchNode 第二份 cache；过期后可能走默认路径 | `FlowletCtx.lastActionOutIf` 是唯一 owner，continuation 直接 `RlRelease` | 同一 flowlet 路径一致，无正常路径静默 ECMP |
| Reward | `0.5*(1-util)+0.3*queueHealth+0.2*(1-reorder²)-trendPenalty`；500 Mbps 与 1 Gbps 空闲端口最高 reward 相同 | `0.6*capacityHeadroom+0.2*queueEmaHealth+0.1*instantQueueHealth+0.1*trendHealth` | reward 与异速链路可用服务能力对齐 |
| Reorder 信号 | 参与 TD target，但修复 flowlet 后通常接近 0，形成动作无关常数 | 保留为诊断，不进入主 reward | 减少无效常数对 TD 学习的污染 |
| Q 网络 | 普通全连接 MLP；端口输出不共享，容易记住端口 ID；ELU Q head | `Q(s,a)=V(s)+P_native(s,a)+0.1*tanh(Rθ(s,a))`；所有 action 共享 scorer，端口置换等变 | 欠训练模型也不能仅凭容量记忆把饱和快链排在空闲慢链前 |
| Last action | 被输入普通 MLP，可能形成端口身份捷径 | 保留在线路契约和诊断中，但不进入 v8 scorer | 避免“上一次选谁就继续选谁”的伪稳定策略 |
| Target 初始化 | online 与 target 独立随机初始化 | 初始化后立即 hard copy | 第一批 Bellman target 不再来自随机 target 网络 |
| Bootstrap | target 网络直接取 `max` | Double DQN：online 选 action、target 估值 | 减少 8 个动作最大值导致的过估计 |
| Epsilon | 请求的 epsilon 在本次决策之后才写入变量 | 决策前应用；train schedule 和 eval epsilon 分离 | 冻结评估可严格保证 `epsilon=0` |
| 训练调度 | wall-clock 随机 sleep + simulation-time gate | 每个 Agent 按 replay sample 数确定更新数：256 samples 后起训，每 4 个新样本更新一次 | 固定流量和 seed 对应确定的 gradient 数 |
| Target 更新 | 与 simulation time/sync thread 耦合 | 每个 Agent 每 100 gradient steps 更新 | lagged target 的语义稳定且可审计 |
| 结束与保存 | daemon trainer 可能还在更新时保存 checkpoint | 停止 trainer、join、drain 所有应做更新，确认 pending=0 后保存 | FCT 行为与 final checkpoint 的时间关系明确 |
| Imitation learning | 可尝试用 ECMP hash action 暖启动，但 observation 没有 flow identity，标签不可学习 | 参数保留兼容但强制禁用 | 当前结果不依赖 ECMP/CONGA imitation label |
| Checkpoint | 无契约 manifest；目录枚举宽松；加载异常打印后继续 | `checkpoint_manifest.json` 严格校验 `rl-core-v8`；只认 `nodeN`；任何加载失败立即终止 | 不允许旧模型被误当成 v8 部署 |
| 诊断 | 训练 loss/reward 为主，难证明动作真正下发 | action_seq、Python/C++ match、fresh flowlet、fallback、behavior/greedy、端口分布、Q range、transition mismatch | 能区分“曲线好看”和“策略真正生效” |
| 测试 | `prisma/tests` 无测试文件 | 25/25：observation、transition、调度、Q结构、置换、饱和快链、serialization、process、logger | 关键契约可回归验证 |

## 3. Observation contract：从隐式猜测变为单一事实源

### 3.1 重构前

旧 Python 预处理会执行：

```text
destination = destination / 64
last_action = last_action / 8
features = features / 5000
```

而网络随后会把 destination cast 为 `int64` 再 one-hot。当前 overlay destination
为 0–7，因此除以 64 后除 0 外均为 0.x，cast 后几乎都变成 0。网络事实上无法
区分目的 leaf。

旧 C++ 每个出口生成 5 个号称 CONGA 的特征：cost、local CE、remote CE、
queue derivative、covariance。但 `lb=rl` 走 mode 7，CONGA 的初始化和
`RouteInput` 只在 CONGA mode 中维护这些指标。调试 output `568075741` 已证明
其中多项在 RL 模式下为常数。

### 3.2 v8

`rl_contract.py` 成为 Python 侧唯一契约：

```text
[destination_overlay, last_action_index]
  + N * [queue_occupancy,
         queue_ema,
         dre_utilization,
         queue_trend,
         headroom,
         link_capacity]
```

- destination 保持整数，供 one-hot 使用；
- last action 只按 `N-1` 缩放，当前仅用于诊断；
- 6 个连续量均由 C++ 限制在 `[0,1]` 后量化至 `[0,5000]`；
- Python 验证 shape、feature scale、范围、整数 action 和 finite 值；
- train 和 eval 都调用同一个 `preprocess_observation()`。

## 4. Transition 与训练时序

ns3-gym 的一次真实交互语义是：

```text
env.step(a_t) -> (s_{t+1}, r_{t+1}, done, info)
```

重构前先将 `(s_t,a_t)` 存入 `Agent.pending`，再在下一次 `env.step()` 返回时弹出。
这相当于把当前返回的 reward/next state 再延迟一拍，模型训练的不是实际执行动作
产生的结果。

v8 在同一次调用返回后立即构造 `AlignedTransition`，并用 `action_seq` 验证 C++
实际消费动作的顺序。最终训练 output `775665073` 中：

- 136144 个新 flowlet；
- 136143/136143 次 action match；
- 136136 条 transition，恰好等于决策数减去 8 个 Agent 的首个无前驱动作；
- mismatch=0，pending=0。

训练量也不再由线程“抢到多少时间”决定。每个 Agent 的目标 gradient 数只取决于
本地 replay 样本数，并在仿真结束后 drain 完成。`775665073` 共执行 33526 次
gradient update、331 次 target update，最终所有 Agent `epsilon=0.1`。

## 5. Flowlet 路由所有权

重构前同时存在三种时间尺度：

- 新 flowlet gap：20 us；
- 额外 minimum dwell：1.5 ms，即 gap 的 75 倍；
- SwitchNode preferred-route cache：约 1 ms。

这意味着 cache 可以先于 dwell 过期：Agent 仍不允许新决策，但 continuation path
已经失效，包会进入默认 `SendToDevContinue` 路径。表现为 RL 决策覆盖率低、路径
所有权模糊，并可能静默回退 ECMP。

v8 的规则只有两条：

1. 每个检测到的 20 us 新 flowlet 请求一次新 action；
2. 同一 flowlet 后续包始终读取 `FlowletCtx.lastActionOutIf` 并直接释放到该出口。

SwitchNode cache 仅保留为 20 us 的 defensive cache，不再是正常 continuation path。
最终 frozen output `174791074`：new=129247、fresh ratio=1、
reused packets=201430、dwell reuse=0、fallback=0。

## 6. Reward 与安全 Q 分解

### 6.1 重构前 reward 的异速链路盲点

旧主项使用相对利用率 `(1-utilization)`。空闲的 500 Mbps 和 1 Gbps 链路都能
得到 1.0，因此模型无法仅由 reward 判断哪条链路提供更多绝对剩余服务能力。
output `450009057` 中曾观察到慢链路平均 reward 反而更高。

### 6.2 v8 reward

定义：

```text
headroom(a) = relative_capacity(a) * (1 - dre_utilization(a))

r(a) = 0.6 * headroom(a)
     + 0.2 * queue_ema_health(a)
     + 0.1 * instant_queue_health(a)
     + 0.1 * queue_trend_health(a)
```

reward 与 observation、Q prior 使用同一组 RL-native 信号，且不读取 ECMP、CONGA
或其他算法的动作标签。

### 6.3 v8 Q 网络

```text
Q(s,a) = V(s) + P_native(s,a) + 0.1 * tanh(R_theta(s,a))
```

- `V(s)` 无界，用于拟合折扣回报的公共状态价值；
- `P_native` 给出容量 headroom 与队列健康的安全排序；
- `R_theta` 是真正学习的、action-dependent 的修正项，限制在 ±0.1；
- 每个 action 使用同一套共享 Dense scorer，因此交换端口排列会等价交换 Q 输出；
- last action 不进入 scorer，避免端口身份记忆。

这一设计保证欠训练 checkpoint 不会仅因为“端口 0 通常是快链”就持续选择已饱和
的快链，同时仍允许 RL 在状态接近时学习比手工 prior 更细的选择。

需要坦诚说明：当前系统不是“完全无先验的黑盒神经网络”。里程碑证明的是
“安全归纳偏置约束下的 DQN 系统”在冻结部署中超过 ECMP。下一项必须做的实验是
关闭 learned residual 的 `prior-only` 消融，以量化真正由学习贡献的增益。

## 7. 保存、加载与冻结评估

旧版从目录数量推断 Agent 数量，目录中出现非模型文件时可能出错；单个模型加载
异常只打印 traceback 后继续。更关键的是没有记录 observation、reward、Q 网络和
flowlet contract，几个月前的模型可能被加载到完全不同的运行语义中。

v8 每次保存都会写入 `checkpoint_manifest.json`，加载时必须匹配：

```text
rl_core_version = rl-core-v8
observation producer/layout/scale
reward version and weights
flowlet routing contract
Q-network architecture
training/bootstrap semantics
```

`train=0, lb=rl` 强制要求 `--load_path`，默认 `eval_epsilon=0`。任何缺失 manifest、
旧 core version、模型目录缺失或反序列化失败都会立即终止，不再产生“看似加载了”
的结果。

## 8. 行为证据

### 8.1 调查链上的失败结果

- `816279714` 是触发本次调查的旧模型加载评估，small/large average slowdown
  分别为 518.653/426.187。它与最终 50 ms 配对实验并非严格同配置，只作为
  “旧训练曲线不能代表部署能力”的诊断证据。
- 中间 v7 `830493679` 修通了部分数据链，但冻结策略仍发生快端口 shortcut：
  small/large average 为 20.683/30.205，且存在未完成流。它不是 backup7.13，
  也不是最终 v8，只记录重构过程中发现的模型结构问题。

### 8.2 最终严格配对结果

相同 topology、50 ms flow file、netload 70、DCTCP、PFC=0、IRN=1、seed=100：

| FCT slowdown | ECMP `401470137` | Frozen v8 `174791074` | v8 改善 |
|---|---:|---:|---:|
| Small average | 3.457 | 2.607 | 24.6% |
| Small P95 | 10.840 | 6.344 | 41.5% |
| Small P99 | 16.587 | 9.615 | 42.0% |
| Large average | 4.859 | 3.511 | 27.7% |
| Large P95 | 15.412 | 7.510 | 51.3% |
| Large P99 | 24.455 | 10.647 | 56.5% |

唯一已记录的轻微回退是 large median：RL 2.978，ECMP 2.904（+2.5%）。

冻结评估不是训练期探索行为：129247 次决策全部为 greedy，random=0；快/慢链路
动作占比为 75.2%/24.8%，单端口最高 19.01%，没有再发生端口坍缩。

## 9. AI 流量场景的当前准备度

### 9.1 可以设计，但统一入口还没有接通

`conweave-ns3-main/run.py` 已有三种 traffic mode：

- `cdf`：当前 AliStorage 类随机数据中心流；
- `allreduce`：可配置 round、step、jitter、chunk size、ring rotation；
- `alltoall`：可配置 round、chunk size、burst interval。

现有生成器已经能表达 ring AllReduce 和同步 All-to-All/MoE-like burst 的雏形，
服务器上也已有 AR/AA 配置文件。但当前 `prisma/source/argument_parser.py` 和
`prisma/source/run_ns3.py` 尚未暴露/传递这些参数，因此还不能遵守“所有实验都从
`prisma/main.py` 启动”的项目约定。

后续只需做一层薄参数链：

```text
main.py argument_parser
  -> run_ns3(params)
  -> conweave run.py
  -> All_Reduce_traffic_gen.py / AllToAll_traffic_gen.py
```

这不需要再改 RL core。

### 9.2 AI 流量是否天然是 RL 的优势区间

不是天然优势，但它比当前随机 CDF 流量更可能形成 RL 的可学习优势，原因是：

1. collective traffic 有 round/phase 周期和同步 burst，存在可预测的时间结构；
2. AllReduce、AllToAll 和 incast 会产生快速迁移的热点，固定 hash 或单一局部队列
   规则不一定能同时兼顾容量异构、队列趋势和未来 burst；
3. 多个 collective 与 background traffic 叠加时，最优动作可能依赖 nonlinear
   组合状态，这正是共享 scorer + learned residual 可以发挥作用的区域；
4. 当前20 us flowlet机制与按 round/chunk 产生的间隙相容，能够在 collective 阶段
   边界重新决策，而不在同一 flowlet 内制造乱序。

但当前 v8 仍是“反应式”策略。Observation 只有 destination 和每端口局部状态，
没有 collective ID、当前 round/phase、flow/chunk 剩余字节、deadline 或 coflow
进度。若不增加至少一种 phase/progress 信号，RL 只能比手工算法组合更多拥塞指标，
不能真正提前预测下一个同步 burst。

因此合理路线是先用现有 v8 在 AI-like traffic 上做 zero-shot frozen evaluation，
再根据结果决定是否增加最小的 AI-aware observation；不能预设 AI 流量一定会赢。

## 10. 与 CONGA、DRILL、ConWeave 的机会判断

### CONGA

CONGA 本身就是面向 Clos/leaf-spine、flowlet 和非对称路径设计的分布式拥塞感知
算法，并通过远端交换机反馈获得 path-level 信息。当前 v8 主要依赖 source leaf
的本地 per-egress 状态，没有 CONGA 的显式远端路径反馈。

判断：在静态非对称 + 普通随机流量下超过 CONGA 难度较高；在周期性 AI burst、
多信号组合、CONGA 反馈存在滞后的场景有中等机会。若要稳定超过，最好加入极少量
远端或 phase 信息，而不是继续堆网络层数。

### DRILL

DRILL 以 per-packet、局部队列和随机采样实现微秒级反应，在重负载和 microburst
下很强；代价是需要处理 packet reordering，并且其决策视野主要是当前局部队列。

判断：这是三个算法中相对更现实的首个竞争目标。RL flowlet 能保持顺序，并同时
利用容量、DRE、队列 EMA 和趋势；在异速链路、周期 burst、混合长短流下有
中等偏高机会。但如果 burst 比 flowlet gap 更短、需要逐包反应，DRILL 可能占优。

### ConWeave

ConWeave 是专门面向 RDMA 的细粒度重路由框架，并在可编程交换机中透明掩盖乱序。
其论文报告相对当时 SOTA 的 average/P99 FCT 最高改善 42.3%/66.8%。当前项目虽然
使用 ConWeave ns-3 代码库，但我们的里程碑实验是 DCTCP + IRN，不等于完整 RDMA
硬件语义。

判断：这是当前最难、也最需要公平性约束的基线。在 DCTCP 仿真中“跑赢某组参数”
不应直接写成优于 ConWeave 的 RDMA 设计。若要做可信结论，必须保证相同 transport、
相同 reorder/retransmission 机制、相同 topology/traffic，并使用 collective
completion time 等 AI 指标。当前把握中低。

### 总体判断

- 在合法、预先定义的非对称或 AI-like 场景中，超过三者中的至少一个：希望不小，
  当前判断为中等偏高；
- 在多负载、多种子、多 AI pattern 下普遍超过其中一个：需要 AI 参数链、消融和
  针对性训练后才能判断；
- 同时、普遍超过 CONGA、DRILL 和 ConWeave：以当前仅本地状态的 v8 来说希望不高，
  也不应作为短期报告承诺。

最好的研究叙事不是“RL 在所有网络里取代 SOTA”，而是：

> 在异构容量、同步 collective burst 和动态热点共同出现时，具有安全先验、
> 严格 per-flowlet 所有权和多信号 learned residual 的 RL，能够在不牺牲部署
> 稳定性的前提下，适应固定启发式难以统一覆盖的状态区域。

## 11. 后续实验顺序

一次只推进一个问题：

1. **学习贡献消融**：当前 v8 frozen vs prior-only；已有 ECMP，无需重训。
2. **当前场景复现**：seed 101/102 的 frozen RL 与配对 ECMP。
3. **打通 AI 参数链**：只改 parser 和 `run_ns3.py`，保证所有 traffic mode 均由
   `prisma/main.py` 启动。
4. **AI smoke**：5–10 ms AllReduce/AllToAll，只验证流量、flowlet、action 和
   completion counters。
5. **AI zero-shot**：当前 v8 checkpoint 与 ECMP/DRILL/CONGA/ConWeave 配对评估。
6. **只在 zero-shot 暴露明确缺口时训练 AI policy**，并考虑增加最小 phase/progress
   observation。

AI 场景应报告：collective completion time/iteration time、P95/P99、最慢 rank、
链路利用率、PFC/IRN/retransmission、reorder、unfinished collective，而不应只看
普通单流 FCT。

## 12. 重构范围与框架保留

相对 backup7.13 的关键差异规模：

```text
prisma/main.py                                      +194/-17
prisma/source/argument_parser.py                    +53/-6
prisma/source/forwarder.py                          +319/-99
prisma/source/trainer.py                            +80/-46
prisma/source/learner.py                            +27/-9
prisma/source/models.py                             +129/-46
prisma/source/rl_contract.py                        new
conweave-ns3-main/.../conweave-obs-manager.cc       +199/-205
conweave-ns3-main/.../conweave-obs-manager.h        +16/-6
conweave-ns3-main/.../switch-node.h                 +4/-1
```

以下核心框架文件相对 backup7.13 没有内容变化：

```text
prisma/source/replay_buffer.py
conweave-ns3-main/scratch/network-load-balance.cc
conweave-ns3-main/src/point-to-point/model/switch-node.cc
```

因此这次虽然是机制级重大重构，但仍保留了 PRISMA Agent/Forwarder/Trainer 框架、
ns3-gym 通信方式、ConWeave ns-3 主仿真入口、原 replay buffer 和主要交换机数据面。
改动集中在错误契约和 RL 决策闭环，没有推倒整个项目重写。

## 13. 参考的一手论文

- Mohammad Alizadeh et al., “CONGA: Distributed Congestion-Aware Load
  Balancing for Datacenters,” ACM SIGCOMM 2014:
  <https://people.csail.mit.edu/alizadeh/papers/conga-sigcomm14.pdf>
- Soudeh Ghorbani et al., “DRILL: Micro Load Balancing for Low-latency Data
  Center Networks,” ACM SIGCOMM 2017:
  <https://pbg.cs.illinois.edu/papers/ghorbani17drill.pdf>
- Cha Hwan Song et al., “Network Load Balancing with In-network Reordering Support
  for RDMA,” ACM SIGCOMM 2023 (ConWeave):
  <https://www.comp.nus.edu.sg/~lijl/papers/conweave-sigcomm23.pdf>
- Huimin Luo et al., “SeqBalance: Congestion-Aware Load Balancing with no
  Reordering for RoCE,” 2024:
  <https://arxiv.org/abs/2407.09808>
- Ziyang Li et al., “FLASH: Fast All-to-All Communication in GPU Clusters,”
  2025:
  <https://arxiv.org/abs/2505.09764>
