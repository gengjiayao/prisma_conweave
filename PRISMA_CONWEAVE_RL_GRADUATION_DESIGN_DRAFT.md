# 面向数据中心网络的多智能体强化学习 Flowlet 负载均衡设计与实现

> 文档状态：毕业设计工作稿
> 建立日期：2026-07-13
> 最近更新：2026-07-15
> 写作原则：正文中的设计和数字必须能够追溯到源码、实验记录或正式文献。

## 摘要

本文面向容量非对称数据中心 Leaf-Spine 网络中的 flowlet 级负载均衡问题，构建了
PRISMA Python Agent 与 ConWeave/ns-3 交换机之间的交互式强化学习闭环。系统采用
每 Leaf 一个独立 Agent，在新 flowlet 边界基于本地队列、DRE 利用率、队列趋势、
链路 headroom 与相对容量选择 uplink。为保证冻结部署安全性，策略将 RL-native
congestion prior 与有界 learned residual 组合，并修复了观测语义、transition 对齐、
flowlet 动作执行、确定性训练调度以及模型保存/加载契约。

阶段性严格配对结果表明，在 `leaf_spine_128_100G_asym_OS2`、70% 负载、DCTCP、
PFC=0、IRN=1 场景中，50 ms 轨迹训练得到的 checkpoint 无需重新训练即可在 100 ms
独立轨迹上完成 13629/13629 条流；FECMP 完成 13586/13629 条。对双方共同完成流，
RL 将 small/large 平均 FCT slowdown 分别降低 17.86%/19.20%，P99 slowdown 分别降低
27.98%/47.07%。Prior-only 消融显示平均收益主要来自安全先验，而 learned residual
在两个 seed 上进一步改善 P99/P99.9。本文结论限定于当前仿真拓扑、负载、CDF 与协议
配置；跨负载、跨拓扑、AI collective traffic 和强拥塞感知基线仍需补充验证。

## 1. 绪论

### 1.1 研究背景

待写：数据中心 Clos/Leaf-Spine 网络、ECMP 哈希碰撞和路径状态不均衡问题、flowlet 路由的基本动机。

### 1.2 研究问题

本文关注：能否在 ConWeave/ns-3 仿真环境中，让每个 Leaf 上的独立 DQN Agent 根据路径状态为新 flowlet 选择 uplink，并在容量或流量不均衡时改善 FCT。

### 1.3 主要工作

- PRISMA 与 ConWeave/ns-3 的交互式仿真闭环；
- flowlet 首包保持、每个新 flowlet 的 fresh action 与 context-owned continuation route；
- `[destination,last_action]+8*6` 的 50 维 RL-native 状态及固定量化契约；
- 融合 headroom、queue EMA、instant queue 和 queue trend 的安全奖励；
- `safe prior + bounded learned residual` 的端口置换等变 Q 网络；
- transition 对齐、Double DQN、确定性 replay-sample 调度与严格 checkpoint manifest；
- 非对称拓扑下的严格共同流评估、prior-only 消融和 50→100 ms 时间跨度泛化分析。

### 1.4 章节结构

待全文完成后补写。

## 2. 技术背景与相关工作

### 2.1 数据中心多路径负载均衡

待写：ECMP、flowlet switching、拥塞感知路由的基本机制。

### 2.2 ECMP、DRILL、CONGA 与 ConWeave

待基于论文原文和当前模拟器实现核验，不直接复用旧调研中的“首个”“空白”等判断。

### 2.3 强化学习路由

待写：PRISMA、DQN、多智能体独立学习以及与本文控制粒度最接近的工作。

### 2.4 本文问题边界

待写：本文以 flowlet uplink 选择为核心，不将所有拥塞控制、乱序恢复或交换机部署问题都归入 RL 解决范围。

## 3. 系统需求与总体架构

### 3.1 设计目标

待写：闭环有效、动作可执行、状态可区分、奖励方向一致、实验可复现。

### 3.2 仿真拓扑与模块划分

设计底稿见 `PRISMA_CONWEAVE_RL_TECHNICAL_WORK_REPORT_2026-07-12.md` 第 2 节。

### 3.3 ns-3 与 Python Agent 的交互流程

待写并配时序图：新 flowlet 到达、首包保持、观测生成、ZMQ 交互、动作应用、首包释放、后续包复用路径、经验写入和模型更新。

### 3.4 多 Agent 组织方式

待写：每个 Leaf 一个独立 Agent，Spine 不部署 Agent；说明该边界的实现依据和潜在协同限制。

## 4. 强化学习路由设计

### 4.1 决策粒度与动作空间

待写：20 us flowlet gap，新 flowlet 选择 8 个 Spine uplink 之一。

### 4.2 per-flowKey 路由驻留

早期实现存在目的地址级粘滞和额外 dwell，可能造成跨流污染并让新 flowlet 无法及时
获取动作。当前 v8 以 20 us gap 检测 flowlet 边界，每个新 flowlet 必须请求 fresh
action；同一 flowlet 的后续包由 `rl_flowlet_context` 持有路径，不再静默回退 ECMP，
额外 dwell 为 0。该设计把决策边界、路径所有权和 action sequence 对齐为同一语义。

### 4.3 观测空间

当前观测为 50 维：`[destination_overlay,last_action]+8*6`。每个 egress 的六项特征依次为
queue occupancy、queue EMA、RL-native DRE utilization、queue trend、capacity-aware
headroom 和 relative link capacity。C++ 将连续特征固定量化到 `[0,5000]`，Python 只做
一次除以 5000 的转换；`last_action` 保留用于协议诊断，但不进入 Q scorer，避免端口身份捷径。

### 4.4 奖励函数

v8 使用与当前观测同源的安全奖励：0.6×headroom、0.2×queue EMA health、
0.1×instant queue health、0.1×queue trend health。headroom 已同时表达相对链路容量和
DRE utilization；queue trend 按链路字节速率归一化。奖励不使用 ECMP/CONGA 路由标签，
乱序序号仅作为 per-flow 诊断，不参与跨 flowlet 的错误惩罚。

### 4.5 DQN 模型与训练

Q 值分解为 `V(s)+P_native(s,a)+0.1*tanh(R_theta(s,a))`。其中 `P_native` 是固定安全
先验，`R_theta` 是共享的 per-action scorer，所有 action layer 共享参数并保持端口置换
等变。训练使用对齐的 `env.step(a_t)->(s_t,a_t,r_{t+1},s_{t+1})`、经验回放、Double DQN、
精确 target 初始化和按 replay insertion 数量驱动的确定性更新调度。冻结评估固定
epsilon=0，并用 checkpoint manifest 拒绝不匹配的旧模型。

## 5. 系统实现与关键问题修复

### 5.1 首包保持与动作执行

待从 C++/Python 调用链核验后写入。

### 5.2 状态信号接入与归一化

待写：4/5 端口特征曾为零、归一化尺度不匹配及其修复。

### 5.3 模型输入支路修复

待写：端口特征支路未连接、单维 LayerNorm 消除 lastAction 信息的问题。

### 5.4 奖励时间尺度修复

待写：原吞吐指标生命周期错位、DRE 方案及带宽归一化。

### 5.5 路由一致性修复

待写：一次 RL 动作如何稳定控制整个 flowlet，避免乱序和跨流污染。

## 6. 实验设计

### 6.1 实验问题

建议围绕以下问题组织：

- RQ1：当前 RL 闭环能否学习到与链路状态相关的非均匀动作？
- RQ2：在严格配对条件下，RL 能否在容量非对称拓扑中优于 ECMP？
- RQ3：RL 与 DRILL、CONGA、ConWeave 的性能差距出现在哪些指标和流大小区间？
- RQ4：负载、容量异构程度和随机种子如何影响结论？
- RQ5：哪些设计和系统边界限制了 RL 的性能上限？
- RQ6：短轨迹训练得到的冻结策略能否无需重新训练，在更长独立轨迹上保持正确性和性能优势？

### 6.2 实验环境

待记录：服务器软硬件、操作系统、Python/TensorFlow/ns-3 版本、代码 commit/补丁、拓扑和链路参数。

### 6.3 对比方法

ECMP、DRILL、CONGA、ConWeave、RL。所有方法必须共享拓扑、流量输入、拥塞控制、缓存、PFC/IRN 和仿真窗口。

### 6.4 流量与场景

当前主场景使用 AliStorage2019 CDF、`leaf_spine_128_100G_asym_OS2` 容量非对称拓扑、
70% 负载和 seed 100/101。50 ms 用于训练闭环、冻结部署和消融；100 ms 使用独立
`T_100ms` flow file 检验冻结策略的时间跨度泛化。后续补 60%/80% 负载，其他 CDF、
AI collective traffic 和对称拓扑作为扩展场景。

### 6.5 指标与统计方法

按 1 BDP（5000 Bytes）划分 small/large flow，报告 absolute FCT 与 FCT slowdown 的
Avg、Median、P95、P99 和 P99.9，并独立报告未完成流、flowlet fresh ratio、action
sequence match、fallback 和端口最大动作占比。若两种方法完成集合不同，分位数统一在
共同完成流上计算，未完成流单列，避免删失样本让失败方法的 tail 指标虚假变好。

### 6.6 公平性与复现控制

所有对比复用相同 topology、flow file、seed、CC、buffer、PFC/IRN 和 duration；RL
加载固定 checkpoint，评估 epsilon=0、禁止保存覆盖模型。每轮保存 run manifest、
policy action diagnostics 和 output ID，并对 FCT 身份字段计算哈希。seed 变化与 flow
file 变化分开解释：前者用于执行随机性复现，后者用于轨迹泛化，二者不混为同一证据。

## 7. 实验结果与分析

### 7.1 学习闭环与训练行为

训练 output `775665073` 产生 136144 个 fresh flowlet transition 和 33526 次确定性
gradient update，loss 在早期下降后进入稳定区间，reward 从探索期低点恢复；保存的
checkpoint 在冻结 output `174791074` 中以 epsilon=0 执行 129247 次新 flowlet 决策，
action match=129246/129246、fallback=0。曲线图和分量统计由对应 TensorBoard 日志生成。

### 7.2 对称拓扑

已有阶段结果显示 RL 未优于 ECMP，但正式数字需与当前冻结版本和公平评估方式重新核验。

### 7.3 容量非对称拓扑

50 ms seed 100 场景中，full v8 output `174791074` 完成 6660/6660 条流，FECMP
output `401470137` 完成 6654/6660 条。统一比较共同完成的 6654 条流后，RL 的 small
slowdown Avg/P99 分别改善 24.67%/42.03%，large slowdown Avg/P99 分别改善
27.95%/56.47%。未完成流不进入 FECMP 分位数，作为独立失败指标保留。

### 7.4 与强基线的比较

待补 DRILL、CONGA、ConWeave 的同输入实验。

### 7.5 敏感性与消融

Prior-only output `695895724` 表明安全先验贡献了大部分平均收益；full v8 在 seed 100
上进一步改善 small/large P99 slowdown 19.10%/7.87%，large P99.9 改善 13.67%。
seed 101 的 full/prior outputs `727315208`/`822665277` 使用完全相同的 6660 条流，
对应改善为 small P99 23.4%、large P99 11.2%、large P99.9 11.0%。因此 learned
residual 的可复现贡献主要是 tail correction，而非普遍降低每条流的完成时间。

### 7.6 50→100 ms 时间跨度泛化

将 50 ms 训练得到的同一 checkpoint 直接用于 100 ms `T_100ms` flow file，不继续
训练。RL output `615298051` 完成 13629/13629 条流，fresh ratio=100%、action
match=299117/299117、fallback=0，最大端口占比 19.40%；同配置 FECMP output
`812362251` 完成 13586/13629 条，43 条未完成流均为大流。

在共同完成的 13586 条流上：

| 指标 | RL相对FECMP |
|---|---:|
| Small slowdown Avg | -17.86% |
| Small slowdown P99 | -27.98% |
| Large slowdown Avg | -19.20% |
| Large slowdown P99 | -47.07% |
| Small absolute FCT Avg | -17.97% |
| Large absolute FCT Avg | -5.16% |
| Large absolute FCT P95 | -22.84% |

按流到达时间分段后，前 50 ms 到后 50 ms 的 additive slowdown penalty 在 small
flow 上为 RL +3.067、FECMP +3.038，在 large flow 上为 RL +3.891、FECMP +3.852。
几乎相同的增量说明后半程恶化主要来自持续负载和队列积累，而不是 checkpoint 超过
训练窗口后失效。RL 的相对优势在后半程缩小，但 average 和 slowdown tail 仍优于
FECMP。完整记录见
`EXPERIMENT_615298051_812362251_TEMPORAL_GENERALIZATION_2026-07-15.md`。

## 8. 讨论

### 8.1 RL 在非对称场景中可能有效的原因

safe prior 直接利用 relative-capacity-aware headroom、DRE utilization 和队列健康度，
能够在高速链路产生热点时把 flowlet 分散到仍有 headroom 的路径。bounded residual 只能
在 prior 接近的候选动作间修正排序，因而不会恢复旧版本的端口坍缩。消融结果显示 prior
解释大部分 average 收益，learned residual 则降低高代价坏例和 P99/P99.9。该机制也
解释了当前 tail/reliability 优先、部分 median 与 large absolute P99 让步的权衡。

### 8.2 RL 难以超越 DRILL、CONGA 和 ConWeave 的原因

从可观测性、反馈时延、动作粒度、多 Agent 非平稳性、奖励目标、探索代价、协议功能和工程开销八个方面分析。每一点标明证据来自论文、源码还是本文实验。

### 8.3 有效范围与局限性

当前已验证的泛化仅是同一非对称拓扑、70% 负载、AliStorage2019 CDF、DCTCP、PFC=0、
IRN=1 下从 50 ms 训练轨迹到 100 ms 独立轨迹的时间跨度泛化。它证明策略不依赖固定
仿真窗口，不能证明跨拓扑、跨负载、跨 CDF、故障或 AI collective traffic 泛化。
100 ms 共同流结果中，RL 的 median slowdown 分别比 FECMP 高 10.33%（small）和
16.89%（large），large absolute FCT P99 高 16.22%；同时 FECMP 有 43 条大流未完成，
存在删失偏差。当前模型更接近 tail/reliability-oriented routing prototype，距离生产部署
仍缺少推理开销、控制面容错、长时间动态负载、跨规模复现和硬件约束验证。

### 8.4 后续改进

待根据实验定位具体方向，不写泛化的“更换更强模型”。候选方向包括动作屏蔽、集中训练分散执行、直接优化 FCT 的信用分配、离线训练和轻量推理。

## 9. 总结与展望

阶段性结果表明，本文已从早期训练/加载断层和端口坍缩版本推进到一个可保存、可冻结、
可严格复现的 flowlet-level safe-prior-constrained RL 原型。该策略在非对称拓扑中相对
FECMP 改善 average 和 tail slowdown，并在 50→100 ms 实验中表现出无需按仿真窗口
重新训练的受限时间跨度泛化。后续工作将补充 100 ms prior-only、负载曲线、独立训练
复现、AI collective traffic 以及 DRILL/CONGA/ConWeave 对比，逐步确定适用边界。

## 参考文献

待按学校格式统一整理，优先使用原始论文和官方实现。

## 附录 A：关键参数

待从冻结版本自动提取。

## 附录 B：实验命令与结果索引

- v8 training：`775665073`
- 50 ms full frozen RL：`174791074`
- 50 ms FECMP：`401470137`
- 50 ms prior-only：`695895724`
- seed 101 full/prior：`727315208` / `822665277`
- 100 ms full frozen RL/FECMP：`615298051` / `812362251`

## 附录 C：主要实现文件

待列出源码路径、职责和正文引用位置。
