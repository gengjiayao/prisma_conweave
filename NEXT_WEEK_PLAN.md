# 下周工作计划：PRISMA x ConWeave

> 起始日期：2026-03-31
> 前置状态：reward 修复已验证（0.34→0.82），CONGA 观测已接入，但 Agent 策略尚未改善

---

## 已确定的问题

### P1 Reward 对动作缺乏区分度（最高优先级）

**现状**：Agent 选不同端口得到几乎相同的 reward，Q 网络无法学到哪个动作更好。

**证据**：
- 动作分布在训练前后完全均匀（每端口 ~12.5%）
- R_norm 始终 0.53，训练 12M 步未变化
- 即使 epsilon 衰减到 0.05，Q 网络的 argmax 仍等效随机

**根因假设**：当前 R_norm 是 **交换机级别** 的全端口 dupAck EMA。Agent 选了端口 A，但 reward 中的 R_norm 反映的是所有端口的平均乱序水平。因此"选 A"和"选 B"的 reward 没有差异。

**验证方案**：从 replay buffer 中抽样，统计不同 action 对应的 reward 分布。如果分布重合，确认此问题。

### P2 训练闭环正确性待验证

**现状**：Loss 能下降，说明 Q 网络在拟合某个目标。但需确认：
- Replay buffer 是否被正确填充（样本数、(s,a,r,s') 四元组是否完整）
- Trainer 线程是否在按预期频率更新 Q 网络
- Target network 是否在按 sync_step 频率同步

### P3 吞吐指标 T 重设计

**现状**：T 因 flow EMA 生命周期错位被置零。但优化吞吐是最终目标之一，需要重新设计可行的吞吐度量。

**方案候选**：
- 方案 A：用端口真实字节计数器（accTxBytes 增量 / 时间窗口 / 带宽）替代 flow EMA
- 方案 B：用 flowlet 级别的 FCT slowdown 作为吞吐代理指标
- 方案 C：用 CONGA score（已融合本地+远端拥塞）替代 T

---

## 下周任务排期

### Week 1 Day 1-2：确诊 reward 区分度问题

**目标**：确认不同 action 是否对应不同 reward。

**方法**：
1. 在 Forwarder.run() 中添加诊断日志：每次写入 replay buffer 时记录 (action, reward) 到 CSV
2. 短训练（simul_time=2），收集约 10K 条样本
3. 按 action 分组统计 reward 均值和方差
4. 如果各 action 的 reward 分布完全重合 → 确认 P1，进入修复

**产出**：诊断 CSV + 统计结论

### Week 1 Day 2-3：修复 reward 的动作区分度（如 P1 确认）

**目标**：让 "选端口 A 且 A 乱序少 → 高 reward"、"选端口 B 且 B 乱序多 → 低 reward" 的因果链成立。

**方案**：
1. 在 ComputeReward() 中，将 R_norm 从全端口平均改为 **当前动作端口（m_lastActionOutIf）的 per-port dupAck EMA**
2. 具体：`R_norm = m_ackWins[m_lastActionOutIf].emaDup` （已有数据，只需改取值逻辑）
3. 对应修改 reward CSV 的 R_norm 列以便验证

**产出**：修改后的 conweave-obs-manager.cc + 短训练验证

### Week 1 Day 3-4：验证 Agent 能否学到策略

**目标**：R_norm 应随训练下降，动作分布应偏离均匀。

**方案**：
1. 使用修复后的 per-port R_norm 跑长训练（simul_time=10）
2. 对比训练前后的动作分布和 R_norm 趋势
3. 如果 R_norm 下降 → 证明系统可行，进入 T 指标重设计
4. 如果 R_norm 仍不变 → 排查 P2（训练闭环）

**产出**：长训练 reward 曲线 + R_norm 趋势 + 动作分布对比

### Week 1 Day 4-5：T 指标重设计（如 Agent 已能学习）

**目标**：重新引入吞吐信号。

**方案**：用端口字节计数器计算 per-port 吞吐：
```
T_port = (accTxBytes_now - accTxBytes_prev) / (dt * bwBps)
```
在 ComputeReward 中为 m_lastActionOutIf 计算 T_port，加入 reward。

**产出**：三维 reward（T_port + qSmooth + R_norm_per_port）的训练验证

---

## 关键文件索引

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| conweave-obs-manager.cc:ComputeReward() | reward 权重覆盖 | 已完成 |
| conweave-obs-manager.cc:BuildObservation() | CONGA 特征 + lastAction | 已完成 |
| conweave-obs-manager.cc:ComputeReward() | R_norm 改为 per-port | 待做 |
| conweave-obs-manager.cc:ComputeReward() | T 重设计为 per-port 吞吐 | 待做 |
| forwarder.py:run() | 诊断日志（action, reward） | 待做 |
| WEEKLY_REPORT_0326.md | 本周周报 | 已完成 |
| KNOWN_ISSUES.md | 问题跟踪 | 已完成 |

---

# 项目总体规划与问题分析

> 更新日期：2026-03-27
> 截止目标：2026-06 前完成，在 FCT 指标上打赢 ECMP

---

## 一、当前全局认知

### 已验证的成果

1. **PRISMA-ConWeave 通信管道打通**：Python RL Agent 与 NS3 仿真器通过 ZMQ 逐步交互，每个 leaf 交换机一个独立 Agent
2. **RL 决策粒度正确**：以 flowlet 为单位决策（非逐包），flowlet 级覆盖率接近 100%
3. **Reward 的 T 指标缺陷已定位并修复**：flow EMA 生命周期与 flowlet 调度时间尺度错位，导致 T 恒为 0。消融后 reward 从 0.34 提升至 0.82
4. **观测空间已接入真实 CONGA 特征**：ce_local、ce_remote_min、age、cov + lastAction
5. **R_norm 和 qSmooth 已确认是 per-port 的**：reward 在设计上已经和动作端口绑定

### 核心瓶颈

**在 netload=20 的低负载下，8 个 uplink 端口的 per-port reward 几乎无差异**：

| 端口 | avg_reward | avg_R_norm |
|------|-----------|-----------|
| 最好 | 0.8249 | 0.5316 |
| 最差 | 0.8182 | 0.5409 |
| **差距** | **0.0067** | **0.0093** |

差距在千分位级别。DQN 在如此微弱的信号下无法有效区分动作好坏。

**这不是 bug，而是实验条件的问题**：低负载下网络空闲，任何路由策略都差不多。RL 要发挥作用，需要在网络出现拥塞（负载不均）时才有优化空间。

---

## 二、现有三个 Reward 维度的评估

### qSmooth（队列健康度，当前权重 0.4）

**设计**：per-port 的 avgQueueBytes 经 ACC 阶梯打分后 EMA 平滑。

**问题**：netload=20 下恒等于 1.0（队列几乎为空）。即使提高负载，qSmooth 作为 sigmoid 形态的指标，在队列未严重拥塞时区分度也不高。

**评估**：设计本身合理，但在低/中负载下信息量接近零。在高负载下（netload=50+）会开始有区分度。**建议保留，但降低权重**。

### R_norm（乱序率，当前权重 0.6）

**设计**：per-port 的 dupAck EMA。优先取 per-flow R_norm（但因 flow 映射过期，实际回退到 per-port R_sw）。

**问题**：
1. 低负载下所有端口的 R_norm 趋同（~0.53），无区分度
2. R_norm 是 EMA，对历史有很强的惯性。即使 Agent 在某一步选了好端口，R_norm 的变化也被 EMA 平滑掉了，Agent 感受不到即时反馈
3. R_norm~0.53 的高乱序率在低负载下反直觉——可能是因为 RL 的随机探索（频繁切端口）本身就在制造乱序。一旦策略稳定（不切端口），R_norm 应大幅下降

**评估**：方向正确（乱序是 flowlet 路由的核心指标），但 EMA 的惯性削弱了即时反馈。**建议保留，考虑减小 EMA 系数让信号更灵敏**。

### T（吞吐/利用率，当前权重 0.0）

**设计缺陷已确认**：flow EMA 20ms TTL 与 flowlet 调度时间尺度错位。

**重设计方案**：

**推荐方案：per-port 真实吞吐率**
```
T_port = (accTxBytes_now - accTxBytes_prev) / (dt_sec * bwBps / 8)
```
- `accTxBytes` 在 `EgressPortStats` 中已有维护（OnMacTx 中累加）
- `dt_sec` 是两次 ComputeReward 之间的时间间隔
- 不依赖 flow 映射表，直接从端口硬件计数器取值
- 归一化到 [0,1]，反映该端口在上一个 reward 窗口内的真实利用率

**与 R_norm 的协同**：T_port 高说明该端口在传数据（好），R_norm 低说明没乱序（好）。两者正交互补：
- 高 T + 低 R_norm = 最优（高吞吐无乱序）
- 高 T + 高 R_norm = 有吞吐但乱序多（次优）
- 低 T + 低 R_norm = 端口空闲（不好也不坏）
- 低 T + 高 R_norm = 最差（没吞吐还乱序）

---

## 三、项目不足与风险

### 不足 1：实验负载过低（netload=20）

这是当前最大的单点问题。netload=20 意味着平均链路利用率 20%，远低于拥塞阈值。所有路由策略（ECMP、CONGA、random）在此负载下表现几乎相同。

**RL 的价值在于高负载下的智能调度**。必须在 netload=50~80 下验证。

### 不足 2：缺少 ECMP baseline 对比

目前所有训练只和"随机策略"比较（epsilon=1.0 时的 reward vs 训练后）。要证明 RL 有价值，需要：
- 固定 ECMP 策略的 FCT 数据（ConWeave 自带 `--lb=ecmp` 模式）
- 固定 CONGA 策略的 FCT 数据（`--lb=conga`）
- RL 策略的 FCT 数据
- 三者在相同负载下的对比

### 不足 3：评估指标未对齐行业标准

学术论文评估路由算法通常用：
- **FCT slowdown**（Flow Completion Time / ideal FCT）分 small/medium/large flow
- **99th percentile FCT**（尾延迟）
- **throughput**（总吞吐）

当前只看 reward 内部分量，未输出 FCT。ConWeave 的 `fctAnalysis` 工具已经会生成 `*_out_fct.txt`，需要解析利用。

### 不足 4：单 episode 训练

当前每次运行只有 1 个 episode。Agent 无法跨 episode 积累经验。理想情况应该多 episode 训练，让 Agent 在不同流量模式下学习泛化策略。

### 不足 5：T 指标缺失导致 reward 与 FCT 脱节

FCT 的核心驱动因素是吞吐（T）和排队延迟（qSmooth），乱序（R_norm）是次要因素。当前 T=0 权重意味着 reward 完全不反映吞吐。即使 Agent 学到了降低 R_norm 的策略，也不能保证 FCT 改善。T 的重设计是连接 reward 与 FCT 的关键桥梁。

---

## 四、里程碑规划（2026-03-27 ~ 2026-06-01）

### Phase 1：让 RL 在高负载下学到有效策略（4月1日-4月15日）

**目标**：在 netload>=50 下，Agent 的动作分布偏离均匀，R_norm 随训练下降。

| 任务 | 内容 | 预计耗时 |
|------|------|---------|
| 1.1 | 提高训练负载到 netload=50，跑 baseline（无 RL 修改） | 1 天 |
| 1.2 | 重设计 T 为 per-port 真实吞吐率，加入 reward | 1 天 |
| 1.3 | 调整 R_norm 的 EMA 系数，加快即时反馈 | 0.5 天 |
| 1.4 | 三维 reward（T+qSmooth+R_norm）长训练，验证策略改善 | 2 天 |
| 1.5 | 如仍不学习：添加诊断日志，分析 replay buffer 样本分布 | 1 天 |

### Phase 2：对比 ECMP，证明 RL 优势（4月15日-4月30日）

**目标**：在相同负载下，RL 的 FCT < ECMP 的 FCT。

| 任务 | 内容 | 预计耗时 |
|------|------|---------|
| 2.1 | 跑 ECMP baseline（`--lb=ecmp`），记录 FCT 数据 | 1 天 |
| 2.2 | 跑 CONGA baseline（`--lb=conga`），记录 FCT 数据 | 1 天 |
| 2.3 | 用训练好的 RL 模型跑 eval（`--train=0`），记录 FCT | 1 天 |
| 2.4 | 解析 FCT 输出，绘制 small/medium/large flow 的 FCT CDF 对比图 | 2 天 |
| 2.5 | 如 RL 未打赢 ECMP：分析 FCT 差距来源，调整 reward 权重 | 3 天 |

### Phase 3：多负载/多拓扑泛化验证（5月1日-5月20日）

**目标**：RL 在多种条件下稳定优于 ECMP。

| 任务 | 内容 | 预计耗时 |
|------|------|---------|
| 3.1 | 不同负载测试（netload=30/50/70/90）| 3 天 |
| 3.2 | 不同流量模式（AliStorage/WebSearch CDF）| 2 天 |
| 3.3 | 不同拓扑规模（64/128/256 hosts）| 3 天 |
| 3.4 | 多 episode 训练，验证策略泛化 | 2 天 |

### Phase 4：论文撰写与整理（5月20日-6月1日）

| 任务 | 内容 | 预计耗时 |
|------|------|---------|
| 4.1 | 实验数据整理，绘制最终对比图表 | 3 天 |
| 4.2 | 论文/报告撰写 | 5 天 |

---

## 五、下周具体行动（3月31日-4月4日）

### Day 1：高负载 baseline

```bash
# 无 RL 修改，仅提高负载
python3 main.py --train=1 --netload=50 --buffer=50 --simul_time=5 \
  --exploration_schedule_timesteps=30000 --exploration_final_eps=0.05 \
  --replay_buffer_max_size=100000 --pfc=0 --irn=1 --cc=dctcp \
  --session_name=high_load_baseline --gamma=0.9
```

验证高负载下各端口 reward 是否出现差异。

### Day 2：T 指标重设计

在 ComputeReward() 中实现 per-port 真实吞吐率：
- 用 `accTxBytes` 增量 / 时间窗口 / 带宽
- 权重分配：T=0.3, qSmooth=0.2, R_norm=0.5

### Day 3-4：高负载长训练

用新 reward 在 netload=50 下长训练（simul_time=10），观察：
- 动作分布是否偏离均匀
- R_norm 是否下降
- reward 是否有上升趋势

### Day 5：ECMP baseline 采集

用 ConWeave 自带的 `--lb=ecmp` 跑相同负载，记录 FCT 数据作为对比基准。


---

# 全逻辑闭环收尾任务清单（2026-04-09）

## 已闭环组件

| 组件 | 状态 |
|------|------|
| NS-3 ↔ Python ZMQ 通信 | ✅ |
| 观测空间（5维 CONGA 真实特征 + lastAction） | ✅ |
| 动作空间（8端口选择 + flowlet 驻留锁） | ✅ |
| Reward（qSmooth=0.4 + R_norm=0.6，per-port） | ✅ T 消融后信号有效 |
| DQN 训练（Huber Loss + Q 值裁剪 + target sync） | ✅ |
| 模型保存/加载 | ✅ |
| Train/Test 模式切换 | ✅ |

## 待收尾任务

| # | 任务 | 预计耗时 | 状态 |
|---|------|---------|------|
| 1 | T 指标重设计：用 per-port accTxBytes 增量/时间窗口/带宽 替代失效的 flow EMA，加回 ComputeReward | 1 天 | 待做 |
| 2 | 100Gbps 环境验证：拓扑和 BDP 已校正（100Gbps, BDP=104000），需跑一次训练确认全链路正常 | 0.5 天 | 待做 |
| 3 | 高负载训练（netload=70+）：验证端口间 reward 出现差异化信号，Agent 动作分布偏离均匀 | 1-2 天 | 待做 |
| 4 | ECMP baseline FCT 采集：fecmp 模式下 netload=50/70 各跑一次，拿 FCT slowdown 数据 | 0.5 天 | 待做 |
| 5 | RL test 模式对比：加载训练模型，epsilon=0 纯策略运行，对比 ECMP 的 FCT | 0.5 天 | 依赖 #3 |

**总计约 3-5 天完成全部闭环 + 第一轮 RL vs ECMP 对比数据。**

## 闭环验收标准

- 高负载下 Agent 动作分布偏离均匀分布
- R_norm 随训练时间下降
- Test 模式下 RL 的 FCT slowdown <= ECMP（至少在 large flow 类别上）

## 风险项

| 风险 | 应对 | 追加耗时 |
|------|------|---------|
| 高负载下 per-port reward 差异仍不足 | 调大 R_norm 的 EMA 响应速度，或减小 reward 的时间平滑系数 | +2 天 |
| DQN 对微弱信号不敏感 | 考虑换 PPO 或 SAC | +1-2 周 |
| 100Gbps 下编译/运行异常 | 参照之前 BDP assert 修复经验逐项排查 | +1 天 |


---

# SONiC 交换机部署规划（2026-04-09）

## 一、部署方案选择

采用方案 A（修改 ECMP 权重），工程量最小，SSH 访问即可完成。

RL Agent 作为 SONiC 上的用户态进程运行，通过 Redis DB 读观测、通过 SAI/CLI 改 ECMP 权重。不改数据面逻辑。

```
SONiC 交换机
├─ RL Agent (Python 进程/容器)
│   ├─ 读 COUNTERS_DB → 构造观测
│   ├─ DQN 推理 → 选最优端口分布
│   └─ 写 APP_DB → 更新 ECMP 权重
├─ Redis DB
│   ├─ COUNTERS_DB（端口/队列计数器，ASIC 自动更新）
│   └─ APP_DB（路由表/ECMP 权重，Agent 写入）
└─ ASIC 数据面（按 ECMP 权重哈希转发）
```

## 二、部署阶段与耗时

### Phase 1：环境摸底（1-2 天）

SSH 进入 SONiC 系统，确认基础能力：

```bash
# 确认 SONiC 版本和 ASIC 型号
show platform summary
show version

# 确认可读的计数器
redis-cli -n 2 keys "COUNTERS:oid:*"          # 端口计数器
redis-cli -n 2 keys "COUNTERS:*:QUEUE*"       # 队列计数器

# 确认 ECMP 配置方式
show ip route                                   # 当前路由表
redis-cli -n 0 keys "ROUTE_TABLE:*"           # APP_DB 路由表项
redis-cli -n 1 keys "ASIC_STATE:SAI_OBJECT_TYPE_NEXT_HOP_GROUP*"  # ECMP 组
```

**需要确认的关键项：**
- [ ] 端口字节计数器（SAI_PORT_STAT_IF_OUT_OCTETS）可读且更新频率足够（秒级）
- [ ] 队列深度计数器可读（SAI_QUEUE_STAT_CURR_OCCUPANCY_BYTES）
- [ ] ECMP next-hop-group 的 member weight 可修改
- [ ] 丢包计数器可读（SAI_PORT_STAT_IF_OUT_DISCARDS）

### Phase 2：观测采集模块（2-3 天）

开发 Python 脚本，周期性从 COUNTERS_DB 读取并构造观测向量：

| 观测维度 | SONiC 数据源 | Redis key 示例 |
|---------|-------------|---------------|
| 队列深度（归一化） | COUNTERS_DB QUEUE | COUNTERS:oid:0x15*:SAI_QUEUE_STAT_CURR_OCCUPANCY_BYTES |
| 端口发送字节（计算利用率） | COUNTERS_DB PORT | COUNTERS:oid:0x1*:SAI_PORT_STAT_IF_OUT_OCTETS |
| 端口丢包数 | COUNTERS_DB PORT | COUNTERS:oid:0x1*:SAI_PORT_STAT_IF_OUT_DISCARDS |
| ECN 标记包数 | COUNTERS_DB PORT | COUNTERS:oid:0x1*:SAI_PORT_STAT_ECN_MARKED_PACKETS（如支持） |
| lastAction | Agent 本地维护 | 内存变量 |

注意：远端拥塞（ce_remote）在方案 A 中暂不可用，用本地指标替代。

### Phase 3：动作执行模块（2-3 天）

RL Agent 推理后输出 8 个端口的权重分布，写入 ECMP：

```python
# 伪代码
weights = agent.get_port_weights(obs)  # [0.05, 0.05, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1]
for i, member_oid in enumerate(ecmp_members):
    redis_cli.hset(f"ASIC_STATE:{member_oid}", "SAI_NEXT_HOP_GROUP_MEMBER_ATTR_WEIGHT", int(weights[i] * 100))
```

决策频率：每 100ms-1s 更新一次权重（不需要 flowlet 级，ECMP 权重是概率性的）。

### Phase 4：集成测试（2-3 天）

- 单交换机闭环测试：Agent 读计数器 → 推理 → 改权重 → 观察流量分布变化
- 用 iperf/perftest 生成可控流量，验证权重修改确实影响了转发行为
- 压力测试：Agent 崩溃时 ECMP 回退到等权重（安全兜底）

### 总耗时：1-2 周

| Phase | 耗时 | 依赖 |
|-------|------|------|
| 环境摸底 | 1-2 天 | SSH 访问 |
| 观测采集 | 2-3 天 | Phase 1 确认计数器可用 |
| 动作执行 | 2-3 天 | Phase 1 确认权重可改 |
| 集成测试 | 2-3 天 | Phase 2+3 |

## 三、SSH 访问是否足够

**基本足够。** 方案 A 全部在用户态完成：

| 操作 | 是否需要 SSH 以外的权限 |
|------|----------------------|
| 读 COUNTERS_DB | 否，redis-cli 即可 |
| 写 APP_DB（改 ECMP 权重） | 否，redis-cli 即可（需 root/admin） |
| 部署 RL Agent 进程 | 否，Python 脚本即可 |
| 部署为 Docker 容器 | 需要 docker 权限（SONiC 上通常有） |
| 修改数据面逻辑（P4） | **需要厂商 SDK**，方案 A 不涉及 |

## 四、可能需要厂商确认的信息

| 信息 | 重要性 | 替代方案 |
|------|--------|---------|
| ASIC 支持的计数器完整列表 | 高 | SSH 进去自己枚举 redis keys |
| ECMP 权重修改的生效延迟 | 中 | 实测（发流量 + 改权重 + 抓统计） |
| 队列深度计数器的采样精度 | 中 | 实测，精度不够可用端口字节增量替代 |
| 最大 ECMP group 成员数 | 低 | 通常 >= 64，8 端口远不到上限 |

## 五、部署后的观测空间对比

| 维度 | NS-3 仿真 | SONiC 交换机 | 备注 |
|------|----------|-------------|------|
| 队列深度 | 仿真 API | ASIC 硬件计数器 | 精度更高 |
| 端口利用率 | accTxBytes 增量 | 端口字节计数器增量 | 直接对应 |
| 本地拥塞 (ce_local) | CONGA DRE | 可用 ECN 标记率替代 | 语义接近 |
| 远端拥塞 (ce_remote) | CONGA piggyback | **不可用**（需 INT/自定义协议） | 方案 A 的主要损失 |
| 反馈新鲜度 (age) | CONGA age | 不适用 | 用计数器采样间隔替代 |
| 路径覆盖 (cov) | CONGA cov | 不适用 | 可省略 |
| 丢包率 | 仿真统计 | 丢包计数器 | 精度更高 |
| ECN 比例 | 无 | **新增**（仿真中没有） | 部署的独特优势 |

**部署后的观测可能比仿真更好**——硬件计数器精度高、延迟低，且有仿真中没有的 ECN 等信息。唯一损失是远端拥塞反馈，需要在后续考虑 INT 方案补回。

## 六、安全兜底设计

RL Agent 部署到生产交换机必须有安全机制：

| 风险 | 兜底措施 |
|------|---------|
| Agent 进程崩溃 | ECMP 回退到等权重（默认行为） |
| Agent 输出异常权重（某端口 100%） | 限制单端口最大权重 <= 40% |
| 推理延迟过长 | 超时后跳过本轮更新，保持上一次权重 |
| 网络性能恶化 | 监控 FCT/丢包，超阈值自动关闭 Agent 回退 ECMP |
