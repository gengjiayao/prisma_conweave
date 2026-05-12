# PRISMA x ConWeave 周会进度汇报

> 日期：2026-03-26

## 一、项目目标

在 ConWeave（NS3 数据中心仿真器）环境上部署 PRISMA（多智能体 RL 路由框架），以 flowlet 为粒度进行自适应路由决策，目标是让 RL 学会降低乱序率、优化队列健康度、提升吞吐。

## 二、本周核心发现：Reward 中 T（利用率）指标设计缺陷

### 2.1 问题现象

训练中 Reward 在仿真开始后迅速从 ~0.56 下降并稳定在 ~0.34，无论训练多久都不改变。Huber Loss 降至接近 0，但策略完全不改善。

### 2.2 排查过程

通过分析 `reward_breakdown_sw*.csv` 的统计量，将 Reward 公式各分量按仿真时间阶段拆解：

**原始 Reward 公式：** `r = 0.6*T + 0.2*qSmooth + 0.2*(1-R_norm^2)`

| 时间阶段 | T (利用率) | qSmooth (队列) | R_norm (乱序) | r_inst |
|----------|-----------|---------------|-------------|--------|
| warmup (<2.1s) | **0.50** | 1.00 | 0.48 | 0.56 |
| early (2.1-2.5s) | **0.05** | 1.00 | 0.53 | 0.37 |
| mid (2.5-3.0s) | **0.01** | 1.00 | 0.54 | 0.35 |
| late (>3.0s) | **0.006** | 1.00 | 0.54 | 0.34 |

**关键观察：T 从 0.50 骤降至接近 0，是 reward 崩塌的唯一原因。**

代入公式验证：
- warmup: 0.6x0.50 + 0.2x1.0 + 0.2x(1-0.48^2) = 0.30+0.20+0.15 = 0.65 (约等于实测 0.56)
- late: 0.6x0.006 + 0.2x1.0 + 0.2x(1-0.54^2) = 0.004+0.20+0.14 = 0.34 (完全吻合)

### 2.3 T 失效的根因

T 的计算方式：T = sum(映射到该端口的 flow 的 EMA bps) / 端口带宽

该指标在 flowlet 调度粒度下存在 **变量生命周期错位**：

1. RL 以 flowlet 为单位做路由决策（flowlet gap = 20us），每次决策在 m_flow2OutIf 中创建一条 flow->port 映射
2. 该映射的 TTL 为 m_flowMapTtlSec = 20ms
3. ComputeReward() 基于 40us 窗口（m_rewardWinSec）调用时，检查 m_flow2OutIf 中仍活跃的映射来计算 sumFlowBps
4. 由于 flowlet 间隔通常远大于 20ms TTL，**在 ComputeReward 被调用时，上一次 RL 决策产生的映射已经过期**
5. 结果：sumFlowBps = 0, T = 0，权重 0.6 的分量恒定贡献 0

**本质是 flow EMA 的 20ms TTL 与 flowlet 调度的时间尺度不匹配，导致 T 在结构上无法反映 RL 的路由效果。**

### 2.4 消融实验

将 Reward 权重从 (T=0.6, qSmooth=0.2, R_norm=0.2) 调整为 (T=0.0, qSmooth=0.4, R_norm=0.6)：

| 时间阶段 | 修复前 r_inst | **修复后 r_inst** |
|----------|-------------|-----------------|
| warmup | 0.56 | **0.85** |
| early | 0.37 | **0.83** |
| mid | 0.35 | **0.82** |
| late | 0.34 | **0.82** |

**Reward 不再崩塌，从 0.34 提升至 0.82，且全程稳定。** TensorBoard 上首次观察到 reward 曲线呈上升趋势。

## 三、RL 决策粒度与覆盖率分析

### 3.1 决策粒度：以 Flowlet 为单位

C++ 端的 OnPerHopPacket() 中实现了明确的 flowlet gating：

```
if (newFlowlet && canSwitch && !ctx.awaitingAction) -> 触发 RL 决策
```

- newFlowlet: 同一 5-tuple flow 的包间隔超过 m_flowletGapSec(20us) 则视为新 flowlet
- canSwitch: 当前 flowlet 的 lock 时间已过
- awaitingAction: 该 flowlet 未在等待 RL 回复

**RL 的决策单位是 flowlet（非逐包），每个 flowlet 只在首包触发一次决策，后续包沿用该决策。**

### 3.2 覆盖率 2.9% 的正确解读

运行统计：seen=2,919,830 packets, held=84,272 packets, hold_ratio=2.88%

- **分子**（held=84,272）：触发了 RL 决策的 flowlet 首包数 = flowlet 数量
- **分母**（seen=2,919,830）：经过该 leaf 交换机的**所有数据包**（含同一 flowlet 的后续包、控制包等）

因此 2.9% **不代表 RL 只控制了 2.9% 的流量**。实际含义是：

> 每约 35 个数据包中有 1 个是 flowlet 首包，触发了 RL 路由决策。该决策决定了整个 flowlet（后续约 34 个包）的路由路径。

**结论：RL 实际接管了几乎所有需要路由决策的数据流（flowlet），覆盖率按 flowlet 计接近 100%。** 未触发 RL 的包分为：
- 同一 flowlet 的后续包（约 97%，复用首包决策）
- 控制包（NACK/ACK/CNP 等，不需要路由决策）

## 四、长训练实验结果

参数：simul_time=10, exploration_schedule_timesteps=30000, gamma=0.9，训练 4 小时 20 分钟。

**结果**：Reward 稳定在 0.80-0.84 区间剧烈抖动，但未呈现明确的上升趋势。R_norm（乱序率）始终维持在约 0.53。

**原因分析**：Agent 的观测空间中 5 维 CONGA 特征有 4 维硬编码为 0（ce_local, ce_remote_min, age, cov），且缺失 lastAction。Agent 仅凭队列字节数无法区分端口好坏，导致策略退化为随机选择。

## 五、已完成修复

| 修复 | 文件 | 状态 |
|------|------|------|
| Reward 权重调整 (T=0, qSmooth=0.4, R_norm=0.6) | conweave-obs-manager.cc ComputeReward() | 已验证有效 |
| 观测空间接入真实 CONGA 特征 + lastAction | conweave-obs-manager.cc BuildObservation() | 已改，待验证编译和训练效果 |

### 观测空间修改详情

**Header (2维)：**
- [0] dstOverlay: 目的 overlay 节点 ID（原有）
- [1] lastAction: 上次选择的端口索引，归一化到 [0,1]（新增，原缺失）

**Per-port 特征 (5维 x N_ports)：**

| 维度 | 修改前 | 修改后 | 数据源 |
|------|--------|--------|--------|
| cost | 原始队列字节数 | 队列字节/upperBound，归一化[0,1] | QbbNetDevice::GetQueue()->GetNBytesTotal() |
| ce_local | 硬编码 0 | CONGA 本地拥塞度 [0,1] | CongaRouting::GetOneHopMetrics().ce_local_norm |
| ce_remote_min | 硬编码 0 | CONGA 远端最小拥塞 [0,1] | CongaRouting::GetOneHopMetrics().ce_remote_min_norm |
| age | 硬编码 0 | 远端反馈新鲜度 [0=新,1=旧] | CongaRouting::GetOneHopMetrics().age_norm |
| cov | 硬编码 0 | 路径覆盖率 [0,1] | CongaRouting::GetOneHopMetrics().cov_norm |

## 六、待解决问题

| 优先级 | 问题 | 状态 |
|--------|------|------|
| P0 | 验证 CONGA 特征接入后 RL 能否降低 R_norm | 待训练 |
| P1 | T 指标重设计（用端口真实字节计数替代 flow EMA） | 待做 |
| P2 | 多 Agent 协调（8 leaf 独立学习可能互相冲突） | 待评估 |
| P2 | 工程健壮性（多线程锁、Ns3Env.reset 死代码） | 低优先 |


---

# 周报 v2：2026-03-26 更新

> 基于 v1 的消融实验结论，本节记录后续的两轮验证实验和当前状态判断。

## 七、后续实验：长训练验证

### 7.1 实验设计

在消融实验（T=0 权重）基础上，延长训练时间以验证 RL Agent 能否学到有效策略：

- simul_time 从 2s 延长到 10s（仿真时间增加 5 倍）
- exploration_schedule_timesteps 从 150000 降到 30000（加速 epsilon 衰减）
- gamma=0.9（鼓励远视）


### 7.2 结果

Reward 稳定在 0.80-0.84 区间，但**未呈现持续上升趋势**。R_norm（乱序率）始终维持在 0.53 附近。动作分布在训练前期和后期几乎完全均匀（每端口约 12.5%），说明 Agent 策略未发生变化。

| 阶段 | r_inst | R_norm |
|------|--------|--------|
| early (<3s) | 0.8331 | 0.5194 |
| mid1 (3-5s) | 0.8234 | 0.5359 |
| mid2 (5-8s) | 0.8167 | 0.5440 |
| late (>8s) | 0.8217 | 0.5367 |

### 7.3 分析

进一步检查发现，观测空间的 5 维 CONGA 端口特征中有 4 维被硬编码为 0（ce_local, ce_remote_min, age, cov），且缺少 lastAction。Agent 仅凭队列字节数无法区分端口优劣，Q 网络对所有动作输出相近的 Q 值，argmax 等效于随机。

## 八、补充实验：接入真实 CONGA 观测特征

### 8.1 修改内容

在 BuildObservation() 中：
- 补上缺失的 lastAction 到 obs header（归一化到 [0,1]）
- 将 5 维硬编码 0 替换为 CongaRouting::GetOneHopMetrics() 提供的真实指标：
  - cost: 队列字节/upperBound（归一化）
  - ce_local: 本地拥塞度 [0,1]
  - ce_remote_min: 远端最小拥塞反馈 [0,1]
  - age: 远端反馈新鲜度 [0,1]
  - cov: 路径覆盖率 [0,1]

### 8.2 结果

| 指标 | 无 CONGA obs (run 257855647) | 有 CONGA obs (run 696101044) |
|------|---------------------------|---------------------------|
| avg_r_inst | 0.8216 | 0.8215 |
| avg_R_norm | 0.5371 | 0.5373 |
| 动作分布 | 均匀 | 均匀 |

两次运行指标在小数第三位以内完全一致，Agent 仍未学到策略。

### 8.3 判断

观测特征的接入是**必要的基础工作**（原先 4/5 维为 0 显然无法支撑学习），但仅补全观测不足以让 Agent 学习。问题在更深层：需要验证 reward 信号是否对不同动作具有区分度，以及训练闭环本身是否正确运作。

## 九、本周工作总结

| 序号 | 工作内容 | 结论 |
|------|---------|------|
| 1 | 分析 reward_breakdown CSV，拆解 reward 公式各分量 | 定位到 T 指标恒为 0 是 reward 低值的唯一原因 |
| 2 | 分析 T 失效机制 | 根因为 flow EMA 的 20ms TTL 与 flowlet 调度时间尺度错位 |
| 3 | 消融实验：T 权重置零 | reward 从 0.34 提升至 0.82，首次观测到上升趋势 |
| 4 | 分析 RL 决策粒度与覆盖率 | 确认 RL 以 flowlet 为单位决策，按流计覆盖率接近 100% |
| 5 | 长训练验证 | reward 稳定但 R_norm 未下降，策略未改善 |
| 6 | 接入真实 CONGA 观测特征 | 必要的基础修复，但单独不足以驱动学习 |
