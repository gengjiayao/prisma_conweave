# 阶段性总结：为何 RL 超不过 ECMP（2026-05-11）

> 项目：PRISMA × ConWeave 多智能体 RL 自适应路由
> 节点：观测归一化 Bug 修复后 SNR 突破 1.0，但训练后仍输 ECMP 9-24%
> 用途：阶段总结 + 下一步重设计方向

---

## 一、当前状态

### 已完成（一年多积累）

| 模块 | 状态 |
|------|------|
| PRISMA × ConWeave 跨系统集成 | ✓ 全链路 ZMQ 闭环 |
| CONGA 五维观测接入 | ✓ |
| T_port DRE 重设计（CONGA 风格指数衰减累加器）| ✓ |
| 观测归一化错配修复（feat_scale: 1e6 → 5000）| ✓ |
| 训练稳定性工程（Huber Loss、Q 值裁剪、驻留锁）| ✓ |
| RL 环路时序审计（NS-3 + Python pending + DQN target）| ✓ 全部自洽 |
| 代码层面 RL 标准操作（argmax/td_target/loss/梯度）| ✓ 无 sign 错误 |

### 关键观察数据

| 指标 | 短训 0.5s | 长训 2s | ECMP baseline |
|------|----------|--------|--------------|
| SNR | **1.145** | 0.452 ↓ | — |
| 端口间 reward 差距 | 0.113 | 0.048 | — |
| Agent 动作分布 | 偏离均匀但学反 | 偏离均匀但更错 | 均匀（哈希）|
| <1BDP avg FCT | — | 2.014 | **1.839**（赢）|
| <1BDP 99% FCT | — | 6.815 | **5.794**（赢）|
| >1BDP avg FCT | — | 2.669 | **2.428**（赢）|

**结论**：所有工程 Bug 已修复，但 RL 在公平对比下仍稳定输 ECMP 9-24%。

---

## 二、根本原因诊断（不是 Bug，是设计层面的问题）

### 根因 1：Agent 控制权与 reward 反馈错位

```
RL-SW 统计：seen=7.3M, held=270K, hold_ratio=3.68%
```

- Agent 决策 → 只影响 3.68% 的包（flowlet 首包）
- T_port 反映 → 100% 端口字节（含 96% 不可控的 ECMP/控制包流量）
- **Agent 看到的 reward 主要被它无法控制的流量决定**

实证矛盾：Agent 选 port_22 占 19.3%（最多），但 port_22 的 T_port 反而是 0.451（最低）。说明 Agent 决策对 reward 信号的影响远小于背景流量。

### 根因 2：Per-port 本地 reward 与 FCT 全局目标不对齐

当前 reward = `0.3 × T_port + 0.3 × qSmooth + 0.4 × (1-R_norm²)`

这本质上让 Agent 学习"选负载最均衡 + 队列最浅 + 乱序最少的端口"——**这就是 ECMP 在做的事**。

数学上的必然结果：
- Agent 学得再好 → 逼近 ECMP 的均匀分散
- 但 RL 引入 flowlet 切换 → 额外乱序代价
- 所以 RL 输 ECMP 10-20% 是设计必然，不是训练不到位

### 根因 3：MARL 完全无协调机制（羊群效应）

8 个 Leaf Agent 独立训练，看不到彼此决策。当所有 Agent 都基于"port_22 看似空闲"的局部观察去选 port_22，会同时把流量打到 port_22 上——但每个 Agent 看不到其他 Agent 也在这么做。

### 根因 4：Reward feedback loop 导致信号自污染

```
Agent 选 port A 多
    → A 的 dre_bytes 累加，T_port 上升
    → 表面上"A 的 reward 高"
    → Agent 继续选 A
    → A 真的拥塞，reward 下降
    → Agent 避开 A，转向其他看似闲的 port
    → 循环
```

Agent 学到的不是"哪个 port 客观上好"，而是"哪个 port 暂时没被选过"。

---

## 三、为什么常见的修复都不解决问题

| 已尝试的修复 | 效果 |
|-------------|------|
| 加长训练时间 | SNR 反而下降（信号自污染加剧）|
| 调整 epsilon 衰减 | 不解决核心因果链断裂 |
| 修 reward 权重（T=0 消融） | 只让 reward 看起来上升，未解决根因 |
| 修 obs 归一化 | 让 Agent 能"看到"信号，但信号本身是反向的 |
| 加 sticky exploration | 减少切换乱序但不能纠正学习方向 |

---

## 四、重设计方向：让 RL 稳定超过 ECMP

### 目标设定（务实）

| 目标 | 可行性 | 时间 |
|------|--------|------|
| 稳定超过 ECMP（高负载 10-15% 改善）| **高** | 6-8 周 |
| 接近 CONGA（不输或略输 5%）| 中 | +4 周 |
| 超过 CONGA | 低 | +12 周 |
| **超过 ConWeave** | **不现实** | — |

**核心策略转变**：不是"在所有场景全面打败 SOTA"（不现实），而是"通过设计良好的 RL 在标准条件下稳定优于 ECMP"。

### 三个核心改动

#### 改动 A（必做，最关键）：T_port 只统计 RL hold 的字节

```cpp
// 当前：所有包都累加
OnMacTx(pkt) { dre_bytes += pkt_size }

// 改动：只累加 RL hold 过的 flowlet 的包
OnMacTx(pkt) {
    if (pkt.is_rl_held) {
        dre_bytes += pkt_size
    }
}
```

让 Agent 看到的 T 反映自己的决策后果，建立可学习的因果链。

工程量：1-2 天。需要在 packet metadata 加 hold 标记位（C++ NS-3）。

#### 改动 B（必做）：Warm Start from ECMP

- **Phase 1**（前 1-2 万步）：模仿学习，Q 网络拟合 ECMP 的选择
- **Phase 2**（剩余步数）：从 ECMP 策略开始 RL fine-tune

效果：保证最差等于 ECMP，best case 在难场景超越。

工程量：2-3 周。新增 IL 训练阶段。

#### 改动 C（推荐）：实验条件调整到 ECMP 真正会输的场景

- netload 从 50 提升到 **70~80**（ECMP 碰撞概率激增）
- 不是"特化"，是数据中心常见高负载

工程量：1 天（参数调整）。

### 不必改的部分（保留）

- ZMQ/Protobuf 跨系统集成
- CONGA 五维观测 + lastAction
- DRE T_port 累加器架构（只改累加条件）
- Huber Loss + Q 值裁剪 + flowlet 驻留锁
- DQN 算法本身

约 **70% 现有代码可复用**，避免一年多工作浪费。

---

## 五、时间预算（重设计完整路径）

| 阶段 | 任务 | 时间 |
|------|------|------|
| 1 | T_port hold-only 改造（C++ 修改 + 单元验证）| 1 周 |
| 2 | 单点验证（T 改造后 SNR 是否提升、动作分布是否对齐）| 1 周 |
| 3 | Warm Start IL 训练阶段实现 | 2 周 |
| 4 | netload=70+ 长训练（含 IL Phase 1 + RL Phase 2）| 2 周 |
| 5 | Test 模式对比 ECMP + 实验数据收集 | 1 周 |
| 6 | 论文撰写 | 4 周 |
| **总计** | | **~11 周（~2.5 个月）** |

时间上和硕士毕业要求兼容。

---

## 六、关键风险

| 风险 | 应对 |
|------|------|
| T_port hold-only 改造后 SNR 不升 | 改回 per-flowlet end-to-end FCT reward（工程量 +3 周）|
| Warm Start IL 训练不收敛 | 降到只用 ECMP 输出做 supervised learning，不强求 Q 值对齐 |
| netload=70 下 NS-3 仿真过慢 | 降回 1Gbps + 缩短 simul_time，保数据点完整性 |
| 即使重设计完成仍输 ECMP | 论文转向 measurement + methodology + lessons learned，仍可发表 |

---

## 七、决策建议

**不推荐**：从零完全重写。一年多工作中的工程基础（跨系统集成、CONGA 观测、DRE 思想、诊断方法论）非常有价值，没必要扔掉重来。

**推荐**：在当前代码基础上做**三处定向改动**（T hold-only + Warm Start + 实验条件），预计 2 个月内可达到"稳定超过 ECMP"。

**论文定位调整**：从"RL 路由全面超越启发式"调整为"RL 路由的设计陷阱与解决方案"——这条路径有完整故事（T 时间尺度错位 → obs normalize → 控制权-反馈错配 → 设计修复 → 超过 ECMP），学术价值更高且更可达成。

---

## 八、当前 test 训练状态

正在跑 `--train=0` 加载 `normalize_fix_long/final` 模型。

预期结果：
- FCT 比 train 模式略好（去除 5% 探索噪声）
- 但仍输 ECMP 5-15%（核心问题是 reward 设计而非探索）

是否值得跑：**让它跑完即可，不依赖结果做决策**。结果存档供答辩参考。

下一步重心：**T_port hold-only 改造**。
