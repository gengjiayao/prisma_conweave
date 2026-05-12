# PRISMA x ConWeave 项目文档索引

> 最后更新：2026-04-02

---

## 文档清单

| 文件 | 内容 | 创建时间 |
|------|------|---------|
| `README.md` | 项目简介（原始） | 初始 |
| `README_PRISMA.md` | PRISMA 框架使用说明（原始） | 初始 |
| `APPROACH.md` | 项目方法论、系统架构图、关键文件索引、运行方式 | 03-25 |
| `KNOWN_ISSUES.md` | 问题跟踪总表（P0~P3 分级） | 03-25 |
| `WEEKLY_REPORT_0326.md` | 周报 v1+v2：T 缺陷发现、消融实验、CONGA 观测接入、长训练结果 | 03-26 |
| `NEXT_WEEK_PLAN.md` | 下周计划 + 项目总体规划（Phase 1~4 里程碑） | 03-26~27 |
| `QA.md` | 技术 Q&A：per-port vs per-flow、Spine 不设 Agent 等 7 个问题 | 03-27 |
| `2026-04-02DISCUSS.md` | 阶段性讨论：信噪比诊断、ECMP/CONGA/ConWeave 竞争力分析、AI workload 场景 | 04-02 |

---

## 阶段性关键发现时间线

### 发现 1（03-25）：Reward 中 T 指标的设计缺陷

**问题**：训练中 reward 稳定在 0.34 不变，Loss 降到 0 但策略不改善。

**排查方法**：将 reward 公式的三个分量（T/qSmooth/R_norm）按仿真时间阶段拆解统计。

**根因**：T（端口利用率，权重 0.6）的计算依赖 flow-to-port 映射表（TTL=20ms），但 RL 以 flowlet 为粒度决策（间隔远大于 20ms）。ComputeReward 调用时映射已过期，T 在结构上恒为 0。**flowlet 调度粒度与 reward 指标的时间尺度错位。**

**验证**：消融 T 权重后 reward 从 0.34 提升至 0.82，TensorBoard 首次观测到上升趋势。

### 发现 2（03-26）：RL 决策覆盖率的正确解读

**问题**：hold_ratio=2.9%，看似 RL 只控制了 3% 的包。

**澄清**：分子是 flowlet 首包数（触发决策的），分母是所有包（含后续包和控制包）。RL 以 flowlet 为单位决策，每个 flowlet 首包决策后后续约 34 个包复用该路径。**按 flowlet 计，RL 覆盖率接近 100%。**

### 发现 3（03-26）：观测空间 4/5 维硬编码为 0

**问题**：长训练（12M 步）后动作分布仍然均匀，Agent 什么都没学到。

**根因**：BuildObservation() 中 5 维 CONGA 特征有 4 维（ce_local, ce_remote_min, age, cov）硬编码为 0.0f，且缺失 lastAction。Agent 仅凭队列字节数无法区分端口。

**修复**：接入 CongaRouting::GetOneHopMetrics() 的真实数据 + 补上 lastAction。

### 发现 4（04-02）：低负载下 per-port reward 无区分度

**问题**：接入 CONGA 特征后仍然不学习。

**诊断**：netload=50 下 8 个端口的 avg_reward 差距仅 0.007，端口内 reward 标准差 0.045，信噪比 SNR=0.15。Q 网络将所有动作 Q 值拟合到均值 0.84，argmax 等效随机。

**根因**：对称 Leaf-Spine 拓扑 + 中等负载 = 所有端口天然趋同。RL 没有优化空间。

**方向**：提高负载至 70+（放大 ECMP 碰撞差异）或使用非均匀流量模式。

### 发现 5（04-02）：1Gbps 降速的影响

**问题**：此前为加速训练将拓扑文件中 100Gbps 改为 1Gbps。

**影响**：序列化延迟差 100 倍，BDP 差 100 倍，RTT 参数（8.32μs）与实际不匹配，训练耗时增加约 100 倍。1G 下训练的模型对 100G 无意义。

**修复**：已改回 100Gbps，BDP 校正为 500000。

---

## 当前代码改动状态（相对于 origin）

| 文件 | 改动 | 状态 |
|------|------|------|
| `conweave-obs-manager.cc` ComputeReward() | T 权重覆盖为 0，重新分配给 qSmooth(0.4) 和 R_norm(0.6) | 已验证 |
| `conweave-obs-manager.cc` BuildObservation() | 接入真实 CONGA 特征 + lastAction | 已改 |
| `conweave-obs-manager.h` EgressPortStats | 补 accAckPkts/accAckBytes 字段 | 编译修复 |
| `config/leaf_spine_128_100G_OS2.txt` | 1Gbps 改回 100Gbps | 已改 |
| `run.py` topo2bdp | BDP 从 5000 改回 500000 | 已改 |

---

## 下一步行动

1. 跑 100G ECMP baseline（验证环境 + 拿 FCT 数据）
2. 跑 100G RL 训练（验证高带宽下是否有学习信号）
3. 重设计 T 指标（per-port 真实吞吐率）
4. 高负载（netload=70+）对比实验
