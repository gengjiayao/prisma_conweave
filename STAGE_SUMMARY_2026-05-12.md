# 阶段性总结：近期密集修改回顾（截至 2026-05-12）

> 用途：把过去几周的问题诊断、修复、当前逻辑环路统一归并
> 范围：T 指标重设计 → 观测归一化 → T 方向 → Q 网络结构 Bug
> 旧文档保留：每个子问题仍可查阅独立 MD

---

## 一、问题时间线（5 个连续 Bug，按发现顺序）

| 序号 | 发现日期 | 问题 | 关联文档 | 状态 |
|------|---------|------|----------|------|
| ① | ~2026-04 | T 指标基于 flow EMA + 20ms TTL → 与 flowlet 时间尺度错位，T ≈ 0 | T_REDESIGN_DRE.md（旧版被 patch）| ✓ 修复 |
| ② | ~2026-04 | T 第一次重设计用 ComputeReward dt → dt 由决策频率决定，T 被 clamp 到 1 | T_REDESIGN_DRE.md | ✓ 修复 |
| ③ | 2026-05 早 | 观测归一化 `feat_default=1e6` 把 CONGA 特征压缩成 ~0，Q 网络输入接近全零 | OBS_NORMALIZE_BUG.md | ✓ 修复 |
| ④ | 2026-05-12 上午 | T_port 作为 reward 正项使用，但 CONGA DRE 设计语义是 cost（越大越要避开），方向反了 | REWARD_AUDIT_2026-05-12.md | ✓ 修复 |
| ⑤ | 2026-05-12 下午 | `DQN_buffer_model` 丢弃 split[2]（40 维 CONGA 特征），Q 网络真实输入只有 17 维 | MODEL_SPLIT2_DISCARDED_BUG_2026-05-12.md | ✓ 修复（待验证）|

**关键观察**：每个 bug 都"独立合理但实际错误"，且**任何一个未修复都会让其他修复看不出效果**。

---

## 二、每个修复的"前因 → 后果 → 改动"

### ① T 指标基于 flow EMA 的时间尺度错位

- **前因**：原始 PRISMA T 用 per-flow dupAck EMA + 20ms TTL，flowlet 决策间隔远小于 20ms → 取出的是过期或空数据
- **后果**：T 恒为 0；Agent 看不到吞吐信号
- **改动**：废弃 flow-EMA T，转向 per-port 累加器思路

### ② T 用 ComputeReward 两次调用 dt 计算

- **前因**：第一次重设计想用"两次 ComputeReward 间发的字节 / dt"
- **后果**：dt 由 RL agent 决策频率决定，dt 极小 → T = bytes/(dt·bw) 被 clamp 到 1，永远饱和
- **改动**：dt 解耦于决策事件，改为 DRE 累加器

### ③ 观测归一化量纲错配

- **前因**：`feat_default = 1e6` 适用于 byte-count（数十万级），但 CONGA 五维特征实际范围 0~1040
- **后果**：归一化后所有 CONGA 特征 ≈ 0；即使 Q 网络结构正确，输入也是噪声级别
- **改动**：`feat_default = 5000.0`（match 1Gbps 下 BDP）
- **效果**：短训 SNR 从 0.21 → 1.145

详见：`OBS_NORMALIZE_BUG.md`

### ④ T_port 方向（reward vs cost）

- **前因**：T_port 用 DRE 累加器实现（CONGA 风格），但在 reward 公式中作为 **正项**（`w_util × T_port`）
- **后果**：Agent 学到"选 DRE 高的端口" = 与 CONGA 反向；DRE 高意味端口繁忙 → Agent 越往拥塞端口送
- **改动**：`r_inst = w_util × (1.0 - T_port) + ...`
- **物理语义**：现在 T_port 越小（端口越闲）→ reward 越大 → Agent 应学会选闲端口

详见：`REWARD_AUDIT_2026-05-12.md`

### ⑤ Q 网络丢弃 split[2]（最致命）

- **前因**：`models.py:DQN_buffer_model` 只用了 `split[0]`（dstOverlay）和 `split[1]`（lastAction），**忘记把 `split[2]`（40 维 CONGA）接入网络**
- **后果**：Q 网络真实输入只有 17 维（16 维 one-hot + 1 维 lastAction）。前面所有 reward 设计 / obs 归一化 / T 方向修复都是"对一个看不到拥塞状态的 Agent 调味"
- **改动**：在 `tensors_2_concat` 中加入 split[2] → LayerNorm → Dense(32) 分支
- **方法论价值**：数值正确 + Loss 收敛 ≠ 网络真正用到了输入

详见：`MODEL_SPLIT2_DISCARDED_BUG_2026-05-12.md`

---

## 三、当前逻辑环路（修完所有 5 个 bug 后）

```
                        NS-3 端                                Python Agent 端
┌─────────────────────────────────────────┐    ┌───────────────────────────────────────────┐
│                                          │    │                                            │
│  Packet TX → OnMacTx                     │    │  forwarder.run() 循环                       │
│     ├─ dre_bytes 累加（τ=1ms 指数衰减）   │    │     │                                       │
│     ├─ qSmooth = EMA(1/(1+L/qRef))       │    │     │ 接收 obs (42 维)                     │
│     ├─ R_norm = EMA(dupAck/总ack)        │    │     │                                       │
│     └─ queueDeriv = EMA(队列变化率)        │    │     │ normalize_obs (feat=5000)            │
│                                          │    │     ▼                                       │
│  Flowlet 边界 → BuildObservation         │ ━► │  Q 网络（DQN_buffer_model）                  │
│     └─ obs = [dst, lastAct, 40D CONGA]   │    │     ├─ split[0] → one_hot → Dense(16)       │
│                                          │    │     ├─ split[1] → LN → Dense(16)            │
│  ComputeReward (每端口)                   │    │     ├─ split[2] → LN → Dense(32)  ← 新接入！│
│     r_inst = 0.3×(1-T) + 0.3×qSmooth +   │ ━► │     └─ Concat → Dense(32) → Dense(32) → 8   │
│              0.4×(1-R²) - 0.05×deriv     │    │                       │                     │
│     r_level = α×r_inst + (1-α)×r_prev    │    │                       ▼                     │
│                                          │    │  ε-greedy: argmax 或 random              │
│  RL hold 标记：当 Agent 选定端口             │    │     │                                       │
│  packet 从该端口发出                       │ ◄━ │  返回 action (1 of 8 ports)              │
│                                          │    │                                            │
│  (s_{k-1}, a_{k-1}, r_k, s_k) 写入        │    │  Pending: (s_prev, a_prev) 等下次 r_k      │
│  Replay Buffer                           │    │     │                                       │
│                                          │    │     ▼                                       │
│  ←———————————————————————————————        │    │  Trainer (每 N 步)                          │
│                                          │    │     ├─ Sample batch                        │
│                                          │    │     ├─ TD: r + γ×max Q'(s')                │
│                                          │    │     ├─ Huber Loss + Q clip [-5,5]          │
│                                          │    │     └─ 梯度更新 + soft target copy           │
└─────────────────────────────────────────┘    └───────────────────────────────────────────┘
```

### 关键不变式

1. **obs 42 维**：1 (dstOverlay) + 1 (lastAction) + 40 (8 ports × 5 CONGA features)
2. **reward 范围** ∈ [0, 1]（被 (1-x) 形式约束）
3. **Q 网络真实输入维度** = 16 + 16 + 32 = 64 维 → 32 → 32 → 8 action
4. **T_port 语义**：cost（越大越拥塞，作为减项 `(1-T)` 进入 reward）
5. **EMA 平滑层数**：3 层（qScore→qSmooth、r_inst→r_level、dupAck→R_norm）

---

## 四、各分量的"信号强度 vs 期望"

| 分量 | 权重 | 当前方向 | 是否对齐期望 | 实际信号强度 |
|------|------|---------|-------------|------------|
| `(1-T_port)` | 0.3 | 端口越闲 → 越大 | ✓ 已修复 | 端口间差距 0.05~0.3，**主要驱动** |
| `qSmooth` | 0.3 | 队列越浅 → 越大 | ✓ | 低负载下几乎恒为 1.0，**沉默贡献** |
| `(1 - R_norm²)` | 0.4 | 乱序越少 → 越大 | ✓ | 端口间差距 0.01~0.03，**弱信号** |
| `-queueDerivPenalty` | -0.05 | 队列上升 → 减分 | ✓ | 影响 ≤ 0.05，**几乎无影响** |

**真正驱动 Agent 区分端口的，目前只有 T_port 一项**（占 30%）。低负载下 qSmooth 和 R_norm 都接近常数。

---

## 五、短训验证结果（742371611）

- **代码层面**：5 个 bug 全部修完，训练能跑、无报错
- **reward 信号**：方向正确（port 24 T 最低 0.49 → reward 最高 0.73）
- **SNR**：0.27~0.41（八个 leaf），未达到期望的 > 2
- **Action 分布**：仍未跟随 reward（最高 reward 的 port 24 仅被选 4.4%）
- **FCT**：与修复前几乎一致

**结论**：split[2] 修复是"代码层面 ✓"，"学习层面 ?"。0.5s 短训样本量不足（每 leaf ~9k 决策），加上 Q 网络新分支参数全是随机初始化的，无法在如此短训内表现出效果。**必须长训才能下定论**。

---

## 六、当前长训验证目标

```bash
python3 main.py --train=1 --netload=50 --buffer=50 --simul_time=2 \
  --exploration_schedule_timesteps=150000 --exploration_final_eps=0.05 \
  --replay_buffer_max_size=200000 --pfc=0 --irn=1 --cc=dctcp \
  --session_name=split2_fixed_long --gamma=0.9
```

### 验证三选一

| 长训后结果 | 解读 |
|----------|------|
| SNR 单调上升 + action 跟随 reward + FCT < ECMP | split[2] 是过去 5 个月的根本 bug，全链路打通 |
| SNR 上升但 action 仍偏离 | reward 设计层面还有问题（如 per-port 局部 reward 与全局 FCT 错位）|
| SNR 仍 < 1 | Q 网络对 40 维信号的容量/超参不够，或还有更深 bug |

---

## 七、尚未推翻 / 需要重新评估的旧判断

以下结论来自 `STAGE_REVIEW_2026-05-11.md` 第二节，在 split[2] 修复后需要重新审视：

1. **"Agent 控制权与 reward 反馈错位"（hold_ratio 3.68%）**
   - 旧推论：Agent 看到的 reward 主要被它无法控制的流量决定
   - 现状：仍可能成立，但**前提是 Q 网络真的能基于 40 维 CONGA 特征做决策**——之前网络只有 17 维，根本谈不上"利用反馈"。长训若 SNR 上升说明这条推论被部分推翻
   - 应对：观察长训 SNR 走向，再决定是否启动 `T_port hold-only` 改造

2. **"Per-port 本地 reward ≈ ECMP 数学等价"**
   - 旧推论：当前 reward 本质上在让 Agent 学 ECMP 均衡策略
   - 现状：在 17 维输入下确实如此（Agent 只能看 dst+lastAction，等价于哈希）；但 40 维输入下 Agent 可以基于端口具体状态决策——不再等价
   - 应对：长训观察 action 分布是否相对 ECMP 哈希均匀分布有偏移

3. **"MARL 羊群效应"**
   - 旧推论：各 leaf agent 独立决策无协调，会同时选同一个空闲端口
   - 现状：仍是潜在问题，但只有 Agent 能真正"看清楚"端口状态后才会暴露。如果长训中观察到端口振荡，则坐实
   - 应对：长训后看动作分布是否在 8 个端口间稳定、还是振荡式集中

4. **"Reward feedback loop 自污染"**
   - 旧推论：Agent 选 A → A 的 T 升 → reward 升 → 继续选 A → 拥塞 → reward 跌
   - 现状：T_port 方向已经反过来了——选 A 多 → A 的 T 升 → `(1-T)` 跌 → reward 跌。**这个循环现在反而是有利的**（自动避开热门端口）
   - 应对：观察长训中 action 分布是否周期振荡，如有则说明反馈过强、需加 EMA 阻尼

---

## 八、下一步路线图

```
[当前节点: 修完 5 个 bug，长训中]
       │
       ▼
  长训 split2_fixed_long
       │
   ┌───┴───┐
   │       │
SNR > 2   SNR < 1
   │       │
   ▼       ▼
检查 FCT  Q 网络
是否 <    容量/超参
ECMP      调优（dense
   │      宽度、学习率）
   ├──────┐
   │      │
FCT <    FCT ≈
ECMP     ECMP
   │      │
   ▼      ▼
✓ 论文   重设计 reward
路径    （T hold-only
打通     或 per-flowlet
        end-to-end）
```

---

## 九、文档索引（按问题归类）

| 类别 | 旧文档（保留）| 新文档 |
|------|-------------|--------|
| T 指标设计 | `T_REDESIGN_DRE.md` | —— |
| 观测归一化 | `OBS_NORMALIZE_BUG.md` | —— |
| Reward 方向 | —— | `REWARD_AUDIT_2026-05-12.md` |
| Q 网络结构 | —— | `MODEL_SPLIT2_DISCARDED_BUG_2026-05-12.md` |
| RL 环路自洽 | `RL_LOOP_AUDIT.md` | —— |
| 阶段评估 | `STAGE_REVIEW_2026-05-11.md` | **本文档** |
| 综合诊断方法论 | —— | 本文档 第七节 |

---

## 十、一句话总结

> 过去几周连续修复了 5 个独立 bug（T 时间尺度 × 2、obs 归一化、T 方向、Q 网络结构）。**前 4 个的修复价值都被第 5 个吃掉了**——网络看不到 CONGA 特征，再好的 reward 设计也是徒劳。现在 5 个全部修完，正在长训验证是否能让 RL 真正学起来；如果 SNR 在长训中能持续上升 + FCT 跑过 ECMP，过去一年的工程基础就被全部串通了。
