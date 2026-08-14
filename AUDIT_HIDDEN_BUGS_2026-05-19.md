# RL 环路深度 Audit — 隐藏 Bug 归档
> 日期：2026-05-19
> 触发：异速 spine 场景准备阶段的系统性环路审计
> 共发现 **5 个 hidden bug** + 3 个结构性弱点

---

## 摘要表

| # | 严重度 | 文件:行 | 问题 | 影响 | 状态 |
|---|---|---|---|---|---|
| **1** | 🔴 高 | `conweave-obs-manager.cc:485` | `ResolveLinkBandwidthBps` 硬编码返回 1Gbps，忽略 outIf | 异速场景下 RL 看不到链路速度差，T_port/qBdpBytes 全错 | ✅ 已修 (2026-05-19) |
| **2** | 🟡 中 | `conweave-obs-manager.cc:405-417` | `queueDerivEma` 计算 dt 时 `lastQueueSampleSec` 已被更新 → dt=0 永远 | 8 维 obs 死掉、queueDerivPenalty 永远=0 | ✅ 已修 (2026-05-19) |
| **3** | 🔴 高 | `forwarder.py:549` + `trainer.py:87` | replay buffer add 不传 ecmp_action；IL phase 又 return 短路 DQN | Design B 前 N 步 Q-network 完全没训练，"IL 保底"无效 | ✅ 已修 (2026-05-19) |
| **5** | 🟡 中 | `conweave-obs-manager.cc:222-238` | CONGA features 用 per-port bdpBytes 缩放 | bug #1 修复后，异速场景下慢链反而看起来"更空闲" | ✅ 已修 (2026-05-19) |
| **6** | 🔴 高 | `models.py DQN_buffer_model` | LayerNorm(axis=1) 用在 size=1 的 lastAction 维 → 输出永远 0 | lastAction 特征完全死掉，Bug #11 的 /8 修复也白修 | ✅ 已修 (2026-05-19) |

> Bug #4 是 `RlTimeoutFallback` 空壳函数，非 active bug，归到结构性弱点。

---

## Bug #1（已修）：ResolveLinkBandwidthBps 硬编码 1Gbps

**位置**：`conweave-ns3-main/src/opengym/model/conweave-obs-manager.cc:485-498`

**原代码**：
```cpp
double ConweaveObsManager::ResolveLinkBandwidthBps(uint32_t outIf)
{
    (void) outIf;
    return 1e9; // 1Gbps
}
```

**问题**：函数签名收 `outIf` 但实际忽略，所有端口都被当作 1Gbps。在对称拓扑下没影响（所有链路确实 1Gbps），但在异速拓扑下：
- `T_port = dre_bytes * 8 / (tau * bwBps)` — 500Mbps 端口的 T_port 永远封顶在 0.5
- `qBdpBytes = bwBps * rtt / 8` — 所有端口共用 1040 字节的 BDP scale
- RL 完全看不到链路速度异构

**修复**：
```cpp
double ConweaveObsManager::ResolveLinkBandwidthBps(uint32_t outIf)
{
    if (!m_sw || outIf == 0 || outIf >= m_sw->GetNDevices()) return 1e9;
    Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(m_sw->GetDevice(outIf));
    if (!dev) return 1e9;
    uint64_t bps = dev->GetDataRate().GetBitRate();
    return (bps > 0) ? static_cast<double>(bps) : 1e9;
}
```

**注意**：此修复**引入了 Bug #5**（CONGA feature 缩放也变成 per-port，让异速场景 obs 信号方向反了），需要一并修。

---

## Bug #2：queueDerivEma 永远是 0

**位置**：`conweave-ns3-main/src/opengym/model/conweave-obs-manager.cc:400-420`

**原代码**：
```cpp
if (q >= 0) {
  if (st.lastQueueSampleSec == 0.0) st.lastQueueSampleSec = now;
  st.queueIntBytes += double(q) * (now - st.lastQueueSampleSec);
  st.lastQueueSampleSec = now;          // ← line 405: 先更新成 now

  double alpha = 0.2;
  st.avgQueueBytes = (1.0 - alpha) * st.avgQueueBytes + alpha * double(q);

  double dt = now - st.lastQueueSampleSec;  // ← line 412: now - now = 0 !
  if (dt > 1e-9) {  // 永远不进入
    double qDeriv = (double(q) - st.lastQueueBytes) / dt;
    double derivAlpha = 0.15;
    st.queueDerivEma = (1.0 - derivAlpha) * st.queueDerivEma + derivAlpha * qDeriv;
  }
  st.lastQueueBytes = double(q);
}
```

**问题**：`lastQueueSampleSec = now` 在 line 405 已经更新，line 412 计算 `dt = now - lastQueueSampleSec` 永远是 0，`queueDerivEma` 永远停留在初始 0.0。

**下游影响**：
1. obs 第 4 维 `queueDeriv_b` 对所有端口都是 `(0+1)*0.5*bdpBytes = bdpBytes/2`，常数零信息（8 个端口×1 维死掉 = 8 维 obs 死信号）
2. reward 里 `queueDerivPenalty = 0.05 * (queueDerivEma/bdp) = 0`，"队列上升趋势"惩罚永远不触发

**修复**：把 `lastQueueSampleSec = now` 移到 queueDeriv 计算**之后**：
```cpp
if (q >= 0) {
  if (st.lastQueueSampleSec == 0.0) st.lastQueueSampleSec = now;

  double dt = now - st.lastQueueSampleSec;  // 用 OLD lastQueueSampleSec
  if (dt > 1e-9) {
    double qDeriv = (double(q) - st.lastQueueBytes) / dt;
    double derivAlpha = 0.15;
    st.queueDerivEma = (1.0 - derivAlpha) * st.queueDerivEma + derivAlpha * qDeriv;
  }

  st.queueIntBytes += double(q) * (now - st.lastQueueSampleSec);
  st.lastQueueSampleSec = now;
  double alpha = 0.2;
  st.avgQueueBytes = (1.0 - alpha) * st.avgQueueBytes + alpha * double(q);
  st.lastQueueBytes = double(q);
}
```

---

## Bug #3：IL phase 训练完全断路（双重 bug）

### 3a：ecmp_action 不传入 replay buffer

**位置**：`prisma/source/forwarder.py:549-555`

**原代码**：
```python
Agent.replay_buffer[self.index].add(
    pend["obs"],
    pend["action"],
    float(r_env),
    next_obs_for_buffer,
    done_for_buffer,
    # ← 缺第 6 个参数 ecmp_action
)
```

**问题**：`replay_buffer.add()` 定义为 `add(obs_t, action, reward, obs_tp1, done, ecmp_action=-1)`，默认 -1。

**后果**：所有 transition 的 `ecmp_action` 字段都是 -1。

### 3b：trainer.step() 在 IL phase 中 return 短路 DQN

**位置**：`prisma/source/trainer.py:73-89`

**原代码**：
```python
if in_il_phase:
    valid_mask = ecmp_actions_t >= 0
    if np.sum(valid_mask) >= max(1, Agent.batch_size // 4):
        # ... call train_il
        pass
    self.gradient_step_idx += 1
    return  # ← 即使 IL training 被跳过，DQN 也不跑
```

**叠加效果**：
- 因为 3a，`valid_mask` 全 False，`np.sum(valid_mask) < batch/4`，`train_il` 跳过
- 因为 3b，DQN training 也被 return 跳过
- **在 IL phase 期间 Q-network 完全没更新**

**解释了 Design B FCT=3.97 > Design A FCT=3.83**：
- Design A (il_phase=0)：0 步开始 DQN 训练
- Design B (il_phase=20000)：前 20000 步什么都没训练 = 浪费 40% 训练步数

### 修复

**forwarder.py:549-555**：
```python
Agent.replay_buffer[self.index].add(
    pend["obs"],
    pend["action"],
    float(r_env),
    next_obs_for_buffer,
    done_for_buffer,
    int(pend.get("ecmp_action", -1)),  # ← 加上
)
```

同样应该修以下其他 add 调用点（NN/target 路径，丢包路径）：
- `forwarder.py:407-414` (lost packet)
- `forwarder.py:420-428` (lost packet)
- `forwarder.py:772-776` (NN signaling)
- `forwarder.py:788-792` (target signaling)
- `forwarder.py:801-805` (NN real)
- `forwarder.py:813-820` (target real with prioritized)
- `forwarder.py:824-828` (target real)

**trainer.py:87**：
```python
if in_il_phase:
    valid_mask = ecmp_actions_t >= 0
    if np.sum(valid_mask) >= max(1, Agent.batch_size // 4):
        # ... train_il
    self.gradient_step_idx += 1
    # 不要 return，继续往下走 DQN training
```

或者保留 return 但在 IL phase 也并行做 DQN，看用户偏好。

---

## Bug #5：CONGA 特征用 per-port bdpBytes 缩放（异速场景方向反了）

**位置**：`conweave-ns3-main/src/opengym/model/conweave-obs-manager.cc:222-238`

**原代码**：
```cpp
double bw = ResolveLinkBandwidthBps(ifx);
double bdpBytes = std::max(1.0, bw * m_rttGuessSec / 8.0);
cost_bytes  = (uint32_t)std::lround(clamp01(m.score)              * bdpBytes);
ce_local_b  = (uint32_t)std::lround(clamp01(m.ce_local_norm)      * bdpBytes);
ce_remote_b = (uint32_t)std::lround(clamp01(m.ce_remote_min_norm) * bdpBytes);
queueDeriv_b = (uint32_t)std::lround((derivNorm + 1.0) * 0.5 * bdpBytes);
cov_b       = (uint32_t)std::lround(clamp01(m.cov_norm)           * bdpBytes);
```

**问题分析**：

`ce_local_norm` 来自 CONGA `QuantizingX`，**已经**按链路速度归一（`ratio = X*8 / (bitRate * dreTime/alpha)`）。所以同样**相对**拥塞下，慢链和快链的 ce_local_norm 相同。

但接下来 `* bdpBytes` 用了**per-port** BDP：
- 快链：50% 相对负载 → ce_local_norm=0.5 → cost_bytes = 0.5 * 1040 = 520
- 慢链：50% 相对负载 → ce_local_norm=0.5 → cost_bytes = 0.5 * 520 = 260

经 Python `/5000` 归一后：
- 快链 obs[i] ≈ 0.104
- 慢链 obs[i] ≈ 0.052

**RL 看到慢链的 obs 值更小 → 学到"慢链更空闲" → 偏好慢链**。

但实际上：50% 负载的慢链只有 250Mbps 可用，50% 负载的快链有 500Mbps 可用。**RL 应该偏好快链才对**。所以这个 obs 信号方向**反了**。

**注意**：这个 bug 在 ResolveLinkBandwidthBps 修复前不存在（因为所有 bdpBytes 都=1040 一致）。是 Bug #1 修复**引入**的。

**修复**：obs 缩放用固定值（不用 per-port bdpBytes）。reward 端的 `bw` 仍用 per-port（不变）：
```cpp
// 用固定 1Gbps BDP 作为 obs 特征 scale，避免异速带来的反向信号
const double obsBdp = 5000.0;  // matches Python feat_scale
cost_bytes  = (uint32_t)std::lround(clamp01(m.score)              * obsBdp);
ce_local_b  = (uint32_t)std::lround(clamp01(m.ce_local_norm)      * obsBdp);
ce_remote_b = (uint32_t)std::lround(clamp01(m.ce_remote_min_norm) * obsBdp);
queueDeriv_b = (uint32_t)std::lround((derivNorm + 1.0) * 0.5 * obsBdp);
cov_b       = (uint32_t)std::lround(clamp01(m.cov_norm)           * obsBdp);
```

**说明**：用 `5000` 是为了和 Python `normalize_obs` 里的 `feat_scale=5000` 配对，让归一后特征落在 [0, 1]。

---

## Bug #6：LayerNorm 把 lastAction 压成 0

**位置**：`prisma/source/models.py DQN_buffer_model` (line 271+)

**原代码**：
```python
input_size_splits = [1, 1, 40]  # dstOverlay, lastAction, CONGA features

split = SplitLayer(num_or_size_splits=input_size_splits)(inp)
for s in range(len(input_size_splits)):
    if s == 0:
        flattened_split = layers.Flatten()(one_hot_layer(split[s]))  # one-hot for dstOverlay
    else:
        flattened_split = layers.LayerNormalization(
            center=False, scale=False, trainable=False, axis=1
        )(split[s])
    out_split = layers.Dense(units=32, ...)(flattened_split)
```

**问题**：`LayerNormalization(axis=1)` 在 size-1 维上的行为：
- 输入 shape = `[batch, 1]`
- 沿 axis=1 计算 mean/std → mean=value, std=0
- 归一化：`(value - mean) / sqrt(0 + eps) = 0 / eps = 0`
- `center=False scale=False` → 没有 bias/gain 救场
- **输出永远是 0**

后果：
- `lastAction` 经 LayerNorm 后永远=0
- Dense(32) 只学到 bias，没有 lastAction 输入依赖
- Bug #11（lastAction /= 8）的"分段归一"修复**没起任何作用**

`split[2]` CONGA 特征是 40 维，LayerNorm(axis=1) 正常工作（对 40 个特征做 standardization）。

`split[0]` dstOverlay 走 one-hot，不进 LayerNorm，不受影响。

**修复（option A，最小改动）**：size=1 的 split 跳过 LayerNorm：
```python
for s in range(len(input_size_splits)):
    if input_size_splits[s] == 0: continue
    if s == 0:
        flattened_split = layers.Flatten()(one_hot_layer(split[s]))
    elif input_size_splits[s] == 1:
        flattened_split = split[s]  # 直接用，不做 LayerNorm
    else:
        flattened_split = layers.LayerNormalization(
            center=False, scale=False, trainable=False, axis=1
        )(split[s])
    out_split = layers.Dense(units=32, ...)(flattened_split)
```

**Option B**：把 lastAction 也 one-hot 化（更对称的处理），但要改 input_size_splits 和归一化逻辑，改动较大。

---

## 辅助发现（非 active bug，归档）

### 弱点 A：RlTimeoutFallback 空壳

**位置**：`conweave-ns3-main/src/point-to-point/model/switch-node.cc:135-149`

```cpp
void SwitchNode::RlTimeoutFallback()
{
    if (!m_rlMgr) return;
    Ptr<ConweaveObsManager> mgr = DynamicCast<ConweaveObsManager>(m_rlMgr);
    if (mgr) {
        // mgr->OnRlTimeout();   // ← commented out
    }
}
```

`m_rlTimeoutEv` 只在 RlRelease 里 Cancel()，**从未被 Schedule()**。如果 Python 崩溃 / ZMQ 断连，C++ hold 永久卡住，m_busy=true 后续所有包 bypass → fallback 到 ECMP。不是 active bug，但缺乏安全网。

### 弱点 B：m_egressIfs vs nexthops 顺序依赖内存分配

**位置**：`network-load-balance.cc::CalculateRoute` BFS

`nbr2if[now]` 是 `map<Ptr<Node>, Interface>`，`Ptr<Node>` 用裸指针地址比较，所以 BFS 遍历邻居的顺序是**内存地址顺序**。

在 NS-3 中，`NodeContainer::Create()` 创建节点时通常按 ID 顺序分配内存，所以**实测上**邻居遍历顺序 ≈ Node ID 顺序，跟 `m_overlayNeighbors` 的 ID 顺序一致 → m_egressIfs 与 nexthops 顺序对得上。

但这是**脆弱假设**，依赖于堆分配器行为。手动验证方法：在 C++ 加一行日志比较 m_egressIfs[i] vs nexthops[i]。

handoff 文档里"实验 A 34x"的假设把这个当主因，但即使错位，也只是**均匀置换**，不会产生 hotspot 灾难——所以不能解释 34x。

### 弱点 C：最终 Dense 用 ELU 激活

**位置**：`models.py DQN_buffer_model` line 296：

```python
out = layers.Dense(num_actions, activation='elu', ...)(out)
```

Q-network 最后一层用 ELU 而不是 linear，把 Q 值 hard-bound 到 `(-1, ∞)`。
- 我们的 reward ∈ [0, 1]，gamma=0.9 → 理论 Q ∈ [0, 10]
- 实际 Q clip 在 [-10, 10]
- ELU 不让 Q 低于 -1 → 对"高代价"状态低估

对当前 reward 设计影响小（所有 Q 都非负），但**不规范**。建议改 linear。

---

## 修复建议优先级

**立刻修（4 个 .cc/.py 文件，1 次重编）**：
1. Bug #2 — obs-manager.cc 移动 lastQueueSampleSec 赋值
2. Bug #3a — forwarder.py 加 ecmp_action 参数
3. Bug #3b — trainer.py 去掉 IL phase 的 return
4. Bug #5 — obs-manager.cc CONGA scaling 改为固定 5000
5. Bug #6 — models.py 给 size=1 split 跳过 LayerNorm

**择机改进**：
- 弱点 A — 加 RL hold timeout safety net
- 弱点 B — explicit sort m_egressIfs by node ID (or vice versa)
- 弱点 C — 最后一层改 linear activation

---

## 重要预警

**异速场景下不修 Bug #5 + #6 直接跑训练是不公平的对照**：
- Bug #5 让 RL 在 obs 里把慢链看成"更空闲"，方向反
- Bug #6 让 lastAction 死掉，sticky/inertia 信号也没了

建议：**先把 #2 #3 #5 #6 patch 一并上**，重编一次，再跑异速训练。

---

## 历史背景：14 个 bug 的累计

加上之前 handoff 里的 14 个修复历史，目前累计：
- **历史已修**：Bug #1-#11, #13（13 个，handoff 列表）
- **历史已尝试回滚**：#12, #14（2 个）
- **本次新发现**：Bug #1（隐藏的 ResolveLinkBandwidthBps，已修）、Bug #2、Bug #3（双重）、Bug #5（#1 修复引入）、Bug #6

**新发现的 4 个未修 bug 都不影响对称拓扑下的训练**（除了 Bug #6 ——但对称下没有 lastAction sticky 需求，影响小），所以历史训练数据仍然有意义。

但**异速场景**：Bug #2 #3 #5 #6 都会产生方向性误差，必须修。
