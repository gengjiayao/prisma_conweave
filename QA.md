# PRISMA x ConWeave 技术 Q&A

> 记录日期：2026-03-27

---

## Q1: 指标粒度——per-port vs per-flow，T 也应该是 per-port 吗？

### 回答

**是的，T 应该设计为 per-port。** 三个维度应保持统一粒度：

当前状态：
- qSmooth：per-port（`m_portStats[m_lastActionOutIf].qSmooth`）
- R_norm：per-port（`m_ackWins[m_lastActionOutIf].emaDup`，回退路径）
- T（待重设计）：per-port（`accTxBytes` 增量 / 时间窗口 / 带宽）

这和 RL 的动作语义一致——Agent 的动作是"选哪个 uplink 端口"，reward 应反映"选了这个端口后该端口的表现"。per-port 是动作空间的自然粒度。

---

## Q2: 如果导师要求 per-flow 跟踪，出发点是什么？效果会更好吗？

### 回答

**per-flow 跟踪的学术出发点是合理的，但在当前架构下实现成本高且不一定更好。**

#### 导师可能的出发点

1. **更精细的因果归因**：per-port 指标是该端口上所有流的聚合。如果端口 A 上有 100 条流，RL 选了把 flow X 放到 A，但 A 的 R_norm 高可能是其他 99 条流造成的，和 flow X 无关。per-flow 指标可以精确到"flow X 本身的乱序/延迟"。

2. **学术论文的标准做法**：很多 RL 路由论文（如 Teal、DOTE）用 per-flow FCT 或 per-flow delay 作为 reward，这在 review 时更有说服力。

3. **可解释性**：per-flow 指标可以直接关联到最终评估指标（FCT），链路更短更清晰。

#### 为什么在当前架构下不一定更好

1. **时间尺度问题**：RL 决策发生在 flowlet 首包时刻，此时该 flow 的后续 FCT/delay/乱序还没发生。per-flow reward 只能是延迟的——要等 flow 结束才能算 FCT，但 RL 的 step 在首包就需要 reward。这会导致 reward 严重滞后，credit assignment 困难。

2. **稀疏信号**：per-flow reward 只在 flow 结束时产生一次反馈，而不是每次决策都有。对于大流（>1MB），一次决策可能要等几百毫秒才知道结果。DQN 在如此稀疏的 reward 下学习效率很低。

3. **实现复杂度**：需要在 C++ 端维护 flow ID -> RL 决策的映射表，在 flow 完成时回溯到对应的 Agent 和 replay buffer，跨 leaf 的流还需要协调。当前 PRISMA 的 MARL 框架不原生支持这种延迟回溯。

#### 折中方案

如果导师坚持 per-flow，可以考虑：
- **per-flowlet delay**（而非 per-flow FCT）：在 flowlet 结束时（下一个 flowlet gap 到来时）计算该 flowlet 的平均包延迟。时间尺度短（毫秒级），信号密度高，且仍和具体决策直接关联。
- **per-flow FCT 作为评估指标**（而非训练 reward）：训练用 per-port 指标（信号密度高），评估用 per-flow FCT（和论文标准对齐）。

---

## Q3: 仅在 Leaf 上建 Agent，Spine 不设 Agent，是否正确？

### 回答

**完全正确，这是 Leaf-Spine 架构下的标准做法。**

#### 为什么 Spine 不需要 Agent

在 2 层 Clos（Leaf-Spine）拓扑中，数据包的路由决策只发生在一个地方：**Leaf 的 uplink 选择**。

```
Host -> Leaf (选 uplink 端口) -> Spine (无选择，直接下行) -> Leaf -> Host
```

- **Leaf -> Spine**：Leaf 有 8 个 uplink 端口连接 8 个 Spine，选哪个是核心路由决策
- **Spine -> Leaf**：每个 Spine 到目标 Leaf 只有**唯一一条 downlink**，没有选择空间。Spine 查目的 IP 就知道下发到哪个 Leaf，这是确定性的转发，不需要智能决策

给 Spine 加 Agent 是多余的——动作空间只有 1（唯一的下行端口），不存在优化空间。

#### 是否会扰乱 RL 的自举（bootstrap）

**不会。** 原因：

1. **MDP 的完整性**：从 Leaf Agent 的视角，状态 = (目的地, 各端口拥塞信息)，动作 = 选 uplink 端口，reward = 该端口的表现。这个 MDP 是自洽的，不依赖 Spine 上的决策（因为 Spine 没有决策）。

2. **状态转移的确定性**：Agent 选了端口 A（连 Spine-A），Spine-A 确定性地把包送到目标 Leaf。没有"Spine 的策略"这一不确定因素。对 Leaf Agent 来说，环境的转移动态是稳定的。

3. **和行业实践一致**：所有数据中心路由算法（ECMP、CONGA、Letflow、ConWeave）的决策点都在 Leaf。Spine 只做 ECMP 或确定性下发。Google Orion、Microsoft CONGA 论文中也是如此。

#### 更准确的说法

与其说"Spine 不需要 Agent"，不如说"在 2 层 Clos 中，路由决策空间完全在 Leaf 上"。如果是 3 层 Fat-Tree 拓扑（Leaf-Spine-Core），中间层的 Spine 可能有多条上行链路可选，那时 Spine 上的 Agent 才有意义。

---

## Q4: PRISMA 原始的 overlay 图建模和 Leaf-Spine 物理拓扑的关系？

### 回答

PRISMA 原生支持任意 overlay 图，每个节点一个 Agent。在 Leaf-Spine 适配中：

- overlay 图有 16 个节点（8 Leaf + 8 Spine）
- 但只有 8 个 Leaf 节点创建了 Agent（通过 `m_isToR` 判断）
- Spine 节点在 overlay 图中有编号（8-15），但不创建 ZMQ socket、不挂接 RL Manager
- 每个 Leaf Agent 的 overlay 邻居 = 8 个 Spine（反映 uplink 连接关系）
- 动作空间 = 8（选择 8 个 Spine 中的哪一个作为 uplink）

这个映射是正确的。overlay 图的存在只是为了让 PRISMA 框架知道"谁和谁相邻"，实际的 RL 决策完全在 Leaf 上。

---

## Q5: per-port R_norm 全端口趋同（~0.53）的原因？

### 回答

在 netload=20 的低负载下，8 个端口的 R_norm 差异仅 0.01。这有两层原因：

1. **物理对称性**：Leaf-Spine 是完全对称的拓扑，8 个 Spine 规格相同。在低负载下，没有拥塞热点，所有上行路径的质量几乎一样。

2. **随机探索的自均衡**：epsilon=1.0 时 Agent 均匀随机选端口，每个端口承受相同的流量和乱序。这反过来让 R_norm 趋同。这是一个"鸡和蛋"的问题——需要先有不均匀的端口负载才能产生差异化的 R_norm，但随机策略恰好是最均匀的。

**在高负载（netload>=50）下**，某些端口会因流量突发而出现排队和拥塞，R_norm 的端口间差异应该会放大。这也是为什么提高训练负载是当前最优先的行动。

---

## Q6: 乱序指标 R_norm 的 EMA 惯性是否是问题？

### 回答

**是一个需要关注但暂不紧急的问题。**

当前 R_norm 使用 EMA 平滑（dupAck 比例的指数移动平均），参数不详但通常 alpha=0.05~0.1。这意味着 R_norm 需要 10-20 个采样窗口才能反映策略变化。

对于 RL 学习，Agent 在 step t 选了好端口，但 R_norm 的改善要到 step t+10~20 才体现在 reward 中。这引入了 credit assignment 的延迟。

**但在当前阶段不是瓶颈**：首先需要在高负载下验证端口间 R_norm 有足够差异，然后才值得优化 EMA 响应速度。如果高负载下 R_norm 差异显著（比如 0.2 vs 0.7），即使有 EMA 延迟，DQN 也能学到。

---

## Q7: 队列指标 qSmooth 在低负载下恒为 1.0，是否意味着设计有问题？

### 回答

**设计没有问题，但在低负载下确实没有信息量。**

qSmooth 使用 ACC 阶梯打分：`qScore = 1 / (1 + L/qRef)`，其中 L 是平均队列字节，qRef 是参考阈值。低负载下 L 接近 0，qScore 接近 1。

这和现实一致——低负载下确实没有队列拥塞，qSmooth=1.0 是正确的。在高负载下（netload>=50），某些端口会出现队列积压，qSmooth 会显著低于 1.0 并产生端口间差异。

**无需修改设计，提高负载即可激活。**
