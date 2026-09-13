# PRISMA-ConWeave 拓扑与带宽迁移检查说明

> 范围：本指南保留 2026-07-19 对旧 PRISMA/v8 管道的迁移检查。物理参数检查原则仍有参考价值；其中的模型结构与旧默认值不代表当前八系数原生策略。当前入口见[选定策略说明](../prisma/experiments/path_learning/README.md)，版本与结论见[研究总记录](../RESEARCH_HISTORY.md)。

日期：2026-07-19  
适用版本：当前 `rl-core-v8` 里程碑版本  
目的：以后修改拓扑、链路带宽、链路时延或上行端口数时，避免破坏 C++—Python 状态语义、误用旧模型，或形成不公平的算法对比。

## 一、当前实验场景的实际配置

当前使用的拓扑名称是：

```text
leaf_spine_128_100G_asym_OS2
```

但名称中的 `100G` 是历史遗留命名，拓扑文件里的实际链路速率已经缩放为：

- 普通链路：`1Gbps`
- 人工降速的非对称 Leaf—Spine 链路：`500Mbps`
- 当前快慢链路带宽比：`2:1`
- 链路时延：`1000ns`

具体配置文件：

```text
conweave-ns3-main/config/leaf_spine_128_100G_asym_OS2.txt
```

Leaf 128～135 连接 Spine 136～139 的链路为 1 Gbps，连接 Spine 140～143 的链路为 500 Mbps。后续修改时，应复制出一个具有新名称的拓扑文件，不要直接覆盖当前里程碑拓扑。

## 二、一般不需要随带宽修改的内容

### 1. C++ 与 Python 之间的 5000 编码尺度

C++ 将已经归一化到 `[0,1]` 的六项端口特征量化为 `[0,5000]` 的整数，Python 再统一除以 5000：

```text
物理含义上的 0.5
→ C++ 发送 2500
→ Python 恢复成 0.5
```

`5000` 是 C++—Python 通信协议中的定点数精度，不代表 5000 字节、5000 Mbps 或任何物理阈值。链路从 1 Gbps 改成 100 Gbps 时，不应修改它。

如果未来确实要修改编码尺度，必须同时修改：

```text
conweave-ns3-main/src/opengym/model/conweave-obs-manager.cc
prisma/source/rl_contract.py
```

还要重新训练并重新验证模型。正常的拓扑和带宽迁移不需要这样做。

### 2. 已经能够自动读取或归一化的指标

当前实现直接读取每个 `QbbNetDevice` 的实际 `DataRate`，并自动计算：

- `relative_link_capacity = 当前端口带宽 / 本 Leaf 最快端口带宽`
- `dre_utilization = 近期衰减发送字节 / (时间常数 × 端口带宽)`
- `capacity_aware_headroom = relative_link_capacity × (1 - dre_utilization)`
- `queue_trend = 队列变化速度 / 端口服务速率`，再编码到 `[0,1]`
- 队列参考尺度包含 `带宽 × RTT` 形式的 BDP

因此，如果保持相同负载比例和相同快慢链路比例：

```text
1 Gbps / 500 Mbps
```

改成：

```text
100 Gbps / 50 Gbps
```

快、慢端口的 `relative_link_capacity` 仍分别为 `1.0` 和 `0.5`。RL 状态不依赖带宽单位本身。

### 3. 初次迁移时不应直接调整的算法权重

当前安全先验和 reward 使用相同的四项主要信号：

```text
0.6 × headroom
+ 0.2 × queue_ema 健康度
+ 0.1 × queue_occupancy 健康度
+ 0.1 × queue_trend 健康度
```

学习残差限制在 `[-0.1, 0.1]`。这些都是无量纲参数。只改变带宽时，应先保持不变进行验证，不要一开始同时改拓扑和算法权重，否则无法判断结果变化来自哪里。

## 三、需要随场景检查的物理和时间参数

当前仍有一组带物理尺度的参数：

| 参数 | 当前值 | 含义 | 何时重点检查 |
|---|---:|---|---|
| `m_dreTau` | 1 ms | 近期端口负载的衰减时间 | 流量突发周期或网络时间尺度明显改变 |
| `m_queueEmaTau` | 100 μs | 队列 EMA 平滑时间 | RTT、排队变化速度明显改变 |
| `m_flowletGapSec` | 20 μs | 新 Flowlet 的包间隔阈值 | RTT、链路时延、传输协议行为改变 |
| `m_rttGuessSec` | 8.32 μs | BDP 估算的 RTT 基础值 | 拓扑层数或链路时延改变 |
| `m_rewardWinSec` | 40 μs | reward 相关统计时间尺度 | RTT或流量节奏改变 |
| `m_queueRefAlphaBdp` | 1.0 | BDP 在队列参考值中的比例 | 极端带宽/RTT组合 |
| `m_queueRefBetaBuf` | 0.001 | 总缓存容量在队列参考值中的比例 | MMU 缓存规模改变 |
| `m_queueRefMinBytes` | 16 KiB | 队列参考值下限 | 低速链路、小包或小缓存拓扑 |

队列参考值为：

```text
q_ref = max(
    alpha × BDP,
    beta × MMU总缓存,
    最小参考字节数
)
```

其中当前 BDP 使用的有效时间参考为：

```text
max(m_rttGuessSec, m_rewardWinSec)
```

按当前值实际是 40 μs。因此，如果以后改变 RTT 或拓扑层数，必须重新核实 BDP 参考是否合理。

这些参数目前主要位于：

```text
conweave-ns3-main/src/opengym/model/conweave-obs-manager.h
conweave-ns3-main/src/opengym/model/conweave-obs-manager.cc
```

## 四、不同修改的难度和旧模型兼容性

### 情况 A：所有链路按相同比例放大，端口数和时延不变

示例：

```text
1 Gbps / 500 Mbps → 100 Gbps / 50 Gbps
```

难度较低：

- 不改 5000 编码尺度；
- 不改 50 维状态结构；
- 不改安全先验权重；
- 先保持时间参数不变；
- 重新生成与新带宽匹配的流量；
- 重新训练并冻结评估。

旧 checkpoint 在维度上可以加载，但没有证据证明它一定能够跨绝对带宽泛化。正式结论应来自重新训练/评估，或专门的跨带宽冻结评估。

### 情况 B：改变快慢链路比例

示例：

```text
2:1 → 4:1
```

难度较低到中等。C++ 会自动产生新的相对容量，安全先验也会自动响应，但旧模型的学习残差只在原比例上训练过，不能直接假设效果不变。

### 情况 C：改变 RTT、链路时延、拓扑层数或突发周期

难度中等。除带宽外，需要重新核实：

- Flowlet gap；
- DRE 时间常数；
- 队列 EMA 时间常数；
- reward 时间尺度；
- BDP 和队列参考值。

### 情况 D：改变每个 Leaf 的上行端口数

当前 8 个候选上行端口对应：

```text
2 + 8 × 6 = 50 维状态
```

如果上行端口数变为 `N`，状态维度会变为：

```text
2 + N × 6
```

动作空间也会从 8 变成 N。C++ 与 Python 会按公式检查新维度，但原 8 动作 checkpoint 的网络结构不兼容，必须重新建立和训练模型。

### 情况 E：改变 Leaf 或目的节点数量

`destination` 在 Python 模型中使用 one-hot 编码。目的节点数量改变时，旧模型的输入结构可能不兼容，也应按新拓扑重新训练。

## 五、实际修改入口

### 1. 新建拓扑文件

从当前拓扑复制新文件，并给出能够反映真实速率的新名称。每条链路格式为：

```text
源节点 目的节点 带宽 时延 错误率
```

例如：

```text
128 136 100Gbps 1000ns 0
128 140 50Gbps 1000ns 0
```

### 2. PRISMA 主入口参数

所有实验继续统一从：

```text
python3 main.py
```

启动。必须确认：

- `--topo` 指向新的拓扑名称；
- `--bw` 与 Host—Leaf NIC 带宽一致；
- `--netload` 的含义和原实验一致；
- 新流量文件由当前参数重新生成，而不是误用旧带宽下的流量；
- RL 与 ECMP、DRILL、CONGA、ConWeave 使用同一拓扑、流量和传输参数。

### 3. BDP 映射

当前存在手工拓扑名称到 BDP 的映射，新增拓扑名称或改变带宽/RTT后必须检查：

```text
conweave-ns3-main/run.py
conweave-ns3-main/scratch/network-load-balance.cc
```

尤其在 `--irn=1` 时，错误的 BDP 映射会影响 IRN/RDMA 行为，不能只修改拓扑文件而忽略这里。

### 4. ConWeave 对比参数

ConWeave 的超时、路径暂停和 VOQ 等时间参数对链路速率与 RTT 敏感。带宽缩放后，如果比较 ConWeave，应检查 `run.py` 中的：

```text
cwh_extra_reply_deadline
cwh_path_pause_time
cwh_extra_voq_flush_time
cwh_default_voq_waiting_time
cwh_tx_expiry_time
```

不能把未适配的 ConWeave 结果当作 RL 的胜利。CONGA、DRILL 也必须在相同拓扑和流量条件下重新跑，不能沿用旧带宽结果。

## 六、迁移后的最小验证流程

### 第一阶段：静态配置核对

1. 检查拓扑文件中的实际链路速率和时延。
2. 检查 `--bw` 与 Host—Leaf 链路一致。
3. 检查新拓扑名称已加入必要的 BDP 映射。
4. 确认每个 Leaf 的候选端口数和预期一致。

### 第二阶段：短仿真语义核对

先运行短仿真，确认：

- 每个端口读出的 `bw_bps` 正确；
- 快慢端口的 `relative_link_capacity` 正确；
- `headroom`、DRE、队列特征不是常数，也没有长期饱和在 0 或 1；
- observation 维度与 action 数一致；
- `action_mismatch = 0`；
- 正常 RL 路径的 `fallback = 0`；
- 所有流能够完成，没有异常丢包或死锁。

### 第三阶段：训练和冻结评估

1. 用新拓扑重新训练；
2. 保存模型；
3. 使用 `--train=0` 冻结加载；
4. 在独立流量或独立 seed 上评估；
5. 再进行更长时间评估；
6. 最后重跑所有对比算法。

## 七、结果解释边界

- 状态归一化使跨带宽迁移更容易，但不能代替实验验证。
- “旧模型能够成功加载”只说明输入和动作维度兼容，不代表策略已经跨带宽泛化。
- 只改变绝对带宽通常不是重构级工作。
- 改变 RTT、端口数和流量时间尺度时，才需要系统检查时间参数或重新设计模型输入结构。
- 每次只改变一类因素，并保留当前里程碑拓扑、checkpoint 和实验结果，避免无法追溯。

