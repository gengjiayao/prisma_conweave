# 实验记录：50 ms Checkpoint 的 100 ms 时间跨度泛化

日期：2026-07-15  
Full frozen v8：`615298051`  
FECMP：`812362251`

## 1. 实验问题与结论边界

使用在 50 ms AliStorage2019 流量窗口上训练并保存的 `rl-core-v8` checkpoint，
不继续训练，直接在 100 ms 独立流量文件上进行冻结评估，回答：

1. 策略超过训练窗口后是否发生状态漂移、端口坍缩或动作失效；
2. 后半程 FCT 上升是否为 RL 独有，还是持续负载的共同效应；
3. 无需按仿真时长重新训练时，RL 是否仍优于同配置 FECMP。

本实验只证明同一 topology、netload、CDF 和协议配置下的时间跨度泛化，
不外推为跨拓扑、跨负载或跨流量分布的普适泛化。

## 2. 配置

- checkpoint：`rl_core_v8_safe_flowlet_train_50ms/final`
- topology：`leaf_spine_128_100G_asym_OS2`
- traffic：AliStorage2019 CDF，netload 70
- duration：100 ms，独立 `T_100ms` flow file
- seed：100
- CC：DCTCP；PFC=0；IRN=1；buffer=50
- RL：`train=0`、epsilon=0、greedy frozen
- baseline：FECMP，同一 topology/flow file/seed/network parameters

## 3. 执行正确性与完成率

RL：

- 13629/13629 条流完成；
- 299118 个新 flowlet，fresh ratio=100%；
- action match=299117/299117；
- fallback=0；
- 最大单端口占比 19.40%，无端口坍缩；
- 决策量约为 50 ms 评估的 2.3 倍，checkpoint 全程稳定。

FECMP：

- 13586/13629 条流完成；
- 43 条未完成流全部为大流，其中前 50 ms 到达 19 条、后 50 ms 到达 24 条。

为避免完成集合不同造成删失偏差，性能表统一只比较双方共同完成的 13586 条流；
FECMP 未完成的 43 条流作为独立可靠性指标报告。

## 4. 100 ms 严格共同流结果

负数表示 RL 相对 FECMP 改善。

| 指标 | RL | FECMP | RL变化 |
|---|---:|---:|---:|
| Small slowdown Avg | 4.003 | 4.873 | -17.86% |
| Small slowdown Median | 2.657 | 2.408 | +10.33% |
| Small slowdown P95 | 11.928 | 16.970 | -29.71% |
| Small slowdown P99 | 20.375 | 28.289 | -27.98% |
| Small slowdown P99.9 | 31.881 | 43.643 | -26.95% |
| Large slowdown Avg | 5.303 | 6.563 | -19.20% |
| Large slowdown Median | 4.009 | 3.430 | +16.89% |
| Large slowdown P95 | 13.296 | 22.357 | -40.53% |
| Large slowdown P99 | 19.011 | 35.920 | -47.07% |
| Large slowdown P99.9 | 27.543 | 47.577 | -42.11% |
| Small absolute FCT Avg | 250.134 us | 304.933 us | -17.97% |
| Small absolute FCT P99 | 1201.612 us | 1638.980 us | -26.69% |
| Large absolute FCT Avg | 3009.458 us | 3173.345 us | -5.16% |
| Large absolute FCT P95 | 8302.277 us | 10759.808 us | -22.84% |
| Large absolute FCT P99 | 74198.307 us | 63845.258 us | +16.22% |
| Large absolute FCT P99.9 | 114241.093 us | 141503.673 us | -19.27% |

RL 在完成率、average、P95 和 slowdown tail 上保持优势，但 median 以及 large
absolute FCT P99 存在代价。由于 FECMP 的 43 条未完成大流未进入其 FCT 分位数，
其已完成流 large P99 带有有利删失，不能单独用于宣称 FECMP 的整体尾部更好。

## 5. 前后半段分析

| 到达阶段 | RL Small Avg | FECMP Small Avg | RL变化 | RL Large Avg | FECMP Large Avg | RL变化 |
|---|---:|---:|---:|---:|---:|---:|
| 前 0–50 ms | 2.479 | 3.364 | -26.31% | 3.356 | 4.636 | -27.61% |
| 后 50–100 ms | 5.546 | 6.402 | -13.37% | 7.246 | 8.487 | -14.62% |

后半段相对前半段的 additive slowdown penalty：

- small：RL +3.067，FECMP +3.038；
- large：RL +3.891，FECMP +3.852。

两种方法的增量几乎一致，说明超过 50 ms 后的主要恶化来自持续流量注入造成的
队列积累和稳态拥塞，而不是 RL checkpoint 到达训练窗口后“过期”。RL 的相对优势
在后半段缩小，但 average 与 slowdown tail 仍优于 FECMP。

## 6. 可支持的结论

1. 当前策略不是按固定仿真时长记忆动作序列；它能够依据在线队列、DRE、trend 和
   headroom 状态持续决策。
2. 50 ms checkpoint 无需重新训练即可在 100 ms 独立轨迹上稳定运行，证明了受限的
   时间跨度泛化能力。
3. RL 不是在逻辑层面退化：所有流完成、动作契约完整、无 fallback、无端口坍缩。
4. 在严格共同流口径下，RL 的 average 和 slowdown tail 继续超过 FECMP，并避免了
   FECMP 的 43 条未完成大流。
5. 当前策略呈现 tail/reliability 优先、median 与部分 large absolute P99 让步的权衡。

推荐报告表述：

> 在相同拓扑、负载和流量分布下，本文在 50 ms 流量轨迹上训练的冻结策略无需按
> 仿真窗口重新训练，即可在 100 ms 独立轨迹上保持稳定动作契约和相对 FECMP 的
> average/tail slowdown 优势。这构成时间跨度泛化证据，但不等价于跨拓扑或跨分布泛化。

## 7. 后续诊断

下一项运行同配置 100 ms prior-only，区分长时 median/absolute-P99 权衡主要来自
固定 safe prior，还是 learned residual。完成该归因前，不直接重新训练 100 ms 模型。
