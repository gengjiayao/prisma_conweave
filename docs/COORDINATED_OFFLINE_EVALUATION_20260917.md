# 冻结条件策略：错开分配、进度预留与 ConWeave 对照

本轮在当前测试条件下找到了超过 ConWeave 的方案：通过错开不同源的初始分配，减轻目的侧路径集中，再按突发状态控制分散强度。学习版在开发覆盖与保留的 80 主机 All-to-All 上，P99 分别降低 10.58% 和 20.99%。但这部分收益主要来自机制；离线训练在六类流量聚合上额外降低 P99 2.02%，在 All-to-All 上没有额外收益。

日期：2026-09-17。本轮主实验共 7,904 次：80 次预检、384 次开发评测、2,400 次离线训练（含 96 次奖励参照）、960 次验证和 4,080 次独立测试。另有 8 次旧算法回归与 4 次新策略复跑。

正式测试共 22,804,072 条流，全部完成。所有百分比都是候选算法相对所列对照的完成时间变化，负数表示更快。

## 1. 主要结果

学习版相对当前配置 ConWeave 的结果：

- 开发覆盖条件，六类流量等权：P95 -21.64%（95% 区间 -22.05% 至 -21.10%）；P99 -17.69%（95% 区间 -18.23% 至 -17.16%）。
- 开发覆盖条件，All-to-All：P95 -9.61%（95% 区间 -9.89% 至 -9.33%）；P99 -10.58%（95% 区间 -11.16% 至 -10.04%）。
- 完全保留的 80 主机、32/256 KiB All-to-All：P95 -20.66%（95% 区间 -21.08% 至 -20.25%）；P99 -20.99%（95% 区间 -21.67% 至 -20.26%）。
- 开发覆盖条件，All-Reduce：P95 -27.39%（95% 区间 -27.74% 至 -27.11%）；P99 -24.13%（95% 区间 -25.52% 至 -22.62%）。
- 完全保留的 80 主机、5 MiB All-Reduce：P95 -19.33%（95% 区间 -21.69% 至 -16.74%）；P99 -11.70%（95% 区间 -12.73% 至 -10.67%）。

![所有算法与 ConWeave 的配对比较](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/conweave_comparisons.png)

逐流尾延迟之外，集合任务总完成时间（CCT）相对 ConWeave 的变化：

- 开发覆盖 All-to-All：-11.05%（95% 区间 -11.64% 至 -10.39%）；All-Reduce：-21.86%（95% 区间 -22.41% 至 -21.20%）。
- 未见 80 主机 All-to-All：-21.78%（95% 区间 -22.56% 至 -21.03%）；All-Reduce：-18.63%（95% 区间 -19.26% 至 -17.83%）。

上述聚合不代表每个拓扑、消息大小都胜出。按拓扑拆开的 All-to-All（包括全部开发覆盖与未见条件）：

- 对称：P95 -9.46%（95% 区间 -9.76% 至 -9.19%）；P99 -10.27%（95% 区间 -10.80% 至 -9.77%）。
- 非对称：P95 -16.24%（95% 区间 -16.57% 至 -15.88%）；P99 -16.97%（95% 区间 -17.66% 至 -16.27%）。

逐条件检查共 60 项 P95/P99 比较，其中 0 项点估计比 ConWeave 慢。这只描述当前测试矩阵，不代表任意流量都胜出。


## 2. 收益来自机制，还是训练

同一批独立测试，在开发覆盖条件上：

- Causal count control：相对 ConWeave，六类 P95/P99 -20.98% / -16.12%；All-to-All -3.98% / -3.27%；All-Reduce -28.60% / -23.63%。
- Offset allocation：相对 ConWeave，六类 P95/P99 -17.60% / -11.36%；All-to-All -9.81% / -10.59%；All-Reduce -20.78% / -14.66%。
- Progress reservations：相对 ConWeave，六类 P95/P99 -20.94% / -15.60%；All-to-All -3.22% / -2.68%；All-Reduce -27.75% / -24.02%。
- Both, fixed：相对 ConWeave，六类 P95/P99 -17.48% / -11.38%；All-to-All -9.79% / -10.71%；All-Reduce -21.09% / -15.07%。
- Conditional, fixed：相对 ConWeave，六类 P95/P99 -20.88% / -15.99%；All-to-All -9.98% / -11.24%；All-Reduce -27.65% / -23.33%。
- Conditional, learned：相对 ConWeave，六类 P95/P99 -21.64% / -17.69%；All-to-All -9.61% / -10.58%；All-Reduce -27.39% / -24.13%。
- Reward-shuffled：相对 ConWeave，六类 P95/P99 -20.88% / -15.99%；All-to-All -9.98% / -11.24%；All-Reduce -27.65% / -23.33%。

固定参数消融支持这样的解释：仅错开分配就把 All-to-All 的 P99 相对 ConWeave 的变化从计数控制的 -3.27% 推进到 -10.59%；仅加入进度预留为 -2.68%，没有显示出 All-to-All 改善。直接同时使用两种机制为 -10.71%，与仅错开分配相近。条件控制的作用主要是减少无条件分散对普通流量的代价。

学习增量需要与完整方案收益分开判断：

- 学习版相对同机制固定条件策略：P95 -0.96%（95% 区间 -1.44% 至 -0.46%）；P99 -2.02%（95% 区间 -2.68% 至 -1.28%）。
- 学习版相对奖励打乱训练：P95 -0.96%（95% 区间 -1.44% 至 -0.46%）；P99 -2.02%（95% 区间 -2.68% 至 -1.28%）。
- 学习版相对上一轮仅预留方案：P95 -0.83%（95% 区间 -1.31% 至 -0.45%）；P99 -1.87%（95% 区间 -2.43% 至 -1.27%）。
- 学习版相对最初 CABLE：P95 -6.20%（95% 区间 -6.74% 至 -5.65%）；P99 -7.23%（95% 区间 -7.94% 至 -6.53%）。
- 开发覆盖 All-to-All，学习版相对同机制固定参数：P95 +0.41%（95% 区间 +0.06% 至 +0.71%）；P99 +0.75%（95% 区间 +0.09% 至 +1.40%）。
- 保留的 80 主机 All-to-All，学习版相对同机制固定参数：P95 +0.43%（95% 区间 -0.03% 至 +0.80%）；P99 +0.86%（95% 区间 -0.02% 至 +1.62%）。

因此，本轮学习只支持较小的综合改善，不能称作 All-to-All 改善的来源。开发覆盖 All-to-All 上学习版略慢于固定版；保留条件上的点估计也略慢，但区间跨过 0。若专门优化本轮 All-to-All，固定条件版本更有依据；本报告仍按验证集已选定的统一学习模型报告全部测试，未按测试场景切换模型。

相对上一轮仅预留方案也存在代价：开发覆盖 All-Reduce 的 P95 为 +1.69%（95% 区间 +0.96% 至 +2.69%）。改进 All-to-All 不等于替代旧方案后所有指标都会更好。

同机制固定策略使用相同观测、条件函数和动作机制，只有八个参数是否优化不同。奖励打乱组使用相同扰动方向、预算和验证规则，仅打乱回报与方向的对应。若真实奖励组不能稳定超过打乱组，就不能把完整方案相对 ConWeave 的收益全部称为 RL 收益。

![学习归因](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/learning_attribution.png)

验证集统一选择真实奖励组第 8 轮、打乱组第 0 轮。每组保留三个模型，未按流量类型、拓扑或单个测试结果挑模型。

本轮打乱奖励组通过验证选模回退到初始化，因此正式测试策略与同机制固定条件策略一致；这两项对照的相同结果不能被当作两份独立证据。

## 3. 普通流量与传统基线

学习版在开发覆盖条件下，各流量类型相对 ConWeave 的 P95/P99，以及相对上一轮仅预留方案的变化：

- Uniform：对 ConWeave 为 -34.75% / -32.67%；对上一轮仅预留为 -0.63% / -1.11%。
- Storage：对 ConWeave 为 -22.33% / -12.51%；对上一轮仅预留为 +0.30% / -0.89%。
- 快速热点：对 ConWeave 为 -21.17% / -15.33%；对上一轮仅预留为 +0.83% / -0.81%。
- 突发：对 ConWeave 为 -11.68% / -8.12%；对上一轮仅预留为 -1.14% / -0.01%。
- All-to-All：对 ConWeave 为 -9.61% / -10.58%；对上一轮仅预留为 -5.86% / -7.55%。
- All-Reduce：对 ConWeave 为 -27.39% / -24.13%；对上一轮仅预留为 +1.69% / -0.66%。

六类等权，学习版与其他传统基线：

- 对 ECMP：P95 -37.81%（95% 区间 -38.24% 至 -37.39%）；P99 -39.86%（95% 区间 -40.30% 至 -39.21%）。
- 对 CONGA：P95 -26.27%（95% 区间 -26.75% 至 -25.65%）；P99 -22.84%（95% 区间 -23.58% 至 -22.21%）。

![每一类流量的结果](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/family_effects.png)

## 4. 此前最难的条件与全新条件

之前已反复分析的 48 主机、128 KiB 条件现在属于开发条件，不再称作未见条件。本轮用新的流量种子验证它；另外保留 80 主机、256 KiB 检查更大规模／消息。

- 对称、48 主机、128 KiB，Previous reservations 对 ConWeave：P95 +3.27%（95% 区间 +2.12% 至 +4.22%）；P99 +8.18%（95% 区间 +5.46% 至 +11.69%）。
- 对称、48 主机、128 KiB，Conditional, fixed 对 ConWeave：P95 -1.62%（95% 区间 -1.67% 至 -1.59%）；P99 -1.80%（95% 区间 -1.82% 至 -1.79%）。
- 对称、48 主机、128 KiB，Conditional, learned 对 ConWeave：P95 -1.62%（95% 区间 -1.66% 至 -1.59%）；P99 -1.80%（95% 区间 -1.82% 至 -1.79%）。
- 非对称、48 主机、128 KiB，Previous reservations 对 ConWeave：P95 -3.81%（95% 区间 -4.79% 至 -2.62%）；P99 +0.12%（95% 区间 -2.77% 至 +3.39%）。
- 非对称、48 主机、128 KiB，Conditional, fixed 对 ConWeave：P95 -11.02%（95% 区间 -11.23% 至 -10.75%）；P99 -11.78%（95% 区间 -12.32% 至 -11.04%）。
- 非对称、48 主机、128 KiB，Conditional, learned 对 ConWeave：P95 -11.04%（95% 区间 -11.23% 至 -10.77%）；P99 -11.89%（95% 区间 -12.19% 至 -11.61%）。
- 对称、80 主机、256 KiB，Previous reservations 对 ConWeave：P95 -4.72%（95% 区间 -5.07% 至 -4.33%）；P99 -1.44%（95% 区间 -2.47% 至 -0.63%）。
- 对称、80 主机、256 KiB，Conditional, fixed 对 ConWeave：P95 -17.96%（95% 区间 -18.27% 至 -17.53%）；P99 -17.64%（95% 区间 -18.06% 至 -17.15%）。
- 对称、80 主机、256 KiB，Conditional, learned 对 ConWeave：P95 -17.82%（95% 区间 -18.19% 至 -17.34%）；P99 -17.41%（95% 区间 -17.98% 至 -16.73%）。
- 非对称、80 主机、256 KiB，Previous reservations 对 ConWeave：P95 +1.89%（95% 区间 +0.64% 至 +3.17%）；P99 +10.82%（95% 区间 +7.83% 至 +13.64%）。
- 非对称、80 主机、256 KiB，Conditional, fixed 对 ConWeave：P95 -12.68%（95% 区间 -12.85% 至 -12.50%）；P99 -12.20%（95% 区间 -12.56% 至 -11.85%）。
- 非对称、80 主机、256 KiB，Conditional, learned 对 ConWeave：P95 -11.92%（95% 区间 -12.76% 至 -11.15%）；P99 -10.84%（95% 区间 -12.13% 至 -9.61%）。

![All-to-All 完整绝对时间](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/absolute_alltoall.png)

![All-Reduce 完整绝对时间](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/absolute_allreduce.png)

绝对时间图使用明确标注的对数纵轴，保留所有算法、条件和退化点。横轴是“参与者数／消息大小”的离散条件，不代表连续增长的单个变量。

## 5. 实现细节

**错开分配。** 每个源 leaf 按已经收到的目的前缀聚合本地分配账本，在代价接近的路径中选择容量加权分配次数最少的路径。不同源用不同的循环起始相位，减少同步选择。容量使用本地链路和已收到远端链路速率的较小值；没有其他源的实时账本或全局匹配保证。

**进度预留。** 初值按 64 KiB 估计一条新流；观察到的流连续 2 ms 没有新包时，在下一次新流选路前，用其已观察到的最大序号跨度更新 EWMA。它不是实际完成标签，暂停的长流可能被低估。预留随本地观察到的序号进展扣减，并按 20 ms 衰减。观察点在本地入队准入前，之后丢弃的包也可能使估计提前扣减；它不是已经交付给目的端的字节数。

首批同步流尚无历史样本时仍使用初值，不能知道真实的 8/128/256 KiB 消息大小。只有已收到的目的前缀可用于聚合；未收到时按目的 IP /32。

**条件策略。** 使用 200 μs 衰减的新流到达计数形成 burst；gate = burst / (burst + 可训练阈值)。gate 调节预留权重和近优窗口；反馈年龄进一步调节远端报告权重。输入中没有 All-to-All/All-Reduce 标签、参与主机数或消息大小。

模型训练八个参数：远端权重、目的预留权重、端口预留权重、端口历史权重、近优窗口、突发阈值、反馈年龄阈值、大小预测倍率。本地时延系数固定为 1。状态计数和 EWMA 随流量更新，但策略参数在整个运行期间冻结。没有在线 RL，也没有流中切换路径。

新增入口为 `COORD_ENABLE`，默认关闭；分配／预留／条件函数分别由 `COORD_BALANCE`、`COORD_WORK`、`COORD_CONDITIONAL` 控制。训练后的参数为 `COORDINATED_PATH_V1` 文本文件。共享路由框架仍加载旧线性容器，但新模块在选路前覆盖评分；这不是 DQN。

## 6. 训练、选模和测试隔离

这里的“离线”指部署前在仿真器中训练、部署时冻结；训练会生成新的策略扰动 rollout，并非只从固定日志数据集学习。8 轮 ARS、每轮 4 个正负扰动方向、选择 2 个方向更新，真实／打乱两组各 3 个优化种子。每个方向在六类流量上运行；轮次交替拓扑并使用新流量种子。

普通流量损失为 `0.20 log(mean FCT) + 0.40 log(P95) + 0.40 log(P99)`；集合通信损失为 `0.15 log(mean FCT) + 0.25 log(P95) + 0.25 log(P99) + 0.35 log(CCT)`，时间单位 ms。各类型先减去相同流量的 ConWeave 损失，再计算六类平均，加上 0.25 倍最差类型损失，以及 0.25 倍相对上一轮仅预留方案的正 P99 对数退化。reward 为该目标的负值。这些参照结果只供离线优化器使用。

- 训练种子：193301–193308。
- 开发筛查种子：193401、193402。
- 验证种子：193501、193502，比较第 0/4/8 轮；每组统一选择轮次。
- 正式测试种子：193601–193604，模型选择冻结后才生成测试流量。
- 80 主机、32/256 KiB All-to-All，以及 80 主机、5 MiB All-Reduce，从训练／开发／验证中完全排除。32 KiB 位于训练消息大小之间，256 KiB 超过训练大小；5 MiB All-Reduce 的每段仍为 64 KiB，与训练中的 48 主机／3 MiB 相同，其新变化是参与者数、总张量大小和依赖链长度。

协议在预检前冻结。本轮没有单独消融新目标函数、八个参数的边界和训练预算，不能量化它们各自的贡献。

## 7. 评测与统计口径

物理拓扑始终为 128 主机、8 leaf、8 spine；80 主机指集合任务的参与者数，不是更大物理网络。主机 1 Gbps，每 leaf 总上联 6 Gbps。对称为八条 750 Mbps，非对称为四条 1 Gbps 和四条 500 Mbps。保持既有 DCQCN、启用 IRN、关闭 PFC 和适配 ConWeave 参数，未按测试结果修改基线。图中的 ConWeave* 指当前低速测试环境的既有适配配置：reply timeout extra 200 μs、TX expiry 1,000 μs、path pause 16 μs、extra VOQ flush 64 μs、default VOQ waiting 400 μs、aging 2,000,000 μs；具体键值见冻结协议。

四类普通流量保持 50 ms 注入、负载 0.35。direct All-to-All 与 ring All-Reduce 使用真实完成依赖启动流；每个 DAG 传输是独立模拟流，因此后续阶段可为新流再次选路，但不会改动正在传输的同一流。没有评测多阶段复用同一条持久流，也没有 GPU 计算或 NCCL 开销。All-Reduce 的逐流 FCT 不计启动前依赖等待，CCT 包含完整依赖。P95/P99 包含每次运行全部流，包括机架内流。

每次运行单独计算分位数，配对同流量条件比较对数比，再对各类型内部的条件等权、各类型等权聚合；不把不同实验的流合成一个分位数。图中 95% 区间用 10,000 次流量种子块／模型种子重采样，同机制模型保留配对。固定的无相位机制可能在三个部署种子下产生完全相同结果，不能将它们视为三个独立流量样本。四个测试种子的区间不覆盖所有流量／拓扑的不确定性。

## 8. 核验、部署限制与证据

40 次预检的 201,150 次初始选路、8,074,550 次包观测已逐事件重建。8 次旧算法回归以及 4 次新机制复跑逐字节一致。另核验训练后参数和正式测试的代表性运行；全部训练更新、验证选择、测试原始分位数与统计区间分别重算，详见下面的核验文件。

该实现保存每流观测和空闲计时状态，流条目在仿真期间保留，尚未实现硬件内存上界。网络性能计入真实控制报文与传输行为，但没有计入新增算术、查表和计时器的硬件处理延迟。这些结果仅支持当前仿真条件下的比较，不能直接宣称可部署性或普遍超过 ConWeave。

- [实验与训练代码](../prisma/scripts/coordinated_offline_evaluation.py)
- [原生选路实现](../conweave-ns3-main/src/point-to-point/model/coordinated-path.inc)
- [因果重建代码](../prisma/scripts/audit_coordinated_paths.py)
- [统计与绘图代码](../prisma/scripts/analyze_coordinated_offline.py)
- [冻结协议](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/experiment/protocol.json)
- [验证选模与冻结模型](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/experiment/selection.json)
- [全量统计与图文件哈希](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/summary.json)
- [全部逐条件比较](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/condition-comparisons.json)
- [训练更新核验](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/training-verification.json)
- [验证集原始结果核验](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/selection-verification.json)
- [正式测试核验](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/test-verification.json)
- [置信区间独立重算](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/statistics-verification.json)
- [正式测试因果重建](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/test-native-audit.json)
- [逐条件统计独立重算](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/critical-statistics-verification.json)
- [路径负载与状态证据](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/decision-evidence.json)
- [本轮原生补丁](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/native-implementation.patch)
- [详细设计与限制](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/design-notes.md)

![固定展示样本的初始目的侧分配](../../../work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/analysis/assignment_balance.png)

上图固定使用首个测试种子 193601、模型 0、80 主机／256 KiB，展示每条目的侧链路的分配量相对容量比例理想值。该负载分析在仿真结束后使用真实消息大小重建，消息大小不供选路策略使用。它是预先指定的机制示例，不替代全部测试条件的统计。

完整证据目录：`/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_`。原部署模型及论文没有被替换。

复现入口：`python3 prisma/scripts/coordinated_offline_evaluation.py --root /absolute/new/experiment --stage all --workers 20`。需要项目当前原生构建与既有依赖。
