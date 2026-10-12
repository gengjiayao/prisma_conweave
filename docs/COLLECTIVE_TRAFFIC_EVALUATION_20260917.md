# All-to-All 与 All-Reduce 集合通信评测

状态：正式评测与独立核验完成。1920 次运行全部成功，7,234,560 条传输全部完成；检查了 9,553,920 条依赖边。

## 实测结果与论文处理

以下变化均为 **RL 相对所列对照**的百分比，负数表示更快；按全部正式条件等权聚合，没有挑选有利子集。

- **Ring All-Reduce 达到预先规定的正面结果标准**。任务完成时间相对 CONGA 降低 **13.00%**（95% 区间 12.50–13.59%），相对适配 ConWeave 降低 **18.49%**（18.01–18.99%）。P95/P99 相对 CONGA 降低 **18.16%/16.73%**，相对适配 ConWeave 降低 **23.02%/20.67%**。
- **Direct All-to-All 未达到标准，当前 RL 明显退化**。任务完成时间相对 CONGA 增加 **26.42%**（23.14–31.33%），相对适配 ConWeave 增加 **24.63%**（21.20–29.74%）。P95/P99 分别增加 **23.01%/28.06%**、**18.93%/25.19%**。
- **同信息固定规则比 RL 更快**。RL 的任务完成时间在 All-Reduce 上高 **1.38%**（0.82–1.84%），在 All-to-All 上高 **12.55%**（8.40–16.93%）。因此 All-Reduce 对路由基线的优势属于完整方案的效果，不能归因于学习本身超过固定规则。

论文正文新增 Collective Communication 小节，加入 All-Reduce 全矩阵曲线及两类通信的任务完成时间、P95/P99 配对区间图。正文同步报告 All-to-All 负面结果和固定规则对照；附录保留 All-to-All 全矩阵图、调度语义、规模调整和核验方法。未重新训练，也没有测试后修改模型系数。

- [All-Reduce：全部算法、规模、拓扑与消息大小](../../../work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/analysis/collective_allreduce.pdf)
- [两类通信：任务完成、P95/P99 与置信区间](../../../work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/analysis/collective_intervals.pdf)
- [All-to-All：完整负面结果](../../../work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/analysis/collective_alltoall.pdf)
- [全部七个对照和三个指标的结果图](../../../work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/analysis/collective_outcomes.pdf)
- [机器可读完整统计与 95% 区间](../../../work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/analysis/collective-summary.json)
- [独立方法与统计复核](../../../work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/FORMAL_INDEPENDENT_REVIEW.md)

这些结论仅覆盖上述单个集体通信任务的网络仿真，没有 GPU 计算、NCCL 内部优化或背景业务。32/64 是参加任务的主机数；物理拓扑始终为 128 主机。后续 [All-to-All 专项诊断](OFFLINE_ALLTOALL_DIAGNOSIS_20260917.md) 通过 900 次对照运行，将主要瓶颈定位到原评分组合造成的目的侧初始路径负载不均；直接提高反馈频率或允许冻结模型重选路都未解决问题。该诊断不改变本报告的原始性能结果。

## 流量构造

网络沿用论文的 128 主机、8 leaf、8 spine 拓扑，对称和非对称网络每 leaf 上联总容量均为 6 Gbps。正式实验每个集合任务使用 32 或 64 个参与主机，每个机架参与数相同。八个种子 89001–89008 决定具体主机、rank 顺序及 0–5 μs 启动抖动；原生模拟器随机种子统一为 100。

All-to-All 采用直接交换：每个 rank 同时向其余所有 rank 发出独立传输，本地 self-copy 不占网络。每对端的数据量为 1、8、64 KiB，32/64 rank 分别产生 992/4032 条传输。

All-Reduce 采用单环 reduce-scatter 后接 all-gather。每个 rank 的输入张量为 128 KiB、1 MiB、8 MiB，划分成 P 个等大块；两阶段共 2(P−1) 轮，每轮每 rank 发出一个大小为 tensor/P 的数据块。32/64 rank 分别产生 1984/8064 条传输。第 s 轮 rank r 发送块 (r−s) mod P 给后继。符号贡献者集合验证确认最终每个 rank 获得所有完整归约块。

每个后续发送等待本 rank 上一步发送和前驱上一步传入数据均完成。模拟器在真实的发送端 QP 完成回调中释放依赖，采用接收确认后的保守完成时刻，不预先猜测开始时间，也没有每轮全网 barrier。无归约计算耗时或 GPU/NCCL 软件开销，结果仅表示该集合通信调度的网络部分。

语义参考：[MPI All-to-All](https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node101.htm)、[NVIDIA 集合通信说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

## 公平对比与指标

六种基线为 ECMP、weighted ECMP、DRILL、CONGA、LetFlow、适配后的 ConWeave；另保留同信息固定规则和三个已冻结 RL 模型。沿用论文的 DCQCN、IRN、PFC 关闭、50 MiB 缓冲及反馈/ConWeave 参数，没有新训练或测试集调参。RL 与固定规则支付相同反馈开销。

正式矩阵为 2 类通信 × 2 种参与规模 × 3 档消息大小 × 8 个种子 × 2 种拓扑 × 10 次算法/模型执行，共 1920 次。

主要指标是从任务名义开始时刻 2 秒到最后一条传输确认完成的**集合任务完成时间**，包括所有依赖等待。P95/P99 为各次执行内的传输 FCT 分位数，从该条传输实际启动时刻计算；它们不包括启动前的依赖等待。不同算法共享相同 DAG，但后续实际启动时刻自然由各自的完成事件决定。

统计先计算相同条件下 RL/基线的比值，再对 log 比值等权聚合并取指数。八个种子按整块重采样，同时重采样三个训练谱系，执行 10000 次 bootstrap；报告点态 95% 区间。没有将所有运行的流混在一起重新计算 P95/P99。

在正式比较前固定的论文纳入标准：**整个通信类型**的任务完成时间相对 CONGA 和适配 ConWeave 均至少改善 5%，相应 95% 区间完全支持改善，且总体 P95/P99 对两种基线均不恶化；全部运行须正常完成。不挑选单独有利消息大小或拓扑。无论是否纳入论文，报告所有测试类型及固定规则对照。

## 规模预检与实现验证

最初尝试 32/128 rank。128-rank All-to-All 的中档消息单次墙钟耗时增至约 140–210 秒，远高于之前的小规模配置。因此按计算成本把正式参与规模改为 32/64 rank；消息大小、模型、基线、评价指标和纳入标准不变。原规模试运行保存 671 份已完成记录及 16 份明确标记的人工中断记录，不将中断当作算法失败；正式矩阵独立重跑，不混入试运行数据。原预检清单曾将运行总数误写为 960，实际乘积为 1920，修正记录独立保留。

32 次功能预检及 40 次 64-rank 最大消息预检均完成。11 个原生路由/传输共享库与此前版本逐字节一致。新增代码只扩展流启动调度；ECMP、ConWeave 和 RL 的旧负载回归测试产生与原版逐字节相同的 FCT 输出。

每次运行核验 DAG、流量、模型、二进制和配置哈希，并逐条检查实际 start/finish、所有前置依赖、FCT、流大小及完整性。独立审阅还验证环形归约语义及配对统计实现。

## 代码和证据

- 执行：[collective_evaluation.py](../prisma/scripts/collective_evaluation.py)
- 统计与绘图：[analyze_collectives.py](../prisma/scripts/analyze_collectives.py)
- 原生依赖调度：[collective-scheduler.h](../conweave-ns3-main/scratch/collective-scheduler.h)
- 完整证据：`/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p`
- 正式协议和运行：证据目录的 `formal/protocol.json`、`formal/runs/`。
- 原规模记录：`scale-pilot-results.json`、`scale-adjustment.json`、`experiment/`。
