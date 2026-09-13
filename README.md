# PRISMA–ConWeave RL 路由

基于 PRISMA 与 ConWeave/ns-3 的强化学习路由研究项目。当前选定策略通过全网完成回报学习初始路径评分，远端状态经实际反馈报文获取。

- [当前策略、冻结模型、结果与复现入口](prisma/experiments/path_learning/README.md)
- [研究结论、完整探索总结和原始数据索引](RESEARCH_HISTORY.md)
- [拓扑与带宽迁移说明](docs/TOPOLOGY_BANDWIDTH_MIGRATION.md)
- [SONiC 硬件资料](docs/SONIC_HARDWARE_INFO.md)
- [上游 PRISMA 框架说明](README_PRISMA.md)和 [ConWeave 仿真器说明](conweave-ns3-main/README.md)

当前运行方法以选定策略的复现入口为准，上游文档保留框架背景。新的大规模实验输出应写到代码仓库之外，例如 `/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/`。历史 trace、TensorBoard 日志和中间模型已集中归档，恢复方法见研究总记录。

仓库中保留当前模型、源码、必要输入与少量论文重算依赖。旧实验文档的历史原文可从 Git 中查阅，最新结论以研究总记录和当前策略结果为准。
