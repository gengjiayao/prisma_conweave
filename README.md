# PRISMA–ConWeave RL 路由

基于 PRISMA 与 ConWeave/ns-3 的强化学习路由研究项目。当前选定策略通过全网完成回报学习初始路径评分，远端状态经实际反馈报文获取。

- [当前策略、冻结模型、结果与复现入口](prisma/experiments/path_learning/README.md)
- [RL 路由技术总结：状态、动作、奖励、训练与最终效果](docs/RL_ROUTING_TECHNICAL_SUMMARY.md)
- [2026-09-16 实测：冻结 RL 与 ConWeave 对照](docs/RL_VS_CONWEAVE_COMPARISON.md)
- [2026-09-17：NSDI 格式论文更新与 2560 次新评测](docs/NSDI_STYLE_PAPER_UPDATE_20260917.md)
- [2026-09-17：All-to-All / All-Reduce 集合通信评测](docs/COLLECTIVE_TRAFFIC_EVALUATION_20260917.md)
- [2026-09-17：传输期间学习的在线路由重设计与独立对照](docs/ONLINE_ROUTING_REDESIGN_20260917.md)
- [2026-09-17：未完成流成本驱动的多步在线训练与配对消融](docs/FLOW_ACTOR_CRITIC_EVALUATION_20260917.md)
- [2026-09-17：离线 CABLE 的 All-to-All 退化原因与 900 次专项对照](docs/OFFLINE_ALLTOALL_DIAGNOSIS_20260917.md)
- [2026-09-17：预留记账、分散选择与离线重训练的 5800 次实验](docs/OFFLINE_RESERVATION_EVALUATION_20260917.md)
- [2026-09-17：错开初始分配、条件策略与 ConWeave 的 7904 次实验](docs/COORDINATED_OFFLINE_EVALUATION_20260917.md)
- [2026-09-17：离线训练收益诊断、路径敏感性与 68 次单参数实验](docs/RL_TRAINING_DIAGNOSIS_20260917.md)
- [2026-09-18：均衡训练、退步约束与 17080 次独立核验实验](docs/BALANCED_OFFLINE_EVALUATION_20260917.md)
- [研究结论、完整探索总结和原始数据索引](RESEARCH_HISTORY.md)
- [拓扑与带宽迁移说明](docs/TOPOLOGY_BANDWIDTH_MIGRATION.md)
- [SONiC 硬件资料](docs/SONIC_HARDWARE_INFO.md)
- [上游 PRISMA 框架说明](README_PRISMA.md)和 [ConWeave 仿真器说明](conweave-ns3-main/README.md)

当前运行方法以选定策略的复现入口为准，上游文档保留框架背景。新的大规模实验输出应写到代码仓库之外，例如 `/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/`。历史 trace、TensorBoard 日志和中间模型已集中归档，恢复方法见研究总记录。

仓库中保留当前模型、源码、必要输入与少量论文重算依赖。旧实验文档的历史原文可从 Git 中查阅，最新结论以研究总记录和当前策略结果为准。
