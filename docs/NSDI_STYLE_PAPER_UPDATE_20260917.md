# NSDI 格式论文更新与新评测

日期：2026-09-17。用户已明确：使用 NSDI 排版要求，不实际向 NSDI 投稿。

## 论文更新

目标 Overleaf 项目：[NSDI_2027_Fall](https://latex.cstcloud.cn/project/6aa256c8ce0d44250fa33c4f)。本地论文目录：`/home/gengjiayao/ict/work/prisma_conweave_paper`。

标题：**CABLE: Completion-Aware Balancing Learned from Episode Rewards**。

正文改为 Introduction、Background、Motivation、Design、Evaluation、Related Work、Conclusion 七节。最新技术主线是八系数线性策略搜索、全网完成奖励、源 leaf 首包选路、整流保持，以及通过实际收到的报文构造远端状态；旧 DQN、bounded residual 和旧 50 维观察叙述不再作为当前方法。

核心 state/action/reward、训练更新、两阶段训练、反馈与执行机制保留在正文；精确参数、随机种子、基线定时设置和复现规则进入附录。英文改为短句，首次解释必要术语和公式符号。

本地检查版为正文 10 页、参考文献 1 页、附录 2 页。US letter、双栏、10 pt Times 系字体、12 pt 行距、7×9 英寸正文框、0.33 英寸栏距，所有字体嵌入，无超宽/超高排版警告。保留正常模板宏包提示。

八幅矢量图分别展示系统结构、等容量拓扑、P95/P99 负载曲线、工作负载差异、流大小分组、学习与输入组对照、平均/整批完成和完成数量、三个训练谱系。论文没有数值表。

## 新运行矩阵

正式实验 **2560 次**，另有不计入统计的 32 次成功预检。

- 128 主机、8 leaf、8 spine；每 leaf 上联总带宽均为 6 Gbps。
- 对称拓扑：8×0.75 Gbps；非对称拓扑：4×1 Gbps + 4×0.5 Gbps。主机接入 1 Gbps，链路传播 1 μs。
- 六种路由基线：ECMP、weighted ECMP、DRILL、CONGA、LetFlow、ConWeave。
- 三个已冻结 RL 模型加同信息固定评分规则。无测试集调参、无本轮重训。
- 四种工作负载：uniform、storage、mixed hotspot、fast hotspot；名义主机负载 20%/35%/50%；八个新流量种子 88001–88008。
- 主实验 1920 次；35% 负载的 earlier checkpoint、去远端系数组、去历史系数组、zero score 对照共 640 次。
- 公共配置统一为 DCQCN、IRN、关闭 PFC，并统一不同速率的 ECN 最大标记概率为 0.20。ConWeave 使用此前确定的低速适配定时器和 2 秒状态保留。

所有正式运行正常退出，全部提供流完成。共有 **22,899,200 个流实例**，来源于 **96 份不同业务 trace**，不同算法/拓扑/模型重复使用 trace 进行配对。

## 主要结果

以下为条件内配对后对 log 比值等权求均值再取指数，负值表示 RL 指标更低；不是把所有流合并后重新求一个分位数。

- 相对 CONGA：P95 **−29.40%**，P99 **−19.50%**；P99 95% 区间 **[−20.18%, −18.81%]**。
- 相对适配 ConWeave：P95 **−23.01%**，P99 **−17.82%**；P99 95% 区间 **[−18.34%, −17.33%]**。
- 相对 ECMP：P95 **−41.52%**，P99 **−33.51%**。
- 相对固定评分规则：P95 **−0.15%**，P99 **−0.07%**，两项区间均跨零，不能宣称学习总体优于这一规则。
- 在 35% 负载、当前实际报文反馈下，相对各自 earlier checkpoint，继续学习带来 P95 **−1.32%**、P99 **−2.21%**，P99 区间 **[−3.11%, −1.35%]**。
- 相对去远端系数组对照：P95/P99 **−31.39%/−23.97%**；相对去历史系数组：**−5.18%/−6.78%**。这些是冻结模型系数置零敏感性分析，没有重训，也没有省去反馈包。

论文明确保留不利结果：相对 ConWeave，平均 FCT 虽降低 **20.02%**，整批完成时间却增加 **2.86%**，区间 **[2.03%, 3.56%]**；小流 P99 相对固定规则增加 **1.53%**。当前结果仍属于低速率、短注入期、零建模推理耗时的仿真证据，不能当作硬件部署结果。

## 验证与复现

执行脚本：[nsdi_tail_evaluation.py](../prisma/scripts/nsdi_tail_evaluation.py)。统计脚本：[analyze_nsdi_tail.py](../prisma/scripts/analyze_nsdi_tail.py)。新增 native 改动只加入对称 750 Mbps 拓扑的 BDP=4249 B 查表项，并保留与原生计算值的断言；11 个原生共享库与之前冻结版本逐字节相同。

本轮完整证据目录：

```text
/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/nsdi-refresh-20260917-aomgdw2h
```

`experiment/protocol.json` 在正式测试前固定方案；`experiment/runs/` 保留配置、状态、原始 FCT 与执行审计；`experiment/raw-audit.json` 记录 2560 次逐运行核验。两批早期预检设置错误也独立保留，不冒充算法性能失败。

独立复算检查 262 组点估计和置信区间全部一致；20 次原始运行的 166709 条 FCT 抽样，包括全体和大小分组指标，也全部一致。脚本完整模式逐条核验全部原始 FCT；compact 模式用论文数据包即可重绘，复算结果与完整模式一致。

论文中的 `scripts/README.md` 提供重绘命令。`data/current_runs.json`、`current_summary.json`、`current_protocol.json` 和图清单记录紧凑数据与来源；`models/` 含执行用模型。完整重新仿真仍需项目及原生构建环境，论文数据包本身不是完整模拟器分发包。

修改前的云端 ZIP 与本地完整论文目录均保留在本轮证据目录的 `cloud-before/` 和 `local-before/`。

## 交付核验

已更新指定 Overleaf 项目，63 个源文件和资产下载回查哈希全部匹配，包括 14 个冻结执行模型。云端编译返回 success，报告三次 LaTeX 运行、零错误；云端 PDF 已下载并作为本地 `main.pdf` 保存，两者逐字节一致。云端与最终本地重编译 PDF 的全文及分页一致，均为 13 页，字体全部嵌入。

| 检查 | 结果 |
|---|---|
| 页数 | 正文 10、参考文献 1、附录 2 |
| 引用与排版 | 最终本地重编译无未定义引用、无超宽/超高警告 |
| 字体与图 | 全部字体嵌入，8 幅矢量图，无数值结果表 |
| 独立复核 | 技术、语言、结构复核均无 critical/major 遗留项 |
| 机械语言检查 | em dash 0；被动模式 0；30 个否定模式均为必要事实边界，逐项保留说明见复核记录 |
| 云端/本地源文件 | 63 项哈希全部匹配 |
| 工作区检查 | 两个项目均通过 `git diff --check` |

云端单独日志下载端点超时，备用日志和编译输出归档端点返回 404；因此不声称取得云端 `.log`。云端成功状态、零错误统计、实际 PDF 及本地完整日志均已留存。渲染比对中 7 页逐像素相同，其余差异局限于图区域；已检查云端关键图页。详细状态见证据目录 `FINAL_VERIFICATION.json`。最终源文件相对语言复核版本仅清理两个文件的末尾空行，保留前后哈希及等价性记录。

标题经用户确认更新为 **CABLE: Completion-Aware Balancing Learned from Episode Rewards**，缩写取 Completion、Aware、Balancing、Learned、Episode。标题更新记录位于 `/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/cable-title-izsximlu`。


## 后续集体通信实验扩展

已另行完成 1920 次 All-to-All / Ring All-Reduce 正式评测，并同步 NSDI_2027_Fall。该批全部完成 7,234,560 条传输；结合前述独立流量实验，当前论文总计 4480 次正式运行、30,133,760 条传输实例。

All-Reduce 的任务完成时间相对 CONGA / 适配 ConWeave 降低 13.0% / 18.5%，P95/P99 同时改善，因此正文新增该实验。All-to-All 分别退化 26.4% / 24.6%，作为负面结果同时展示。同信息固定规则在两类任务上都优于当前学习策略，论文明确限制学习收益的主张。没有测试后重训练或系数调整。

新增三张矢量图：All-Reduce 完整曲线、两类通信配对置信区间、附录 All-to-All 完整曲线。当前 PDF 为 **12 页正文 + 2 页参考文献 + 3 页附录**，总计 17 页；正文页数计入结论之后浮动到第 12 页的图，不能只按结论文字页号计数。

云端 26 个修改/新增文件逐一哈希核对一致，云端与本地编译均成功。原云端与本地版本均已备份；没有提交 git。

详见 [集体通信实验报告](COLLECTIVE_TRAFFIC_EVALUATION_20260917.md)。完整协议、原始记录、排除的规模试运行、复核意见和云端校验在 `work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p/`。
