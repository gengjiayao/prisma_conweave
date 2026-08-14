# PRISMA ConWeave 后续实验与研究计划

日期：2026-07-14  
当前里程碑：`rl-core-v8` 训练 `775665073`，冻结评估 `174791074`，
配对 ECMP `401470137`。

## 1. 当前结论边界

已经证明：在 `leaf_spine_128_100G_asym_OS2`、netload 70、DCTCP、PFC=0、
IRN=1、50 ms、seed 100 的严格配对场景中，保存后的 v8 模型能够以
`epsilon=0` 完成真实冻结推理，并在 small/large average 和 tail FCT 上超过 ECMP。
Prior-only 消融及 seed 101 复现进一步表明，安全先验贡献大部分 average 收益，learned
residual 在两个 seed 上稳定改善 P99/P99.9。相同 checkpoint 在 100 ms 独立 flow file
上无需重训完成 13629/13629 条流，并在严格共同流口径下继续超过 FECMP 的 average 和
slowdown tail，构成同拓扑/负载/分布下的时间跨度泛化证据。

尚未证明：

- 多 seed 下相对 FECMP 的配对优势是否稳定；
- 时间跨度泛化能否延伸到更长窗口和动态负载；
- 跨负载是否稳定；
- 从头训练的随机初始化是否可复现；
- AI collective traffic 下是否仍有优势；
- 是否超过 DRILL、CONGA 或 ConWeave。

## 2. 执行原则

1. 所有实验只由用户从 `/workspace/PRISMA_conweave/prisma/main.py` 启动。
2. 助手只修改/审查/验证代码并给出参数和命令，不自行启动仿真。
3. 一次只回答一个实验问题；拿到 output ID 后再决定下一步。
4. 能用默认值的参数不重复堆到命令中，但当前网络条件与默认值不同的参数必须显式写出。
5. 先做 5–50 ms 的正确性/归因实验，再投入更长训练。
6. `PRISMA_conweave_backup7.13` 和其他受保护备份永不修改。
7. v8 里程碑压缩包保持只读；后续文档作为 companion file 单独保存。

## 3. 实验顺序与判定规则

### P0：学习贡献消融（已完成）

比较同一场景的三组结果：

1. ECMP：`401470137`（已有）；
2. full frozen v8：`174791074`（已有）；
3. prior-only frozen v8：`695895724`。

prior-only 使用与 v8 完全相同的 observation 和 reward 权重，但 action 只取：

```text
argmax_a P_native(s,a)
```

它加载同一 checkpoint、关闭 exploration，并忽略 learned residual；不重新训练。

判定：

- full RL 明显优于 prior-only：证明 learned residual 提供额外价值；
- 两者接近：ECMP 增益主要来自安全先验，需谨慎表述“学习贡献”；
- prior-only 优于 full RL：当前 residual 在伤害策略，应先修 residual/训练目标，不能继续堆实验掩盖。

实现与复现状态（2026-07-14 晚）：

- 已增加默认关闭的 `--eval_prior_only` 参数；只允许 `train=0`、`lb=rl`、
  `eval_epsilon=0` 使用；
- checkpoint 仍按 v8 contract 正常加载，动作选择只忽略 learned residual；
- prior 的 NumPy 实现与 checkpoint 内 `rl_native_action_prior` 层逐值一致；
- 实现时容器全部 28 项单元测试通过；
- 复现命令：

```bash
python3 main.py \
    --train=0 \
    --load_path=examples/abilene/results/saved_models/rl_core_v8_safe_flowlet_train_50ms/final \
    --eval_prior_only=1 \
    --cc=dctcp \
    --pfc=0 \
    --irn=1 \
    --simul_time=0.05 \
    --buffer=50 \
    --netload=70 \
    --topo=leaf_spine_128_100G_asym_OS2 \
    --session_name=rl_core_v8_prior_only_eval_50ms \
    --save_models=0
```

成功运行后，`policy_actions.json` 必须显示
`fct_policy=rl_native_prior_only`、`eval_prior_only=1`、`eval_epsilon=0`，
且 `fct_is_final_checkpoint_policy=false`，避免把消融结果误标成完整模型结果。

结果（`695895724`）：6660/6660 条流完成，fresh flowlet 100%，action match
130117/130117，fallback=0，无端口坍缩。与 full v8 的流身份完全一致。
Prior-only 已贡献大部分平均收益；learned residual 的额外价值主要体现在 small/large
P99 与 P99.9。严格共同流口径下，full v8 仍在主要 slowdown 与 absolute FCT 指标上
超过 FECMP。完整数字见 `EXPERIMENT_695895724_PRIOR_ONLY_ABLATION_2026-07-14.md`。

### P1：多 seed 配对复现

先补 seed 101 三方配对：

- full frozen v8；
- prior-only；
- FECMP。

若 learned residual 的尾部优势复现，再补 seed 102 的 full frozen v8 与 FECMP；
只有归因仍不清楚时才补 seed 102 prior-only。核心共 5 个新 output，另有 1 个条件 output。
相同负载/时长下，这一步验证 Python/TensorFlow/ns-3 随机性和路径选择稳定性；每组三方
结果必须先核验实际 FCT 流身份，不能只根据 manifest 中的同名 flow file 判断配对成立。

通过标准：三个 seed 上 average FCT 的配对均值优于 ECMP，且不能靠单个异常 seed
支撑全部结论；同时检查 P95/P99、unfinished flow、fallback 和端口坍缩。

当前进度：seed 101 full `727315208` 与 prior-only `822665277` 已完成且流身份完全一致；
learned residual 的 small/large P99 slowdown 改善 23.4%/11.2%，复现 seed 100 的尾部
修正。seed 101 FECMP 尚待补充，因此多 seed 相对基线结论尚未闭环。

### P2：更长时间与不同 flow file

netload 70、seed 100，改用 100 ms：

- frozen v8：`615298051`；
- FECMP：`812362251`；
- prior-only：待运行，用于长时 median/tail 权衡归因。

前两项已经证明 50 ms checkpoint 在独立 100 ms flow file 上无逻辑退化：RL 完成
13629/13629，FECMP 完成 13586/13629；共同流 small/large slowdown Avg 分别改善
17.86%/19.20%，P99 改善 27.98%/47.07%。前后半段对两种算法施加几乎相同的 additive
slowdown penalty，说明主要是持续负载效应。完整记录见
`EXPERIMENT_615298051_812362251_TEMPORAL_GENERALIZATION_2026-07-15.md`。

### P3：负载曲线

以已有 70% 结果为中点，先补：

- netload 60：frozen v8 + FECMP；
- netload 80：frozen v8 + FECMP。

共 4 个新 output，形成 60/70/80 三点曲线。时间允许再补 50%。

报告 average、median、P95、P99、P99.9，并分别统计 small/large flow。

### P4：独立训练复现

使用 seed 101 从头训练一次 v8，再冻结加载评估一次，共 2 个新 output。

通过标准：

- transition、fresh flowlet、fallback contract 全部正确；
- loss 收敛且后期不发散；
- 模型保存后 `epsilon=0` 的结果没有出现训练/加载断层；
- 不要求每项都优于 seed 100，但不能恢复旧端口坍缩。

### P5：打通 AI traffic 参数链

ns-3 `run.py` 已支持 `cdf/allreduce/alltoall`，但 PRISMA parser 和 `run_ns3.py`
尚未传递这些参数。仅增加薄参数链，不修改 v8 RL core：

```text
prisma/main.py
  -> argument_parser
  -> run_ns3(params)
  -> conweave run.py
  -> AllReduce / AllToAll generator
```

需暴露的参数：

- `traffic_mode`；
- `ar_rounds/ar_step_us/ar_jitter_us/ar_chunk_bytes/ar_rotate_ring`；
- `aa_rounds/aa_chunk_bytes/aa_burst_us`；
- 后续如需独立 flow trace，再增加 `traffic_seed`，并写入文件名和 run manifest。

### P6：AI traffic 分层验证

1. 5–10 ms smoke：只检查生成器、流量数量、flowlet、action match、fallback、完成计数；
2. 50–100 ms zero-shot：当前 v8 checkpoint 对比 FECMP；
3. 同场景加入 DRILL、CONGA、ConWeave；
4. 只有 zero-shot 暴露稳定缺口时，才训练 AI-specific policy；
5. 只有确认纯局部状态不足时，才增加最小 phase/progress observation。

AI 场景至少覆盖：

- Ring AllReduce；
- synchronized All-to-All / MoE-like dispatch；
- incast/fan-in；
- collective + background CDF mixed traffic；
- 对称与 2:1 非对称链路；
- 可选动态 link degradation/failure。

AI 指标不只使用普通 FCT，还应记录：

- collective/iteration completion time；
- P95/P99 collective completion；
- 最慢 rank/straggler；
- 每端口利用率与空闲时间；
- PFC/IRN/retransmission/reorder；
- unfinished collective。

### P7：SOTA 对比顺序

推荐顺序：

1. DRILL：最现实的首个竞争目标；
2. CONGA：非对称场景的强基线；
3. ConWeave：RDMA/fine-grained reordering 强基线，最后做。

所有基线必须使用相同 topology、traffic file、seed、CC、PFC/IRN、buffer 和 duration。
ConWeave 的论文结论面向 RDMA；若当前实验使用 DCTCP，报告必须写成模拟器内同配置比较，
不能直接外推为普遍优于其 RDMA 设计。

## 4. 计划实验数量

| 阶段 | 新 output 数 | 累计 | 目的 |
|---|---:|---:|---|
| P0 prior-only | 1 | 1 | 学习归因 |
| P1 seed 101/102 | 5（+1条件） | 6（+1条件） | 配对复现与学习归因 |
| P2 100 ms | 3 | 9（+1条件） | 时间泛化与长时归因 |
| P3 load 60/80 | 4 | 13（+1条件） | 负载曲线 |
| P4 fresh training | 2 | 15（+1条件） | 训练可复现性 |

完成前 9 个核心新 output 可形成最小可信闭环；完成 15 个可形成较完整的普通 CDF
实验章节。AI/SOTA 实验在参数链接通并完成 smoke 后另行计数。

## 5. 当前实验台账

| 状态 | Output ID | 角色 |
|---|---:|---|
| 完成 | 401470137 | 50 ms asymmetric FECMP baseline |
| 完成 | 775665073 | v8 50 ms training |
| 完成 | 174791074 | v8 frozen greedy evaluation |
| 完成 | 695895724 | v8 prior-only frozen ablation |
| 完成 | 727315208 | seed 101 full frozen v8 |
| 完成 | 822665277 | seed 101 prior-only |
| 完成 | 615298051 | seed 100 100 ms full frozen v8 |
| 完成 | 812362251 | seed 100 100 ms FECMP |
| 待运行 | TBD | seed 100 100 ms prior-only |
| 待运行 | TBD | seed 101 50 ms FECMP |

## 6. 当前停止条件

P0 已通过正确性门槛。P1 任一轮若出现 observation/action mismatch、fallback 非零、
流未完成或端口坍缩，先停止并修正确性。

seed 101 已复现 learned residual 的尾部收益，100 ms full/FECMP 已通过时间跨度泛化
门槛。先完成 100 ms prior-only 归因，再补 seed 101 FECMP 和负载曲线；若出现
observation/action mismatch、fallback、未完成 RL flow 或端口坍缩，立即停止扩展实验。
