#!/usr/bin/env python3
"""Audit raw FCT output and write a Chinese report for compare_path_with_conweave."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'prisma/source'))
import path_rollout as env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    task = args.experiment_dir.resolve()
    rows = json.loads((task / 'combined-results.json').read_text())
    summary = json.loads((task / 'combined-summary.json').read_text())
    protocol = json.loads((task / 'protocol.json').read_text())
    previous = json.loads((ROOT / 'prisma/experiments/path_learning/wire-feedback.json').read_text())
    assert protocol['binary_sha256'] == previous['provenance']['native_binary_sha256']
    binary = Path(protocol['binary'])
    assert env.sha(binary) == protocol['binary_sha256']
    for name, h in protocol['binary_dependencies'].items():
        assert env.sha(binary.parent.parent / name) == h
    for model in protocol['models']:
        assert env.sha(Path(model['path'])) == model['sha256']
    assert env.sha(task / 'source/compare_path_with_conweave.initial.py') == protocol['source_sha256']
    supplemental = json.loads((task / 'timer-sensitivity-protocol.json').read_text())
    assert env.sha(task / 'source/compare_path_with_conweave.sensitivity.py') == supplemental['source_sha256']
    trace_hashes, config_groups, metrics_checked, failures = {}, {}, 0, []
    for r in rows:
        folder = Path(r['folder'])
        job = json.loads((folder / 'job.json').read_text())
        assert env.sha(Path(job['config'])) == job['config_sha256']
        assert env.sha(Path(job['traffic'])) == job['traffic_sha256']
        key = (r['family'], r['traffic_seed'])
        trace_hashes.setdefault(key, set()).add(job['traffic_sha256'])
        config = {v.split()[0]: ' '.join(v.split()[1:]) for v in Path(job['config']).read_text().splitlines() if v.strip()}
        config_groups.setdefault(key, {})[(r['arm'], r['fit_seed'])] = config
        status = json.loads((folder / 'status.json').read_text())
        if not r['valid']:
            assert status['returncode'] == 1
            assert 'tailTime is before Phase0-TxTime' in (folder / 'log.txt').read_text()
            failures.append(dict(arm=r['arm'], family=r['family'], traffic_seed=r['traffic_seed'],
                                 exit_code=status['returncode'], completion_lines=r['completion_lines_before_failure']))
            continue
        # Rebuild the offered/completed flow mapping, then independently calculate
        # the reported metrics from raw FCT nanoseconds rather than result.json.
        loaded, flow_map = env.load(folder)
        assert loaded['offered'] == loaded['completed'] == r['offered'] == r['completed']
        fct_path = folder / 'fct_output_file.txt'
        assert env.sha(fct_path) == r['fct_sha256']
        a = np.loadtxt(fct_path, dtype=np.int64, ndmin=2)
        fct = a[:, 6] / 1000.
        n = int(np.ceil(len(fct) * .01))
        metrics = dict(all_mean_us=fct.mean(), all_p99_us=np.percentile(fct, 99),
                       makespan_us=(np.max(a[:, 5] + a[:, 6])-2000000000)/1000.,
                       all_cvar99_us=np.sort(fct)[-n:].mean(), all_max_us=fct.max())
        for k, v in metrics.items():
            # ns-3 truncation versus requested trace-time rounding can differ
            # by 1 ns; env.load explicitly allows this start-time difference.
            tolerance = .0010001 if k == 'makespan_us' else 1e-9
            assert np.isclose(v, r[k], rtol=1e-12, atol=tolerance), (folder, k, v, r[k])
        metrics_checked += 1
    assert len(trace_hashes) == 20 and all(len(h) == 1 for h in trace_hashes.values())
    shared = ['CC_MODE', 'HAS_WIN', 'VAR_WIN', 'ENABLE_IRN', 'ENABLE_PFC', 'IRN_RTO_LOW_US',
              'IRN_RTO_HIGH_US', 'FLOWGEN_START_TIME', 'FLOWGEN_STOP_TIME', 'SIMULATOR_EXTRA_TIME',
              'TOPOLOGY_FILE', 'PACKET_PAYLOAD_SIZE', 'BUFFER_SIZE', 'RANDOM_SEED',
              'ALPHA_RESUME_INTERVAL', 'RATE_DECREASE_INTERVAL', 'RP_TIMER', 'EWMA_GAIN',
              'RATE_AI', 'RATE_HAI', 'MIN_RATE', 'KMAX_MAP', 'KMIN_MAP', 'PMAX_MAP']
    for configs in config_groups.values():
        reference = configs['conweave_historical', 0]
        for (arm, fit), config in configs.items():
            if arm == 'rl_dctcp':
                continue
            for k in shared:
                assert config.get(k) == reference.get(k), (arm, fit, k)
    # Check point estimates directly without relying on the bootstrap implementation.
    for name, comparison in summary['comparisons'].items():
        if not comparison['valid']:
            continue
        actor, base = name.split('_vs_')
        baselines = {(r['family'], r['traffic_seed']): r for r in rows if r['arm'] == base}
        for metric, result in comparison['statistics'].items():
            lr = [np.log(r[metric]/baselines[r['family'], r['traffic_seed']][metric]) for r in rows if r['arm'] == actor]
            assert np.isclose(np.expm1(np.mean(lr))*100, result['delta_pct'], rtol=1e-12)
    snapshot = task / 'source'
    paths = ['prisma/scripts/compare_path_with_conweave.py', 'prisma/scripts/analyze_path_conweave.py',
             'prisma/source/path_rollout.py', 'conweave-ns3-main/config/path_learning_base.txt',
             'conweave-ns3-main/config/leaf_spine_128_100G_asym_OS2.txt',
             'conweave-ns3-main/run.py', 'conweave-ns3-main/scratch/network-load-balance.cc',
             'conweave-ns3-main/src/point-to-point/model/conweave-routing.cc']
    hashes = {}
    for path in paths:
        dest = snapshot / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
        hashes[path] = env.sha(dest)
    audit = dict(status='PASS', primary_and_sensitivity_runs=len(rows), raw_fct_metric_runs=metrics_checked,
                 paired_traffic_conditions=len(trace_hashes), shared_transport_fields=shared,
                 preserved_failures=failures, model_and_binary_hashes=True, source_snapshot_sha256=hashes,
                 summary_sha256=env.sha(task / 'combined-summary.json'))
    env.dump(task / 'verification.json', audit)

    fmt = lambda x: f'{x:+.2f}%'
    def ci(d):
        return fmt(d['delta_pct'])+' ['+', '.join(fmt(x) for x in d['ci95_pct'])+']'
    focus = summary['comparisons']['rl_dcqcn_vs_conweave_historical']['statistics']
    names = {'all_mean_us':'平均 FCT', 'all_p99_us':'P99 FCT', 'makespan_us':'整批完成时间',
             'all_cvar99_us':'最慢 1% 流的平均 FCT', 'all_max_us':'最大 FCT'}
    headline = '\n'.join(f'| {v} | {ci(focus[k])} |' for k,v in names.items())
    arms = '\n'.join(f"| {k} | {v['runs']} | {v['valid']} | {v['offered']:,} | {v['completed_in_valid_runs']:,} |" for k,v in summary['arms'].items())
    comparison_table = []
    for k,c in summary['comparisons'].items():
        if c['valid']:
            s=c['statistics']
            comparison_table.append('| '+k+' | '+' | '.join(fmt(s[m]['delta_pct']) for m in ['all_mean_us','all_p99_us','makespan_us'])+' |')
    family_table = []
    for family in protocol['families']:
        base = [r for r in rows if r['arm']=='conweave_historical' and r['family']==family]
        rl = [r for r in rows if r['arm']=='rl_dcqcn' and r['family']==family]
        absolute = [np.mean([r[m] for r in group])/1000 for m in ['all_mean_us','all_p99_us'] for group in [base,rl]]
        family_table.append('| '+family+' | '+' | '.join(f'{x:.3f}' for x in absolute)+' | '+' | '.join(fmt(focus[m]['per_family_pct'][family]) for m in ['all_mean_us','all_p99_us','makespan_us'])+' |')
    failures_table = '\n'.join(f"| {r['family']} | {r['traffic_seed']} | {r['exit_code']} | {r['completion_lines']:,} |" for r in failures)
    data_link = os.path.relpath(task, args.report.resolve().parent)
    report = f'''# 冻结 RL 与 ConWeave 的实际对照结果

日期：2026-09-16。实验目录：`{task}`。本次实际运行 12 次启动检查、160 次主要对照及 20 次补充定时参数敏感性，共 192 次；没有训练、挑选或修改 RL 模型。

## 1. 同用 DCQCN 时，RL 降低平均和 P99，但延长整批完成时间

在 128 主机、1/0.5 Gbps 非对称 leaf-spine 网络中，相对**采用项目历史低速定时参数、状态保留 2 秒的 ConWeave**，当前三个冻结 RL 模型的平均 FCT 下降 **{abs(focus['all_mean_us']['delta_pct']):.2f}%**，P99 下降 **{abs(focus['all_p99_us']['delta_pct']):.2f}%**，整批完成时间增加 **{focus['makespan_us']['delta_pct']:.2f}%**。双方使用相同 DCQCN、IRN、PFC、窗口及超时配置。

| 指标 | RL 相对历史低速 ConWeave 的变化 [95% 配对 bootstrap 区间] |
|---|---:|
{headline}

负数表示耗时下降，正数表示增加。结果表明平均/尾部单流时延与整批任务完成速度存在取舍，不能总结为 RL 全面超过 ConWeave。此处 ConWeave 是明确列出的低速适配配置，不能称为未调整的原参数版本。

仓库面向 leaf-spine 的原定时参数、默认 2 ms 状态保留版本，在 20 次主测试中有 6 次内部检查退出。因此本报告不计算它的整体 FCT 优劣，也不只挑其成功运行来比较。

## 2. 这次比较控制了传输配置，并保留全部模型

旧 RL 主结果使用 DCTCP；原生代码明确要求 ConWeave 使用 DCQCN。因此本次建立两种 RL 条件：

- `rl_dcqcn`：三个冻结模型转到 DCQCN，`CC_MODE=1, HAS_WIN=0, VAR_WIN=0`。它与 ConWeave 的传输设置一致，是本文主要条件对照；模型仍来自 DCTCP 训练，未重训。
- `rl_dctcp`：保留 `CC_MODE=8, HAS_WIN=1, VAR_WIN=1`，表示原 RL 完整方案。与 ConWeave 的比较包含拥塞控制/窗口差异。

所有运行使用同一冻结二进制及依赖，二进制 SHA-256 为 `{protocol['binary_sha256']}`，与既有严格报文反馈实验一致。RL 继续使用真实 194 B/200 μs 报告、初始选路和整流保持；不启用迁移门控、pressure 或在线更新。ConWeave 使用自身反馈和重排序，不接入 RL 反馈。

| 公共设置 | 数值 |
|---|---|
| 拓扑 | 128 主机、8 leaf、8 spine；主机链路 1 Gbps，上联四快四慢 1/0.5 Gbps；传播 1 μs |
| 流量 | uniform、storage、sweep、mixed、mixed_fast |
| 新测试种子 | 87001–87004，共 20 个流量条件 |
| RL 谱系 | 9101、9102、9103，全部保留、分别运行 |
| 执行随机种子 | 100 |
| 时间 | 50 ms 注入，1 s 排空 |
| 传输 | IRN 开、PFC 关，RTO 10/32 ms，缓冲 50 MiB |
| 业务公平性 | 每个条件所有方法读取字节相同的流量文件；sweep 的背景固定路径规则也相同 |

## 3. ConWeave 三种配置必须分别解读

单位为 μs，只有状态保留栏另标为 ms/s。前两列在主测试前写入实验协议；第三列是看到主测试后，补测仓库已有历史配置，属于**事后参数敏感性检查**。没有基于这些测试对 RL 选点，也没有搜索 ConWeave 参数网格。

| 参数 | `conweave_2ms` | `conweave_2s` | `conweave_historical` |
|---|---:|---:|---:|
| 状态保留 `CONWEAVE_AGING_US` | 2 ms | 2 s | 2 s |
| 额外回复期限 | 4 | 4 | 200 |
| 发送侧到期 `TX_EXPIRY` | 300 | 300 | 1000 |
| 路径暂停 | 16 | 16 | 16 |
| 额外 VOQ 刷新时间 | 16 | 16 | 64 |
| 默认 VOQ 等待时间 | 200 | 200 | 400 |

第一列定时参数来自 [run.py 的 leaf-spine 分支](../conweave-ns3-main/run.py)，状态保留默认值来自 [原生入口](../conweave-ns3-main/scratch/network-load-balance.cc)。第二列只延长状态保留，便于观察低速环境中的状态失效问题。第三列使用 [path_learning_base.txt](../conweave-ns3-main/config/path_learning_base.txt) 和 [path_rollout.make()](../prisma/source/path_rollout.py) 已有的低速定时设置，并采用历史诊断使用过的 2 秒状态保留。

结果对定时设置很敏感：仅延长状态保留的版本明显慢于历史低速参数版本。两组都完整报告，避免把较弱配置下约 47% 的平均 FCT 差距当作稳定结论。

| 比较 | 平均 FCT 变化 | P99 变化 | 整批完成时间变化 |
|---|---:|---:|---:|
{chr(10).join(comparison_table)}

## 4. 全部运行及失败记录

| 方法 | 主测试/补测运行数 | 有效且全部流完成的运行 | 提供流实例数 | 有效运行中的完成流数 |
|---|---:|---:|---:|---:|
{arms}

同一批流量分别交给三个 RL 模型，因此 RL 流实例总数是单个基线的三倍，不代表 RL 承担不同负载。失败运行的已完成子集没有进入 FCT 比较；表中最后一列不包含失败前的部分完成流，不能据此计算默认 ConWeave 的正常终止完成率。

默认 2 ms 版本的六次失败如下：

| 负载 | 流量种子 | 退出码 | 退出前 FCT 日志行数，仅诊断 |
|---|---:|---:|---:|
{failures_table}

六次都出现 `tailTime is before Phase0-TxTime`。源码 [conweave-routing.cc](../conweave-ns3-main/src/point-to-point/model/conweave-routing.cc) 在 `tailTime < phase0TxTime` 时执行 `exit(1)`。这属于该实现/配置下的有效性失败；不能把程序提前退出解释成单纯“传输慢”，也不能从这一条件独立证明完整根因。延长状态保留后这批运行均完成，未删除检查或修改 ConWeave 算法。

12 次启动检查使用独立种子 86999，测试 uniform/storage/mixed_fast、一个 RL 谱系和两个 ConWeave 版本；其中默认 ConWeave 的 uniform 运行也发生同样退出，其余 11 次有效。启动检查不计入上述效果统计。

## 5. 逐负载结果显示收益与整批完成代价

以下针对同用 DCQCN 的 RL 和历史低速 ConWeave。绝对数值单位为 ms：先在每次运行中计算平均/P99，再跨该负载的种子及 RL 谱系做算术平均；百分比则用配对比值的几何平均，不能直接将两列绝对均值相除重现。

| 负载 | ConWeave 平均 | RL 平均 | ConWeave P99 | RL P99 | 平均变化 | P99 变化 | 整批变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(family_table)}

整批完成时间定义为最后一条流完成时刻减业务起点 2.0 s；最大 FCT 是最慢单条流的持续时间，两者不是同一指标。平均/P99 衡量单流完成体验，整批时间更接近这批业务结束的时间，解释结论时须同时保留。

## 6. 统计与核验范围

聚合变化使用 `100×[exp(mean(log(RL指标/对照指标)))−1]`。95% 区间来自 10000 次配对 bootstrap：四个流量种子作为跨五类负载共同的块重采样，同时配对重采样三个冻结模型谱系。三个谱系共享原训练数据，区间不包括重新训练/选点的不确定性。只有四个新流量种子，因此这些区间是探索性结果，不能替代更广的负载、规模和执行种子检验。

已逐运行核对 180 次主测试和补测的配置、流量和模型来源；其中 174 次有效运行从原始 FCT 纳秒值重算平均、P99、CVaR99、最大值和整批时间，并核对提供流/完成流映射。六次失败日志保留。相同 DCQCN 条件的拥塞控制、窗口、IRN/PFC、RTO、业务时间、拓扑、包长、缓冲、ECN 阈值等字段逐项一致。

当前发现支持：冻结 RL 在这一低速场景中，相对已验证能完整运行的 ConWeave 适配配置改善平均与 P99 FCT，同时存在整批完成时间代价。它不支持原参数 ConWeave 的无条件排名，也不验证 100 Gbps、1024 主机、硬件或推理时延下的优势，更没有将两种路由系统的差异全部归因于 RL 学习。

## 7. 数据和复现

- [主实验协议]({data_link}/protocol.json)、[事后定时敏感性协议]({data_link}/timer-sensitivity-protocol.json)。
- [全部指标和区间]({data_link}/combined-summary.json)、[逐运行结果]({data_link}/combined-results.json)。
- [原始运行目录]({data_link}/runs/)，含配置、流量索引、FCT、退出日志及 RL 审计。
- [本次原始数据核验]({data_link}/verification.json)、[源码快照]({data_link}/source/)、[冻结二进制与依赖]({data_link}/binaries/)。
- [运行脚本](../prisma/scripts/compare_path_with_conweave.py)、[重算与报告脚本](../prisma/scripts/analyze_path_conweave.py)。

从仓库根使用已冻结的二进制和单独的新输出目录运行：

```bash
export COMPARISON_RUN=/absolute/path/to/new_comparison_directory
export NS3_BINARY={task}/binaries/scratch/network-load-balance

.venv-cable/bin/python prisma/scripts/compare_path_with_conweave.py \\
  --experiment-dir "$COMPARISON_RUN" --ns3-binary "$NS3_BINARY" --stage smoke
.venv-cable/bin/python prisma/scripts/compare_path_with_conweave.py \\
  --experiment-dir "$COMPARISON_RUN" --ns3-binary "$NS3_BINARY" --stage test
.venv-cable/bin/python prisma/scripts/compare_path_with_conweave.py \\
  --experiment-dir "$COMPARISON_RUN" --ns3-binary "$NS3_BINARY" --stage sensitivity
```

当前目录已保留原始版本和补测版本的脚本哈希。核验逐运行指标和汇总点估计，并重新生成本报告可执行以下命令；bootstrap 区间读取已保存的汇总，不由该核验脚本重新生成：

```bash
.venv-cable/bin/python prisma/scripts/analyze_path_conweave.py \\
  --experiment-dir {task} \\
  --report docs/RL_VS_CONWEAVE_COMPARISON.md
```

初始协议的脚本哈希对应 `source/compare_path_with_conweave.initial.py`，补测协议对应加入 sensitivity 阶段后的脚本。报告重算会核验此历史来源；新的完整实验应保存自己的运行脚本快照，不覆盖本次证据。
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    report_base = args.report.resolve().parent
    external_report = re.sub(r'\]\(([^)]+)\)',
                             lambda m: '](' + str((report_base / m[1]).resolve()) + ')', report)
    (task / 'REPORT.md').write_text(external_report)
    print(json.dumps(audit, indent=2))
    print('WROTE', args.report)


if __name__ == '__main__':
    main()
