#!/usr/bin/env python3
"""Evaluate frozen path actors against ConWeave, retaining failures and CC controls.

No training or model selection. ConWeave parameters follow run.py's leaf-spine
branch; the 2-second aging variant is an explicitly separate sensitivity arm.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import numpy as np
import path_rollout as env

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ['uniform', 'storage', 'sweep', 'mixed', 'mixed_fast']
FITS = [9101, 9102, 9103]
TEST_SEEDS = [87001, 87002, 87003, 87004]
CW = dict(CONWEAVE_REPLY_TIMEOUT_EXTRA='4', CONWEAVE_PATH_PAUSE_TIME='16',
          CONWEAVE_EXTRA_VOQ_FLUSH_TIME='16', CONWEAVE_DEFAULT_VOQ_WAITING_TIME='200',
          CONWEAVE_TX_EXPIRY_TIME='300')
METRICS = ['all_mean_us', 'all_p99_us', 'makespan_us', 'all_cvar99_us', 'all_max_us']


def rewrite_config(folder, overrides):
    path = folder / 'config.txt'
    values = dict(overrides)
    lines = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in values:
            lines.append(fields[0] + ' ' + values.pop(fields[0]))
        else:
            lines.append(line)
    lines.extend(k + ' ' + v for k, v in values.items())
    path.write_text('\n'.join(lines) + '\n')
    j = json.loads((folder / 'job.json').read_text())
    j['config_sha256'] = env.sha(path)
    env.dump(folder / 'job.json', j)


def initialize(task):
    protocol = task / 'protocol.json'
    if protocol.exists():
        p = json.loads(protocol.read_text())
        assert p['binary_sha256'] == env.sha(env.BINARY)
        return p
    (task / 'source').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), task / 'source/compare_path_with_conweave.initial.py')
    models = []
    for fit in FITS:
        source = ROOT / 'prisma/experiments/path_learning/policies' / f'real-s{fit}.txt'
        dest = task / 'models' / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source, dest)
        models.append(dict(fit=fit, path=str(dest), sha256=env.sha(dest)))
    p = dict(created_epoch=time.time(), scope='Exploratory frozen comparison; no training/selection.',
             binary=str(env.BINARY), binary_sha256=env.sha(env.BINARY),
             binary_dependencies={f.name: env.sha(f) for f in sorted(env.BINARY.parent.parent.glob('*.so*')) if f.is_file()},
             source_sha256=env.sha(Path(__file__)), models=models,
             families=FAMILIES, test_seeds=TEST_SEEDS, smoke_seed=86999,
             actors='All three published actors retained; RL DCTCP and frozen-transfer DCQCN arms.',
             conweave_leaf_spine_parameters=CW, conweave_aging_us=[2000, 2000000],
             conweave_source='conweave-ns3-main/run.py leaf_spine branch; aging default 2ms from scratch/network-load-balance.cc',
             primary_comparison='RL DCQCN vs ConWeave DCQCN, same transport/window settings; DCTCP RL is a separate end-to-end scheme comparison.',
             shared=dict(hosts=128, duration_s=.05, drain_s=1, execution_seed=100,
                         IRN=1, PFC=0, RTO_us=[10000, 32000]),
             completion='Keep every failure. Compute uncensored comparative FCT only when every paired run in all workloads/seeds completes all offered flows.',
             bootstrap='10000 paired traffic-seed block resamples across five fixed families and paired resampling of three actor lineages. Four traffic seeds: exploratory, no training uncertainty.',
             interpretation='2s aging is a labeled low-speed sensitivity, not original ConWeave. Frozen RL was trained under DCTCP; DCQCN is transfer without retraining.')
    env.dump(protocol, p)
    return p


def prepare(task, protocol, stage):
    if stage == 'sensitivity':
        settings = {**CW, 'CONWEAVE_REPLY_TIMEOUT_EXTRA': '200',
                    'CONWEAVE_TX_EXPIRY_TIME': '1000', 'CONWEAVE_EXTRA_VOQ_FLUSH_TIME': '64',
                    'CONWEAVE_DEFAULT_VOQ_WAITING_TIME': '400', 'CONWEAVE_AGING_US': '2000000',
                    'SIMULATOR_EXTRA_TIME': '1.0', 'IRN_RTO_LOW_US': '10000',
                    'IRN_RTO_HIGH_US': '32000', 'RL_SWITCH_MODE': '0',
                    'RL_PRESSURE_ENABLE': '0', 'RL_GATE_MODE': '0', 'CONGA_ACK_FEEDBACK': '0'}
        addition = task / 'timer-sensitivity-protocol.json'
        if not addition.exists():
            shutil.copy2(Path(__file__), task / 'source/compare_path_with_conweave.sensitivity.py')
            env.dump(addition, dict(created_epoch=time.time(), settings=settings,
                source_sha256=env.sha(Path(__file__)),
                reason='After the primary comparison, check the historical low-speed configuration already present in path_rollout.make and path_learning_base. No parameter search or actor selection; report both baselines.',
                families=FAMILIES, traffic_seeds=TEST_SEEDS))
        jobs = []
        for seed in TEST_SEEDS:
            for family in FAMILIES:
                name = f'sensitivity_{family}_{seed}_conweave_historical_s0'
                folder = task / 'runs' / name
                if not folder.exists():
                    folder = env.make(name, scenario=family, seed=seed, method='conweave',
                                      enabled=1, feedback=0, extra=settings)
                    j = json.loads((folder / 'job.json').read_text())
                    j.update(arm='conweave_historical', family=family, fit_seed=0,
                             phase=stage, mode=0, reselect=0, margin_us=0, interval_us=50,
                             analysis_label=name, optimizer_seed=0,
                             protocol_sha256=env.sha(addition))
                    env.dump(folder / 'job.json', j)
                jobs.append(folder)
        env.dump(task / f'{stage}-jobs.json', [str(p) for p in jobs])
        return jobs
    seeds = [protocol['smoke_seed']] if stage == 'smoke' else protocol['test_seeds']
    families = ['uniform', 'storage', 'mixed_fast'] if stage == 'smoke' else FAMILIES
    models = protocol['models'][:1] if stage == 'smoke' else protocol['models']
    jobs = []
    for seed in seeds:
        for family in families:
            for arm in ['rl_dctcp', 'rl_dcqcn', 'conweave_2ms', 'conweave_2s']:
                for m in models if arm.startswith('rl') else [dict(fit=0)]:
                    name = f'{stage}_{family}_{seed}_{arm}_s{m["fit"]}'
                    folder = task / 'runs' / (('r128_v2_' if arm.startswith('rl') else '') + name)
                    if not folder.exists():
                        if arm.startswith('rl'):
                            assert env.sha(Path(m['path'])) == m['sha256']
                            folder = env.make_trial(name, family, seed, arm, stage,
                                                    model=m['path'], fit=m['fit'])
                            rewrite_config(folder, {'CC_MODE': '8' if arm == 'rl_dctcp' else '1',
                                                    'HAS_WIN': '1' if arm == 'rl_dctcp' else '0',
                                                    'VAR_WIN': '1' if arm == 'rl_dctcp' else '0'})
                        else:
                            extra = {**CW, 'CONWEAVE_AGING_US': '2000' if arm == 'conweave_2ms' else '2000000',
                                     'SIMULATOR_EXTRA_TIME': '1.0', 'IRN_RTO_LOW_US': '10000',
                                     'IRN_RTO_HIGH_US': '32000', 'RL_SWITCH_MODE': '0',
                                     'RL_PRESSURE_ENABLE': '0', 'RL_GATE_MODE': '0', 'CONGA_ACK_FEEDBACK': '0'}
                            folder = env.make(name, scenario=family, seed=seed, method='conweave',
                                              enabled=1, feedback=0, extra=extra)
                        j = json.loads((folder / 'job.json').read_text())
                        j.update(arm=arm, family=family, fit_seed=m['fit'], phase=stage,
                                 mode=8 if arm.startswith('rl') else 0, reselect=0, margin_us=0,
                                 interval_us=50, analysis_label=name, optimizer_seed=m['fit'],
                                 protocol_sha256=env.sha(task / 'protocol.json'))
                        env.dump(folder / 'job.json', j)
                    jobs.append(folder)
    env.dump(task / f'{stage}-jobs.json', [str(p) for p in jobs])
    return jobs


def execute(folder):
    result = folder / 'comparison-result.json'
    if result.exists():
        return json.loads(result.read_text())
    row = env.evaluate(folder)
    job = json.loads((folder / 'job.json').read_text())
    row.update({k: job[k] for k in ['arm', 'family', 'fit_seed', 'traffic_seed', 'phase']})
    if not row['valid']:
        row['offered'] = int(Path(job['traffic']).read_text().splitlines()[0])
        fct = folder / 'fct_output_file.txt'
        row['completion_lines_before_failure'] = len(fct.read_text().splitlines()) if fct.exists() else 0
        log = folder / 'log.txt'
        row['log_tail'] = log.read_text(errors='replace')[-2500:] if log.exists() else ''
    row['folder'] = str(folder)
    env.dump(result, row)
    print('RESULT', job['analysis_label'], 'VALID' if row['valid'] else 'FAILED',
          row.get('all_mean_us', row.get('error')), flush=True)
    return row


def summarize(task, rows, stage):
    result = dict(stage=stage, runs=len(rows), valid=sum(r['valid'] for r in rows), arms={}, comparisons={})
    for arm in sorted({r['arm'] for r in rows}):
        rr = [r for r in rows if r['arm'] == arm]
        result['arms'][arm] = dict(runs=len(rr), valid=sum(r['valid'] for r in rr),
                                 offered=sum(r['offered'] for r in rr),
                                 completed_in_valid_runs=sum(r['completed'] for r in rr if r['valid']),
                                 failures=[dict(folder=r['folder'], error=r.get('error')) for r in rr if not r['valid']])
    if stage in ['test', 'combined']:
        for baseline in sorted({r['arm'] for r in rows if r['arm'].startswith('conweave')}):
            for actor in ['rl_dcqcn', 'rl_dctcp']:
                pair = actor + '_vs_' + baseline
                subset = [r for r in rows if r['arm'] in [actor, baseline]]
                if not all(r['valid'] for r in subset):
                    result['comparisons'][pair] = dict(valid=False, reason='At least one failed or unfinished paired run; no complete-case FCT comparison.')
                    continue
                lookup = {(r['arm'], r['fit_seed'], r['family'], r['traffic_seed']): r for r in subset}
                stats = {}
                rng = np.random.default_rng(20260916)
                bs = rng.integers(0, len(TEST_SEEDS), (10000, len(TEST_SEEDS)))
                bf = rng.integers(0, len(FITS), (10000, len(FITS)))
                for metric in METRICS:
                    ratios = np.array([[[np.log(lookup[actor, fit, family, seed][metric] /
                                                     lookup[baseline, 0, family, seed][metric])
                                          for seed in TEST_SEEDS] for family in FAMILIES] for fit in FITS])
                    samples = np.array([ratios[f][:, :, s].mean() for f, s in zip(bf, bs)])
                    stats[metric] = dict(delta_pct=float(np.expm1(ratios.mean())*100),
                                         ci95_pct=(np.expm1(np.percentile(samples, [2.5, 97.5]))*100).tolist(),
                                         per_family_pct={f: float(np.expm1(ratios[:, i, :].mean())*100) for i, f in enumerate(FAMILIES)})
                result['comparisons'][pair] = dict(valid=True, statistics=stats)
    env.dump(task / f'{stage}-summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--ns3-binary', type=Path)
    parser.add_argument('--stage', choices=['smoke', 'test', 'sensitivity'], required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    task = args.experiment_dir.resolve()
    env.configure(task, args.ns3_binary or task / 'binaries/scratch/network-load-balance')
    protocol = initialize(task)
    jobs = prepare(task, protocol, args.stage)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, f) for f in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            env.dump(task / f'{args.stage}-progress.json', dict(done=len(rows), total=len(jobs), valid=sum(r['valid'] for r in rows)))
    rows.sort(key=lambda r: r['folder'])
    env.dump(task / f'{args.stage}-results.json', rows)
    print(json.dumps(summarize(task, rows, args.stage), indent=2), flush=True)
    if args.stage == 'sensitivity':
        primary = json.loads((task / 'test-results.json').read_text())
        env.dump(task / 'combined-results.json', primary + rows)
        summarize(task, primary + rows, 'combined')


if __name__ == '__main__':
    main()
