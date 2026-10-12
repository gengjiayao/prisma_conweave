#!/usr/bin/env python3
"""Frozen, dependency-driven All-to-All and ring All-Reduce evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import path_rollout as env
from nsdi_tail_evaluation import COMMON, CW

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/nsdi-refresh-20260917-aomgdw2h')
SEEDS = list(range(89001, 89009))
ARMS = ['ecmp', 'wecmp', 'drill', 'conga', 'letflow', 'conweave', 'fixed', 'rl']
SIZES = {'alltoall': [1024, 8192, 65536], 'allreduce': [131072, 1048576, 8388608]}
START_NS = 2_000_000_000


def generate(family, ranks, size, seed):
    """Topologically ordered transfers with explicit local dependencies."""
    assert family in SIZES and ranks % 8 == 0 and 8 <= ranks <= 128
    rng = random.Random(seed)
    hosts = [16*rack+h for rack in range(8) for h in rng.sample(range(16), ranks//8)]
    rng.shuffle(hosts)
    ready = [START_NS+rng.randrange(5001) for _ in hosts]
    flows = []
    def add(rank, dest, nbytes, stage, chunk, parents):
        flows.append(dict(id=len(flows), src=hosts[rank], dst=hosts[dest], pg=3,
                          bytes=nbytes, release_ns=ready[rank], delay_ns=0,
                          stage=stage, rank=rank, chunk=chunk, parents=parents))
    if family == 'alltoall':
        for rank in range(ranks):
            for offset in range(1, ranks):
                dest = (rank+offset) % ranks
                add(rank, dest, size, 0, dest, [])
        assert len({(f['src'], f['dst']) for f in flows}) == ranks*(ranks-1)
    else:
        assert size % ranks == 0
        for stage in range(2*(ranks-1)):
            for rank in range(ranks):
                parents = [] if stage == 0 else [(stage-1)*ranks+rank, (stage-1)*ranks+(rank-1)%ranks]
                add(rank, (rank+1)%ranks, size//ranks, stage, (rank-stage)%ranks, parents)
        assert len(flows) == 2*ranks*(ranks-1)
    return dict(family=family, ranks=ranks, size=size, seed=seed, hosts=hosts,
                ready_ns=ready, start_ns=START_NS, flows=flows,
                total_bytes=sum(f['bytes'] for f in flows))


def verify_ring_semantics(ranks):
    """Symbolically propagate contributor sets, including the phase boundary."""
    data = [{c: {r} for c in range(ranks)} for r in range(ranks)]
    for step in range(2*(ranks-1)):
        outgoing = [(r, (r-step)%ranks, set(data[r][(r-step)%ranks])) for r in range(ranks)]
        for rank, chunk, value in outgoing:
            dest = (rank+1)%ranks
            if step < ranks-1:
                assert not (data[dest][chunk] & value), 'A contribution would be counted twice'
                data[dest][chunk] |= value
            else:
                assert len(value) == ranks, 'All-gather must forward a fully reduced chunk'
                data[dest][chunk] = value
    assert all(v == set(range(ranks)) for chunks in data for v in chunks.values())


def write_plan(task, family, ranks, size, seed):
    stem = f'{family}_p{ranks}_b{size}_s{seed}'
    meta = task/'plans'/f'{stem}.json'
    if meta.exists():
        return json.loads(meta.read_text())
    x = generate(family, ranks, size, seed)
    plan, trace = meta.with_suffix('.dag'), meta.with_suffix('.txt')
    plan.write_text(str(len(x['flows']))+'\n'+''.join(
        ' '.join(map(str, [f['id'], f['src'], f['dst'], f['pg'], f['bytes'], f['release_ns'],
                          f['delay_ns'], len(f['parents'])]+f['parents']))+'\n' for f in x['flows']))
    # Legacy file supplies total offered count; the DAG controls actual starts.
    trace.write_text(str(len(x['flows']))+'\n'+''.join(
        f"{f['src']} {f['dst']} 3 {f['bytes']} {f['release_ns']/1e9:.9f}\n" for f in x['flows']))
    x.update(plan=str(plan), plan_sha256=env.sha(plan), traffic=str(trace), traffic_sha256=env.sha(trace))
    env.dump(meta, x)
    return x


def initialize(task):
    task.mkdir(parents=True, exist_ok=True)
    for name in ['binaries', 'plans', 'models', 'topologies', 'source', 'runs']:
        (task/name).mkdir(exist_ok=True)
    binary = task/'binaries/scratch/network-load-balance'
    if not binary.exists():
        shutil.copytree(BASE/'binaries', task/'binaries', dirs_exist_ok=True, symlinks=True)
        shutil.copy2(env.NS/'build-cable/scratch/network-load-balance', binary)
    env.configure(task, binary)
    if (task/'protocol.json').exists():
        p = json.loads((task/'protocol.json').read_text())
        assert env.sha(binary) == p['binary_sha256']
        return p
    for filename in ['fixed.txt']+[f'rl_{fit}.txt' for fit in [9101, 9102, 9103]]:
        shutil.copy2(BASE/'paper/models'/filename, task/'models'/filename)
    for f in (BASE/'experiment/topologies').glob('*.txt'):
        shutil.copy2(f, task/'topologies'/f.name)
    for f in [Path(__file__), env.NS/'scratch/network-load-balance.cc', env.NS/'scratch/collective-scheduler.h']:
        shutil.copy2(f, task/'source'/f.name)
    p = dict(created_epoch=time.time(), families=list(SIZES), ranks=[32,64], sizes=SIZES,
             seeds=SEEDS, topologies=['symmetric','asymmetric'], models=[9101,9102,9103],
             binary=str(binary), binary_sha256=env.sha(binary), source_sha256=env.sha(__file__),
             common={**COMMON, 'SIMULATOR_EXTRA_TIME':'5.0'}, conweave=CW,
             primary_runs=1920, jitter_ns=[0,5000], job_start_ns=START_NS,
             placement='Equal participant count in every rack; randomized hosts and rank permutation per seed.',
             alltoall='Direct concurrent transfers for all ordered distinct rank pairs; size is bytes per peer; local self-copy excluded.',
             allreduce='Single-ring reduce-scatter then all-gather, 2(P-1) rounds, size/P bytes per transfer. Each send waits for its previous send and predecessor incoming transfer to finish. No global per-round barrier.',
             dependency='Sender-confirmed QP completion triggers readiness; conservative network dependency model; zero reduction compute/launch delay; no GPU or NCCL implementation is modeled.',
             freeze='All six baselines, three frozen RL fits, and same-information fixed rule; no retraining, test-driven policy selection or timer tuning.',
             primary_metric='Job completion time from 2 s to final confirmed transfer completion.',
             secondary_metrics=['Per-transfer P95 FCT','Per-transfer P99 FCT','Completion fraction'],
             statistics='Equal-weight paired log ratios across rank counts, sizes, topologies and seeds; 10000 seed-block bootstrap draws with paired fit resampling. All three fits retained.',
             admission='For an entire workload family: all runs complete, aggregate job time >=5% lower than both CONGA and adapted ConWeave with 95% intervals entirely below zero, and neither aggregate P95 nor P99 worse than either baseline. No individual-cell selection. Report every tested family and the fixed-rule comparison in the evidence report even if admission fails.',
             references=['https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node101.htm',
                         'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html'])
    p['model_hashes'] = {f.name:env.sha(f) for f in (task/'models').glob('*.txt')}
    env.dump(task/'protocol.json',p)
    for n in [4,8,32,128]:
        verify_ring_semantics(n)
    env.dump(task/'generator-verification.json', dict(symbolic_ring_allreduce=[4,8,32,128],passed=True))
    return p


def prepare(task, p, phase):
    jobs=[]
    seeds=[89999] if phase=='smoke' else p['seeds']
    ranks=[32] if phase=='smoke' else p['ranks']
    for family in p['families']:
        for rank_count in ranks:
            for size in (p['sizes'][family][:1] if phase=='smoke' else p['sizes'][family]):
                for seed in seeds:
                    plan=write_plan(task,family,rank_count,size,seed)
                    for topo in p['topologies']:
                        specs=[(a,0) for a in ARMS if a!='rl']+[('rl',fit) for fit in ([9101] if phase=='smoke' else p['models'])]
                        for arm,fit in specs:
                            name=f'{phase}_{family}_p{rank_count}_b{size}_s{seed}_{topo}_{arm}_f{fit}'
                            folder=task/'runs'/name
                            if folder.exists():
                                jobs.append(folder);continue
                            env.trace=lambda *a, _path=Path(plan['traffic']), **kw:_path
                            topo_file=task/'topologies'/('leaf_spine_128_750M_sym.txt' if topo=='symmetric' else 'leaf_spine_128_100G_asym_OS2_asymmetric.txt')
                            model=task/'models'/(f'rl_{fit}.txt' if arm=='rl' else 'fixed.txt') if arm in ['rl','fixed'] else None
                            extra={**p['common'],'TOPOLOGY_FILE':str(topo_file),'COLLECTIVE_PLAN_FILE':plan['plan'],
                                   'COLLECTIVE_EVENT_FILE':str(folder/'collective-events.txt')}
                            if model:
                                extra.update(RL_SWITCH_MODE='8',RL_PATH_RESELECT='0',RL_GATE_MODE='0',RL_PRESSURE_ENABLE='0',
                                             RL_PRESSURE_MASK='0',RL_SWITCH_EPSILON='0',RL_SWITCH_MARGIN_NS='0',
                                             RL_SWITCH_OUTPUT=str(folder/'transitions.csv'),RL_SWITCH_MODEL=str(model))
                            else:
                                extra.update(RL_SWITCH_MODE='0',CONGA_ACK_FEEDBACK='0')
                                if arm=='conweave':extra.update(CW)
                            folder=env.make(name,scenario='uniform',seed=seed,method='oracle' if model else arm,
                                            remote=int(model is not None),feedback=int(model is not None),period=200,
                                            quantum=1024 if model else 0,gap=0 if model else 100,extra=extra,enabled=1)
                            job=json.loads((folder/'job.json').read_text())
                            job.update(arm=arm,fit=fit,family=family,ranks=rank_count,size=size,seed=seed,topology_name=topo,
                                       phase=phase,plan=plan['plan'],plan_sha256=plan['plan_sha256'],
                                       plan_metadata=str(Path(plan['plan']).with_suffix('.json')),
                                       topology=str(topo_file),topology_sha256=env.sha(topo_file),
                                       model=str(model) if model else None,model_sha256=env.sha(model) if model else None,
                                       protocol_sha256=env.sha(task/'protocol.json'))
                            env.dump(folder/'job.json',job);jobs.append(folder)
    env.dump(task/f'{phase}-jobs.json',[str(f) for f in jobs]);return jobs


def audit(folder):
    job=json.loads((folder/'job.json').read_text())
    job.setdefault('seed',job['traffic_seed'])
    plan=json.loads(Path(job['plan_metadata']).read_text())
    for key in ['binary','config','traffic','plan','topology','model']:
        if job.get(key):assert env.sha(job[key])==job[key+'_sha256'],key
    starts={};ends={};keys={}
    for line in (folder/'collective-events.txt').read_text().splitlines():
        x=line.split();idx=int(x[1]);t=int(x[2]);assert 0<=idx<len(plan['flows'])
        if x[0]=='start':
            assert idx not in starts
            src,dst,sport,dport,size=map(int,x[3:]);f=plan['flows'][idx]
            assert [src,dst,size]==[f['src'],f['dst'],f['bytes']]
            key=(src,dst,sport,dport);assert key not in keys
            starts[idx]=t;keys[key]=idx
        else:
            assert x[0]=='finish' and idx in starts and idx not in ends
            assert t>starts[idx];ends[idx]=t
    for idx,t in starts.items():
        f=plan['flows'][idx]
        assert t>=f['release_ns']
        assert all(dep in ends and t>=ends[dep]+f['delay_ns'] for dep in f['parents'])
    values={}
    for line in (folder/'fct_output_file.txt').read_text().splitlines():
        x=list(map(int,line.split()));assert len(x)==8
        idx=keys[tuple(x[:4])];assert idx not in values
        assert x[4]==plan['flows'][idx]['bytes'] and x[5]==starts[idx]
        assert x[6]==ends[idx]-starts[idx] and x[6]>0
        values[idx]=x[6]/1000
    assert len(values)==len(ends)
    complete=len(values)==len(plan['flows'])
    stage_finish={}
    for idx,finish in ends.items():
        stage=str(plan['flows'][idx]['stage'])
        stage_finish[stage]=max(stage_finish.get(stage,0),(finish-START_NS)/1000)
    row={k:job[k] for k in ['name','arm','fit','family','ranks','size','seed','topology_name','phase']}
    row.update(native_ok=True,analysis_ok=True,all_complete=complete,offered=len(plan['flows']),
               launched=len(starts),completed=len(values),completion_fraction=len(values)/len(plan['flows']),
               makespan_us=(max(ends.values())-START_NS)/1000 if complete else None,
               p95_us=float(np.percentile(list(values.values()),95)) if complete else None,
               p99_us=float(np.percentile(list(values.values()),99)) if complete else None,
               mean_us=float(np.mean(list(values.values()))) if complete else None,
               total_bytes=plan['total_bytes'],fct_sha256=env.sha(folder/'fct_output_file.txt'),
               event_sha256=env.sha(folder/'collective-events.txt'),plan_sha256=job['plan_sha256'],
               config_sha256=job['config_sha256'],binary_sha256=job['binary_sha256'],model_sha256=job['model_sha256'],
               dependency_edges_checked=sum(len(plan['flows'][i]['parents']) for i in starts),
               stage_max_finish_us=stage_finish,folder=str(folder))
    actor=folder/'transitions.csv.audit.json'
    if actor.exists():
        a=json.loads(actor.read_text());assert a['overrides']==a['late_packet_actions']==a['switch']==0
        if complete:
            assert a['unfinished_episodes']==0
            assert a['transitions']==a['terminals']==sum(a['path_assignments'])
        row['actor_audit']=a
    return row


def run_one(folder):
    output=folder/'collective-result.json'
    if output.exists():return json.loads(output.read_text())
    try:
        status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('state')!='complete':env.execute(folder)
        row=audit(folder)
    except Exception as exc:
        job=json.loads((folder/'job.json').read_text())
        job.setdefault('seed',job['traffic_seed'])
        row={k:job[k] for k in ['name','arm','fit','family','ranks','size','seed','topology_name','phase']}
        row.update(native_ok=(json.loads((folder/'status.json').read_text())['returncode']==0),
                   analysis_ok=False,all_complete=False,error=repr(exc),folder=str(folder))
    env.dump(output,row)
    print('AUDIT',folder.name,row.get('all_complete'),row.get('makespan_us'),row.get('error',''),flush=True)
    return row


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--experiment-dir',type=Path,required=True)
    a.add_argument('--phase',choices=['smoke','main'],required=True);a.add_argument('--workers',type=int,default=16)
    args=a.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    task=args.experiment_dir.resolve();p=initialize(task);jobs=prepare(task,p,args.phase);rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_one,f) for f in jobs]):
            rows.append(future.result())
            env.dump(task/f'{args.phase}-progress.json',dict(done=len(rows),total=len(jobs),
                      all_complete=sum(r['all_complete'] for r in rows),analysis_ok=sum(r['analysis_ok'] for r in rows)))
    env.dump(task/f'{args.phase}-results.json',sorted(rows,key=lambda r:r['name']))
    if not all(r['all_complete'] and r['analysis_ok'] for r in rows):sys.exit(1)


if __name__=='__main__':main()
