#!/usr/bin/env python3
"""Frozen, paired tail-latency campaign for the current native path policy."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np
import path_rollout as env
import path_policy_search as policy
ROOT=Path(__file__).resolve().parents[2]
FAMILIES=['uniform','storage','mixed','mixed_fast']
LOADS=[.20,.35,.50]
SEEDS=list(range(88001,88009))
FITS=[9101,9102,9103]
CW={'CONWEAVE_REPLY_TIMEOUT_EXTRA':'200','CONWEAVE_TX_EXPIRY_TIME':'1000',
    'CONWEAVE_PATH_PAUSE_TIME':'16','CONWEAVE_EXTRA_VOQ_FLUSH_TIME':'64',
    'CONWEAVE_DEFAULT_VOQ_WAITING_TIME':'400','CONWEAVE_AGING_US':'2000000'}
COMMON={'CC_MODE':'1','HAS_WIN':'0','VAR_WIN':'0','SIMULATOR_EXTRA_TIME':'1.0',
        'IRN_RTO_LOW_US':'10000','IRN_RTO_HIGH_US':'32000',
        'KMAX_MAP':'7 200000000 400 500000000 400 750000000 400 1000000000 400 2000000000 400 2500000000 400 4000000000 400',
        'KMIN_MAP':'7 200000000 100 500000000 100 750000000 100 1000000000 100 2000000000 100 2500000000 100 4000000000 100',
        'PMAX_MAP':'7 200000000 0.20 500000000 0.20 750000000 0.20 1000000000 0.20 2000000000 0.20 2500000000 0.20 4000000000 0.20'}


def make_trace(task,family,seed,load):
    path=task/'traces'/f'{family}_s{seed}_l{load:.2f}.txt'
    if path.exists():return path
    path.parent.mkdir(exist_ok=True)
    if family=='storage':
        command=['/usr/bin/python3',str(env.NS/'traffic_gen/traffic_gen.py'),'-c',str(env.NS/'traffic_gen/AliStorage2019.txt'),
                 '-n','128','-l',str(load),'-b','1G','-t','.05','-o',str(path),'--seed',str(seed)]
        subprocess.run(command,cwd=env.NS,check=True,stdout=subprocess.DEVNULL)
    else:
        rng=random.Random(seed)
        sizes,weights=([2000,8000,64000,256000],[.5,.35,.1,.05]) if family=='uniform' else ([2000,8000,64000,1048576],[.5,.35,.13,.02])
        mean=sum(s*w for s,w in zip(sizes,weights));rate=128*load*1e9/(mean*8)
        phases=[];t=0.
        if family!='uniform':
            while t<.05:
                phases.append((t,rng.randrange(8),rng.choice([0.,.25,.5])))
                t+=rng.uniform(.0002,.001) if family=='mixed_fast' else rng.uniform(.001,.005)
        t=0.;phase=0;flows=[]
        while True:
            t+=rng.expovariate(rate)
            if t>=.05:break
            s=rng.randrange(128);d=rng.randrange(127)
            if d>=s:d+=1
            if family!='uniform':
                while phase+1<len(phases) and phases[phase+1][0]<=t:phase+=1
                if rng.random()<phases[phase][2]:d=phases[phase][1]*16+rng.randrange(16)
                if d==s:d=(d+1)%128
            flows.append((round((2+t)*1e9),s,d,rng.choices(sizes,weights)[0]))
        flows.sort()
        path.write_text(str(len(flows))+'\n'+''.join(f'{s} {d} 3 {size} {ns/1e9:.9f}\n' for ns,s,d,size in flows))
    env.dump(path.with_suffix('.json'),dict(family=family,seed=seed,load=load,sha256=env.sha(path)))
    return path


def initialize(task,binary):
    env.configure(task,binary)
    if (task/'protocol.json').exists():return json.loads((task/'protocol.json').read_text())
    for name in ['topologies','models','source','runs']:(task/name).mkdir(exist_ok=True)
    topology=env.NS/'config/leaf_spine_128_100G_asym_OS2.txt'
    asym=topology.read_text();sym=[]
    for i,line in enumerate(asym.splitlines()):
        f=line.split()
        if i>=2 and len(f)>=5 and int(f[0])>=128 and int(f[1])>=128:f[2]='750Mbps'
        sym.append(' '.join(f))
    # Native IRN BDP is topology-specific and verified against calculated BDP.
    (task/'topologies/leaf_spine_128_100G_asym_OS2_asymmetric.txt').write_text(asym)
    (task/'topologies/leaf_spine_128_750M_sym.txt').write_text('\n'.join(sym)+'\n')
    manifest=json.loads((ROOT/'prisma/experiments/path_learning/selected.json').read_text())
    scale=np.array(manifest['normalization']);models=[]
    for m,start in zip(manifest['models'],manifest['starting_models']):
        path=task/'models'/f"rl_{m['fit']}.txt";shutil.copy2(ROOT/'prisma/experiments/path_learning'/m['path'],path)
        assert env.sha(path)==m['sha256']
        models.append(dict(arm='rl',fit=m['fit'],path=str(path),sha256=env.sha(path)))
        for arm,theta in [('start',np.array(start['theta'])),('no_remote',np.array(m['theta'])),('no_history',np.array(m['theta']))]:
            if arm=='no_remote':theta[[1,3,4]]=0
            if arm=='no_history':theta[[4,5,6,7]]=0
            out=Path(policy.export(theta,scale,task/'models'/f"{arm}_{m['fit']}.txt"))
            models.append(dict(arm=arm,fit=m['fit'],path=str(out),sha256=env.sha(out)))
    for arm,theta in [('fixed',np.array([1.,.25,0,0,0,0,0,.1])),('neutral',np.zeros(8))]:
        out=Path(policy.export(theta,np.ones(8),task/'models'/f'{arm}.txt'))
        models.append(dict(arm=arm,fit=0,path=str(out),sha256=env.sha(out)))
    source=task/'source/nsdi_tail_evaluation.py';shutil.copy2(__file__,source)
    p=dict(created_epoch=time.time(),question='How do frozen completion-reward path policies affect P95/P99 across balanced-capacity symmetric and asymmetric fabrics?',
           binary=str(binary),binary_sha256=env.sha(binary),source_sha256=env.sha(source),
           models=models,topologies=['symmetric','asymmetric'],hosts=128,leaf=8,spine=8,
           topology_note='Both fabrics have 6 Gbps of aggregate uplink capacity per leaf: 8x0.75 versus 4x1+4x0.5 Gbps; host links 1Gbps, per-link delay 1us.',
           families=FAMILIES,loads=LOADS,traffic_seeds=SEEDS,execution_seed=100,
           main_baselines=['ecmp','wecmp','drill','conga','letflow','conweave'],main_controls=['fixed'],
           ablations=['start','no_remote','no_history','neutral'],ablation_load=.35,
           ablation_note='No retraining. Set coefficient groups to zero; preserve feedback packets, cost and all other mechanisms. Start is the paired round-12 checkpoint; no new shuffled training in this campaign.',
           conweave=CW,common=COMMON,freeze='No reward change, training, model selection or test-driven timer search. All 3 published actor lineages retained.',
           telemetry='194B every200us per spine-leaf feedback link;1024B quantization;receiver-local freshness;local_change constant zero;computation delay zero.',
           protocol_delta='All algorithms use DCQCN and identical ECN maps (max mark probability .2 at 0.5/.75/1Gbps); this differs from the old mixed marking map. All workload nominal host loads use the listed values.',
           metrics='All offered-flow P95/P99. If a run completes with censored flows, report a percentile only when both interpolation ranks lie below every unfinished-flow lower bound; otherwise mark it unidentified. Failed simulator runs have no latency statistics. Always show completion/failure. Makespan requires all flows complete.',
           statistics='Paired log-ratios; 10000 traffic-seed block resamples with paired actor-lineage resampling; no training or validation uncertainty.',
           primary_runs=1920,ablation_runs=640)
    env.dump(task/'protocol.json',p);return p


def prepare(task,p,stage):
    jobs=[]
    if stage=='smoke':
        grids=[(t,l,f,88999) for t in p['topologies'] for l in [.20,.50] for f in ['uniform']]
    else:grids=[(t,l,f,s) for t in p['topologies'] for l in ([.35] if stage=='ablate' else LOADS) for f in FAMILIES for s in SEEDS]
    for topo,load,family,seed in grids:
        traffic=make_trace(task,family,seed,load)
        # make() calls this synchronously while jobs are constructed, before
        # native workers launch; trace data itself remains immutable.
        env.trace=lambda *a, _path=traffic, **kw:_path
        models=[m for m in p['models'] if m['arm'] in (p['ablations'] if stage=='ablate' else ['rl','fixed'])]
        if stage=='smoke':models=[m for m in models if m['fit'] in [0,9101]]
        specs=[dict(arm=a,fit=0,path=None) for a in ([] if stage=='ablate' else p['main_baselines'])]+models
        for m in specs:
            name=f"{stage}_{topo}_{family}_l{load:.2f}_s{seed}_{m['arm']}_f{m['fit']}"
            folder=task/'runs'/name
            if folder.exists():jobs.append(folder);continue
            topology_file='leaf_spine_128_750M_sym.txt' if topo=='symmetric' else 'leaf_spine_128_100G_asym_OS2_asymmetric.txt'
            extra={**COMMON,'TOPOLOGY_FILE':str(task/'topologies'/topology_file)}
            is_rl=m['path'] is not None
            if is_rl:
                extra.update(RL_SWITCH_MODE='8',RL_PATH_RESELECT='0',RL_GATE_MODE='0',RL_PRESSURE_ENABLE='0',
                             RL_PRESSURE_MASK='0',RL_SWITCH_EPSILON='0',RL_SWITCH_MARGIN_NS='0',
                             RL_SWITCH_OUTPUT=str(folder/'transitions.csv'),RL_SWITCH_MODEL=m['path'])
            else:
                extra['RL_SWITCH_MODE']='0';extra['CONGA_ACK_FEEDBACK']='0'
                if m['arm']=='conweave':extra.update(CW)
            folder=env.make(name,scenario=family,seed=seed,method='oracle' if is_rl else m['arm'],
                            remote=int(is_rl),feedback=int(is_rl),period=200,quantum=1024 if is_rl else 0,
                            gap=0 if is_rl else 100,extra=extra,enabled=1)
            job=json.loads((folder/'job.json').read_text())
            job.update(arm=m['arm'],fit=m['fit'],topology_name=topo,load=load,family=family,phase=stage,
                       model=m['path'],model_sha256=env.sha(Path(m['path'])) if is_rl else None,
                       topology=extra['TOPOLOGY_FILE'],topology_sha256=env.sha(Path(extra['TOPOLOGY_FILE'])),
                       protocol_sha256=env.sha(task/'protocol.json'))
            env.dump(folder/'job.json',job);jobs.append(folder)
    env.dump(task/f'{stage}-jobs.json',list(map(str,jobs)));return jobs


def offered_flows(job):
    ports=defaultdict(lambda:10000);dports=defaultdict(lambda:100);out={}
    for line in Path(job['traffic']).read_text().splitlines()[1:]:
        s,d,pg,b,t=line.split();s,d=int(s),int(d)
        out[s,d,ports[s],dports[d]]=dict(size=int(b),start_ns=round(float(t)*1e9))
        ports[s]+=1;dports[d]+=1
    return out


def run_one(folder):
    output=folder/'tail-result.json'
    if output.exists():return json.loads(output.read_text())
    job=json.loads((folder/'job.json').read_text());row=dict(job,native_ok=False,analysis_ok=False)
    try:
        env.execute(folder)
        row['native_ok']=True
        summary,flows=env.load(folder);offered=offered_flows(job)
        row.update(native_ok=True,offered=len(offered),completed=len(flows),all_complete=len(flows)==len(offered))
        times=np.array([v['fct_us'] for v in flows.values()]);unfinished=[v for k,v in offered.items() if k not in flows]
        bounds=np.array([(3049999999-v['start_ns'])/1000 for v in unfinished])
        order=np.sort(times)
        for q in [50,95,99,99.9]:
            pos=(len(offered)-1)*q/100;hi=int(np.ceil(pos));lo=int(np.floor(pos))
            identified=hi<len(order) and (not len(bounds) or order[hi]<bounds.min())
            row[f'p{q:g}_identified']=bool(identified)
            row[f'p{q:g}_us']=float(order[lo]*(hi-pos)+order[hi]*(pos-lo)) if identified and hi!=lo else (float(order[hi]) if identified else None)
        row['mean_us']=float(times.mean()) if row['all_complete'] else None
        row['makespan_us']=max(v['start_ns']+v['fct_us']*1000-2e9 for v in flows.values())/1000 if row['all_complete'] else None
        row['fct_sha256']=env.sha(folder/'fct_output_file.txt')
        row['completion_fraction']=len(flows)/len(offered)
        if (folder/'transitions.csv.audit.json').exists():
            audit=json.loads((folder/'transitions.csv.audit.json').read_text())
            assert audit['overrides']==0 and audit['late_packet_actions']==0
            assert audit['switch']==0
            if row['all_complete']:
                assert audit['unfinished_episodes']==0
                assert audit['transitions']==audit['terminals']==sum(audit['path_assignments'])
            row['actor_audit']=audit
        if len(bounds):row['unfinished_lower_bound_min_us']=float(bounds.min())
        row['groups']={}
        for name,test in [('small',lambda x:x<=8000),('medium',lambda x:8000<x<=64000),('large',lambda x:x>64000)]:
            values=[v['fct_us'] for v in flows.values() if test(v['size'])]
            count=sum(test(v['size']) for v in offered.values())
            row['groups'][name]=dict(offered=count,completed=len(values),p95_us=float(np.percentile(values,95)) if values and len(values)==count else None,
                                    p99_us=float(np.percentile(values,99)) if values and len(values)==count else None)
        row['analysis_ok']=True
    except Exception as exc:
        row.update(error=repr(exc),all_complete=False)
        row['offered']=int(Path(job['traffic']).read_text().splitlines()[0])
        f=folder/'fct_output_file.txt';row['completed']=len(f.read_text().splitlines()) if f.exists() else 0
        row['completion_fraction']=row['completed']/row['offered']
        log=folder/'log.txt';row['log_tail']=log.read_text(errors='replace')[-2200:] if log.exists() else ''
    row['folder']=str(folder);env.dump(output,row)
    print('DONE',folder.name,row['native_ok'],row['completed'],row['offered'],flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True);parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--stage',choices=['smoke','main','ablate'],required=True);parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    task=args.experiment_dir.resolve();p=initialize(task,args.binary.resolve());jobs=prepare(task,p,args.stage)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,f) for f in jobs]
        for f in as_completed(futures):
            rows.append(f.result());env.dump(task/f'{args.stage}-progress.json',dict(done=len(rows),total=len(jobs),native_ok=sum(r['native_ok'] for r in rows),all_complete=sum(r['all_complete'] for r in rows)))
    rows.sort(key=lambda r:r['name']);env.dump(task/f'{args.stage}-results.json',rows)
    print('FINISHED',args.stage,len(rows),Counter((r['arm'],r['all_complete']) for r in rows),flush=True)


if __name__=='__main__':main()
