#!/usr/bin/env python3
"""Continue reward-only path scoring from the three frozen round-12 actors.

Requires an isolated experiment directory, a compatible native executable,
and the completed from-scratch experiment. Never mutates the prior run.
"""
import argparse
import os
from pathlib import Path
import sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-dir',type=Path,required=True)
parser.add_argument('--ns3-binary',type=Path,help='Native executable; defaults to experiment-dir/binaries/frozen/scratch/network-load-balance')
parser.add_argument('--previous-experiment',type=Path,help='Completed from-scratch experiment directory')
parser.add_argument('--stage',choices=['all','setup','train','validate','test'],default='all')
args=parser.parse_args();TASK=args.experiment_dir.resolve()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import path_policy_search as policy
import path_policy_refinement as refine
import path_rollout as env
import json
import time

env.configure(TASK, args.ns3_binary)
read = env.read
write = env.write
make = env.make_trial
execute = env.execute_jobs
from collections import defaultdict
import shutil
import numpy as np

OLD=args.previous_experiment.resolve() if args.previous_experiment else TASK.parent/'cable-rl128-from-scratch-20260913'
FITS=[9101,9102,9103]
ARMS=['real','shuffled']
FAMILIES=['uniform','storage','sweep','mixed']
FAMS=FAMILIES+['mixed_fast']


def setup():
    if (TASK/'setup-complete.json').exists():return
    if not (TASK/'protocol.json').exists():
        old=json.loads((OLD/'final-selection.json').read_text())
        starts=[]
        for m in old['selected']['real']['models']:
            dst=TASK/'start-models'/f"s{m['fit']}.txt";dst.parent.mkdir(exist_ok=True)
            shutil.copy2(m['path'],dst);shutil.copy2(Path(m['path']).with_suffix('.npz'),dst.with_suffix('.npz'))
            assert env.sha(dst)==m['sha256']
            starts.append(dict(fit=m['fit'],theta=m['theta'],path=str(dst),sha256=env.sha(dst),source=m['path']))
        original=TASK/'fixed-models/original.txt';original.parent.mkdir(exist_ok=True)
        previous=TASK/'fixed-models/previous_selected.txt'
        for src,dst in [(OLD/'models/original_fixed.txt',original),(Path(old['selected']['fixed']['models'][0]['path']),previous)]:
            shutil.copy2(src,dst);shutil.copy2(src.with_suffix('.npz'),dst.with_suffix('.npz'))
        write('protocol.json',dict(created_epoch=time.time(),question='Does additional correctly associated reward learning improve the already learned round-12 policies?',
            hosts=128,starts=starts,prior_selection_sha256=env.sha(OLD/'final-selection.json'),
            scope='Initial path scoring only. Same eight measurements, score class, telemetry, flow holding and reward. No expert action targets, expert initialization or expert blending.',
            conditioning='Freeze clipped [.25,4] ratios of old input RMS to RMS measured in the previous real round-12 TRAINING monitor trajectories. Set perturbations on identically zero observations to zero. This changes exploration, never starting scores or simulator observations.',
            normalization_source=str(OLD/'normalization.json'),fits=FITS,arms=ARMS,rounds=24,directions=8,top=4,
            sigma=[.03]*8+[.012]*8+[.004]*8,step=[.03]*8+[.015]*8+[.0075]*8,cap=[.10]*8+[.06]*8+[.03]*8,
            optimization_rng='fit+20000',shuffle_rng='fit+30000',training_families=FAMILIES,
            training_traffic=list(range(84001,84025)),smoke_traffic=83999,
            reward='Same -100*mean(log(meanFCT/1ms)+.25*log(P99/1ms)), actual complete network events. A censored family gets -10000; simulator errors abort.',failure_return=-10000.,
            training_native_per_arm=4608,monitor_native_per_arm=288,
            shuffled='Start from the SAME learned round-12 actor for each fit; permute own 16 direction/sign returns before identical updates. Same rollout, normalization, perturbation and selection budget.',
            validation_traffic=[85001,85002,85003,85004],validation_rounds=[-1,5,11,17,23],validation_native_per_arm=240,
            selection='One additional round shared by all fits per arm, maximizing the same reward. Starting checkpoint allowed; no per-workload or per-fit selection.',
            fixed_grid=dict(remote=[.125,.25,.5],work=[.05,.1,.2],trend=[-.5,0,.5]),fixed_validation_runs=432,
            original_fixed=dict(path=str(original),sha256=env.sha(original)),previous_fixed=dict(path=str(previous),sha256=env.sha(previous)),
            test_traffic=list(range(86001,86013)),test_families=FAMS,
            test_rule='Generate all test traffic only after selection is frozen; no retraining, model selection or reward changes after test.',
            claims='Incremental learning requires mean-FCT 95% upper delta <0 vs paired starting actors AND paired shuffled continuation, P99 upper delta <+1% vs starts. Matching manual rules requires mean upper <+1% and P99 upper <+2% vs original, previous selected and newly selected fixed rules. Exceeding all fixed rules requires mean upper <0 vs all three, with the same P99 noninferiority constraint.',
            bootstrap='10000 paired blocks of 12 traffic seeds across 5 fixed workloads, plus paired resampling of 3 fit lineages for real/start/shuffled. Training and validation data selection are not bootstrapped.',
            environment='Same frozen simulator, 128 hosts + 8 leaf + 8 spine, asymmetric 1/.5Gbps uplinks, 194B/200us wire reports carrying destination prefixes, queues and rates; receiver-local freshness, local_change constant zero, DCTCP, IRN on, PFC off, RTO10/32ms, 50MiB buffers, 50ms injection plus 1s drain; pressure off.',
            conga='Original repository DRE50/flowlet100/aging500us, no added ACK/NACK or diagnostic feedback; never tuned.',default_changed=False,native_source_changed=False))
    p=read('protocol.json')
    if not (TASK/'conditioning.json').exists():
        oldscale=np.array(json.loads(Path(p['normalization_source']).read_text())['scale'])
        oldround=json.loads((OLD/'train-round11.json').read_text());squares=np.zeros(8);n=0;files={}
        for name in oldround['monitor_jobs']:
            f=Path(name);j=json.loads((f/'job.json').read_text())
            if j['label']!='real':continue
            assert j['phase']=='monitor' and j['traffic_seed']==81012
            a=np.loadtxt(f/'transitions.csv.candidates.csv',delimiter=',',skiprows=1,ndmin=2)
            x=a[:,5+np.array(policy.FEATURES)];squares+=(x*x).sum(0);n+=len(x)
            files[str(f/'transitions.csv.candidates.csv')]=env.sha(f/'transitions.csv.candidates.csv')
        factor,observed=refine.direction_scale(oldscale,squares,n)
        write('conditioning.json',dict(created_epoch=time.time(),original_scale=oldscale.tolist(),observed_rms=observed.tolist(),factor=factor.tolist(),count=n,sum_squares=squares.tolist(),training_sources=files))
    jobs=[];pairs=[]
    for m in p['starts']:
        for fam in FAMILIES:
            a=make(f"smoke_start_s{m['fit']}_{fam}",fam,p['smoke_traffic'],'start','smoke',m['path'],m['fit'])
            b=make(f"smoke_source_s{m['fit']}_{fam}",fam,p['smoke_traffic'],'source','smoke',m['source'],m['fit'])
            jobs += [a,b];pairs.append((a,b))
    write('smoke-jobs.json',list(map(str,jobs)));execute(jobs)
    for a,b in pairs:assert env.sha(a/'fct_output_file.txt')==env.sha(b/'fct_output_file.txt')
    write('setup-complete.json',dict(status='PASS',exact_pairs=[list(map(str,x)) for x in pairs]))
    print('SETUP PASS',read('conditioning.json')['factor'],flush=True)


def train():
    if (TASK/'training-status.json').exists() and read('training-status.json')['stage']=='complete':return
    p=read('protocol.json');c=read('conditioning.json');scale=np.array(c['original_scale']);factor=np.array(c['factor'])
    theta={(a,m['fit']):np.array(m['theta']) for a in ARMS for m in p['starts']}
    rng={s:np.random.default_rng(s+20000) for s in FITS};shufflers={s:np.random.default_rng(s+30000) for s in FITS}
    checkpoints=[];alljobs=[];monitors=[]
    for rnd in range(p['rounds']):
        directions={s:refine.conditioned_directions(rng[s],factor) for s in FITS}
        permutations={(a,s):shufflers[s].permutation(16) if a=='shuffled' else np.arange(16) for a in ARMS for s in FITS}
        if (TASK/f'train-round{rnd}.json').exists():
            rec=read(f'train-round{rnd}.json')
            for u in rec['updates']:theta[u['arm'],u['fit']]=np.array(u['theta'])
            checkpoints+=rec['updates'];alljobs+=rec['jobs'];monitors+=rec['monitor_jobs'];continue
        jobs=[];owners={}
        for arm in ARMS:
            for fit in FITS:
                for d in range(8):
                    for sign in [-1,1]:
                        t=theta[arm,fit]+sign*p['sigma'][rnd]*directions[fit][d]
                        model=policy.export(t,scale,TASK/'perturbations'/f'{arm}_s{fit}_r{rnd}_d{d}_sign{sign}.txt')
                        for fam in FAMILIES:
                            f=make(f'train_{arm}_s{fit}_r{rnd}_d{d}_sign{sign}_{fam}',fam,p['training_traffic'][rnd],arm,'training',model,fit)
                            jobs.append(f);owners[f.name]=(arm,fit,d,sign)
        alljobs+=list(map(str,jobs));write('training-jobs.json',alljobs);grouped=defaultdict(list)
        for r in execute(jobs):grouped[owners[r['name']]].append(r)
        updates=[];monitor_jobs=[]
        for arm in ARMS:
            for fit in FITS:
                raw=np.array([policy.reward(grouped[arm,fit,d,sign],p['failure_return']) for d in range(8) for sign in [-1,1]])
                before=theta[arm,fit].copy()
                after,detail=policy.update(before,directions[fit],raw,permutations[arm,fit],step=p['step'][rnd],top=p['top'],cap=p['cap'][rnd])
                theta[arm,fit]=after
                model=policy.export(after,scale,TASK/'checkpoints'/f'{arm}_s{fit}_r{rnd}.txt')
                updates.append(dict(arm=arm,fit=fit,round=rnd,before=before.tolist(),directions=directions[fit].tolist(),raw_rewards=raw.tolist(),permutation=permutations[arm,fit].tolist(),path=model,sha256=env.sha(model),**detail))
                for fam in FAMILIES:monitor_jobs.append(make(f'center_{arm}_s{fit}_r{rnd}_{fam}',fam,p['training_traffic'][rnd],arm,'monitor',model,fit))
        rr=execute(monitor_jobs);checkpoints+=updates;monitors+=list(map(str,monitor_jobs))
        write(f'train-round{rnd}.json',dict(round=rnd,jobs=list(map(str,jobs)),updates=updates,monitor_jobs=list(map(str,monitor_jobs)),monitor_rows=rr))
        write('monitor-jobs.json',monitors);write('training-status.json',dict(stage='running',round=rnd,checkpoints=checkpoints))
        print('ROUND',rnd+1,{a:round(policy.reward([r for r in rr if f'center_{a}_' in r['name']],p['failure_return']),3) for a in ARMS},flush=True)
    write('training-status.json',dict(stage='complete',checkpoints=checkpoints))


def validate():
    if (TASK/'final-selection.json').exists():return
    p=read('protocol.json');trained=read('training-status.json');assert trained['stage']=='complete';configs=[]
    for arm in ARMS:
        for rnd in p['validation_rounds']:
            models=[dict(**m,round=-1) for m in p['starts']] if rnd==-1 else [u for u in trained['checkpoints'] if u['arm']==arm and u['round']==rnd]
            configs.append(dict(id=f'{arm}_r{rnd}',arm=arm,round=rnd,models=models))
    for a in p['fixed_grid']['remote']:
        for w in p['fixed_grid']['work']:
            for t in p['fixed_grid']['trend']:
                cid=f'fixed_a{a}_w{w}_t{t}';weights=np.zeros(24);weights[[0,1,8,11]]=[1,a,w,a*t]
                path=TASK/'fixed-grid'/f'{cid}.txt';env.export_weights(weights,path)
                configs.append(dict(id=cid,arm='fixed',models=[dict(path=str(path),fit=0,sha256=env.sha(path))]))
    jobs=[];owners={}
    for c in configs:
        for m in c['models']:
            for ts in p['validation_traffic']:
                for fam in FAMILIES:
                    f=make(f"val_{c['id']}_s{m['fit']}_{fam}_{ts}",fam,ts,c['arm'],'validation',m['path'],m['fit']);jobs.append(f);owners[f.name]=c['id']
    write('validation-configs.json',configs);write('validation-jobs.json',list(map(str,jobs)));grouped=defaultdict(list)
    for r in execute(jobs):grouped[owners[r['name']]].append(r)
    scores=[dict(**c,reward=policy.reward(grouped[c['id']],p['failure_return']),all_complete=all(r['valid'] for r in grouped[c['id']])) for c in configs]
    selected={a:max((c for c in scores if c['arm']==a),key=lambda c:(c['reward'],c['id'])) for a in ARMS+['fixed']}
    write('validation-scores.json',scores);write('final-selection.json',dict(created_epoch=time.time(),selected=selected,protocol_sha256=env.sha(TASK/'protocol.json'),conditioning_sha256=env.sha(TASK/'conditioning.json'),training_status_sha256=env.sha(TASK/'training-status.json'),default_changed=False))
    print('FROZEN',{a:(v['id'],v['reward']) for a,v in selected.items()},flush=True)


def test():
    if (TASK/'test-status.json').exists():return
    p=read('protocol.json');frozen=read('final-selection.json');specs=[]
    for a in ARMS:specs += [dict(label=a,fit=m['fit'],model=m['path']) for m in frozen['selected'][a]['models']]
    specs += [dict(label='start',fit=m['fit'],model=m['path']) for m in p['starts']]
    specs += [dict(label=k,fit=0,model=p[k]['path']) for k in ['original_fixed','previous_fixed']]
    specs += [dict(label='strong_fixed',fit=0,model=frozen['selected']['fixed']['models'][0]['path']),dict(label='conga',fit=0,model=None)]
    jobs=[];index=[]
    for ts in p['test_traffic']:
        for fam in FAMS:
            for s in specs:
                f=make(f"test_{s['label']}_s{s['fit']}_{fam}_{ts}",fam,ts,s['label'],'test',s['model'],s['fit'],conga=s['label']=='conga')
                j=json.loads((f/'job.json').read_text());j['selection_sha256']=env.sha(TASK/'final-selection.json');env.dump(f/'job.json',j)
                assert Path(j['traffic']).stat().st_mtime>=frozen['created_epoch']
                jobs.append(f);index.append(dict(folder=str(f),label=s['label'],fit=s['fit'],family=fam,traffic=ts))
    write('test-jobs.json',list(map(str,jobs)));write('test-index.json',index);execute(jobs)
    write('test-status.json',dict(stage='complete',runs=len(jobs)));print('TEST COMPLETE',len(jobs),flush=True)


if __name__=='__main__':
    for name,fn in [('setup',setup),('train',train),('validate',validate),('test',test)]:
        if args.stage in ['all',name]:fn()
