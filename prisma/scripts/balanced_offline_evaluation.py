#!/usr/bin/env python3
"""Balanced offline search with fixed-policy tail guards and fresh held-out tests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import shutil
import time

import coordinated_offline_evaluation as base
from coordinated_offline_evaluation import np, env, FAMILIES

SOURCE = Path('/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/coordinated-offline-20260917-s8a4h5j_/experiment')
TOPOS = ['symmetric', 'asymmetric']
METRICS = ['p95_us', 'p99_us']


def read(path):
    return json.loads(Path(path).read_text())


def train_cases(rnd):
    n = [32, 48, 64, 48][rnd % 4]
    return ([dict(family=f, ranks=0, size=0, load=.35) for f in FAMILIES[:4]] +
            [dict(family='alltoall', ranks=n, size=b) for b in [8192, 131072]] +
            [dict(family='allreduce', ranks=n, size=b) for n, b in [(32,1048576),(48,3145728)]])


def init(root):
    p = base.init(root)
    if p.get('campaign') != 'balanced-fixed-guard-v1':
        assert not list((root/'runs').iterdir())
        previous = read(SOURCE/'selection.json')
        old_models = []
        for item in previous['models']['real']:
            target = root/'models'/f"old_coordinated_f{item['fit']}.coord"
            shutil.copy2(item['path'], target)
            old_models.append(dict(fit=item['fit'], path=str(target), sha256=env.sha(target)))
        p.update(campaign='balanced-fixed-guard-v1', created_epoch=time.time(),
            rounds=16, fits=3, directions=4, top=2, checkpoints=[0,4,8,12,16],
            sigma=[.3]*4+[.15]*4+[.075]*4+[.04]*4,
            step=[.15]*4+[.10]*4+[.06]*4+[.03]*4, normalized_update_cap=.3,
            optimizer_seeds=[195101,195102,195103],
            train_seeds=list(range(195301,195317)), validation_seeds=[195501,195502],
            test_seeds=[195601,195602,195603,195604], smoke_seed=195299,
            train_cases=[dict(round=r, seed=195301+r, topologies=TOPOS, cases=train_cases(r)) for r in range(16)],
            validation_cases=base.cases('val'), test_cases=base.cases('test'),
            guard_tolerance_pct=.5,
            objective='Matched fixed conditional actor, same route seed. Each family/topology/metric: mean log(candidate/fixed), averaging seeds and cases. L=mean(all24cells)+2mean(max(cell-log1.005,0))+max(max(cell-log1.005,0)). Reward=-L. P95/P99 equal weights; six families and both topologies equal weights. CCT reported, not optimized.',
            training='From INITIAL,16 ARS rounds,4 antithetic directions,top2,3 fits,real and shuffled matched budgets. Each round includes both topologies,4 streams,small+large All-to-All,2 All-Reduce conditions;16 contexts. New traffic seed per round; ranks cycle32/48/64/48. No in-simulation parameter updates.',
            selection='One checkpoint shared across3fits per reward arm; feasible iff every family/topology/P95-or-P99 geometric-mean regression versus matched fixed <=0.5%. Minimize mean24cells among feasible checkpoints, earlier checkpoint on ties. Round0 allowed and always feasible. Aggregate guard is not a per-condition guarantee.',
            primary_endpoints='Fresh-test P95/P99 learned vs fixed and old trained; report all6families/topologies and unseen80ranks separately. Four independent traffic seeds;3 matched route/optimizer replicates; seed-block and fit bootstrap with family-equal weighting. Report regressions and CIs. No test-based selection.',
            formal_arms=['ecmp','wecmp','conga','conweave','fixed','old','learned','shuffled','last'],
            old_models=old_models, source_selection_sha256=env.sha(SOURCE/'selection.json'),
            training_native=13056, validation_native=1728, test_native=2280, smoke_native=16,
            implementation_change='Training/objective/selection only; identical deployed policy structure and native binary to previous campaign.')
        for key in ['screen_seeds','screen_cases','screen_native']:
            p.pop(key, None)
        assert p['binary_sha256'] == read(SOURCE/'protocol.json')['binary_sha256']
        assert p['library_sha256'] == read(SOURCE/'protocol.json')['library_sha256']
        env.dump(root/'protocol.json',p)
        for file in [Path(__file__),Path(base.__file__),Path(base.prior.__file__),Path(base.coll.__file__),Path(env.__file__)]:
            shutil.copy2(file,root/'source'/file.name)
    return p


def prepare(root,p,phase,c,seed,topology,label,fit=0,model=None,suffix=''):
    native = label not in base.BASELINES
    arm = 'learned' if native else label
    folder = base.prepare(root,p,phase,c,seed,topology,arm,fit,model,
                          suffix=f'_{label}{suffix}',packet_log=False)
    j=read(folder/'job.json')
    if j.get('balanced_arm') is None:
        j['balanced_arm']=label
        env.dump(folder/'job.json',j)
    assert j['balanced_arm']==label
    return folder


def execute(root,jobs,workers,label):
    rows=base.execute(root,jobs,workers,label)
    for r in rows:
        r['native_arm']=r['arm']
        r['arm']=read(Path(r['folder'])/'job.json')['balanced_arm']
    env.dump(root/f'{label}-results.json',rows)
    return rows


def objective(rows,refs,tolerance=.5):
    ref={(base.identity(r),r['fit']):r for r in refs if r['arm']=='fixed'}
    cells=defaultdict(list)
    for r in rows:
        assert r['all_complete']
        b=ref[base.identity(r),r['fit']]
        for m in METRICS:
            cells[r['family'],r['topology_name'],m].append(np.log(r[m]/b[m]))
    assert len(cells)==24
    detail=[dict(family=f,topology=t,metric=m,log_ratio=float(np.mean(cells[f,t,m])))
            for f in FAMILIES for t in TOPOS for m in METRICS]
    v=np.array([x['log_ratio'] for x in detail]);excess=np.maximum(v-np.log1p(tolerance/100),0)
    return dict(loss=float(v.mean()+2*excess.mean()+excess.max()),mean_loss=float(v.mean()),
                feasible=bool(v.max()<=np.log1p(tolerance/100)+1e-12),
                max_regression_pct=float(100*np.expm1(v.max())),cells=detail)


def smoke(root,p,workers):
    cs=[train_cases(0)[0],train_cases(0)[4],train_cases(0)[5],train_cases(0)[6]]
    jobs=[prepare(root,p,'smoke',c,p['smoke_seed'],t,a) for c in cs for t in TOPOS for a in ['fixed','conweave']]
    assert len(jobs)==p['smoke_native']
    execute(root,jobs,workers,'smoke')
    print('SMOKE_COMPLETE',flush=True)


def train(root,p,workers):
    theta={(a,f):base.INITIAL.copy() for a in ['real','shuffled'] for f in range(3)}
    for a,f in theta:
        path=root/'checkpoints'/f'{a}_f{f}_r0.coord'
        if not path.exists():base.export(theta[a,f],path)
    for rnd in range(p['rounds']):
        record=root/f'train-round{rnd}.json'
        if record.exists():
            for u in read(record)['updates']:theta[u['arm'],u['fit']]=np.array(u['after'])
            continue
        case=p['train_cases'][rnd]
        refs=[prepare(root,p,'train',c,case['seed'],t,'fixed',f) for c in case['cases'] for t in TOPOS for f in range(3)]
        baselines=execute(root,refs,workers,f'train-r{rnd}-references')
        jobs=[];owners={};directions={};permutations={}
        for fit in range(3):
            directions[fit]=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000).normal(size=(4,8))
            permutations[fit]=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000+500000).permutation(8)
        for arm in ['real','shuffled']:
            for fit in range(3):
                for d in range(4):
                    for sign in [-1,1]:
                        proposal=np.clip(theta[arm,fit]+sign*p['sigma'][rnd]*directions[fit][d]*base.SCALE,base.LOWER,base.UPPER)
                        model=base.export(proposal,root/'models'/f'train_{arm}_f{fit}_r{rnd}_d{d}_z{sign}.coord')
                        for c in case['cases']:
                            for t in TOPOS:
                                folder=prepare(root,p,'train',c,case['seed'],t,'train_'+arm,fit,model,f'_r{rnd}_d{d}_z{sign}')
                                jobs.append(folder);owners[folder.name]=(arm,fit,d,sign)
        assert len(jobs)==768 and len(refs)==48
        rows=execute(root,jobs,workers,f'train-r{rnd}');groups=defaultdict(list)
        for r in rows:groups[owners[r['name']]].append(r)
        updates=[]
        for arm in ['real','shuffled']:
            for fit in range(3):
                diagnostics=[objective(groups[arm,fit,d,s],baselines,p['guard_tolerance_pct']) for d in range(4) for s in [-1,1]]
                rewards=np.array([-x['loss'] for x in diagnostics])
                permutation=np.arange(8) if arm=='real' else permutations[fit]
                used=rewards[permutation].reshape(4,2)
                top=np.argsort(-used.max(axis=1),kind='stable')[:2]
                std=max(.001,float(used[top].std()))
                delta=p['step'][rnd]*np.mean((used[top,1]-used[top,0])[:,None]*directions[fit][top],axis=0)/std
                delta*=min(1.,p['normalized_update_cap']/max(1e-12,float(np.linalg.norm(delta))))
                before=theta[arm,fit].copy();theta[arm,fit]=np.clip(before+delta*base.SCALE,base.LOWER,base.UPPER)
                model=base.export(theta[arm,fit],root/'checkpoints'/f'{arm}_f{fit}_r{rnd+1}.coord')
                updates.append(dict(arm=arm,fit=fit,before=before.tolist(),after=theta[arm,fit].tolist(),raw_rewards=rewards.tolist(),
                    permutation=permutation.tolist(),directions=directions[fit].tolist(),normalized_delta=delta.tolist(),
                    selected_directions=top.tolist(),diagnostics=diagnostics,model=str(model),model_sha256=env.sha(model)))
        env.dump(record,dict(round=rnd,updates=updates,results_sha256=env.sha(root/f'train-r{rnd}-results.json')))
        env.dump(root/'task-progress.json',dict(stage='train',rounds_complete=rnd+1,rounds_total=p['rounds'],updated_epoch=time.time()))
        print('TRAIN_ROUND_COMPLETE',rnd+1,flush=True)
    env.dump(root/'training-complete.json',dict(rounds=p['rounds'],runs=p['training_native'],created_epoch=time.time()))


def validate(root,p,workers):
    if (root/'selection.json').exists():return
    assert (root/'training-complete.json').exists()
    jobs=[]
    for seed in p['validation_seeds']:
        for t in TOPOS:
            for c in p['validation_cases']:
                for fit in range(3):
                    jobs.append(prepare(root,p,'val',c,seed,t,'fixed',fit))
                    jobs.append(prepare(root,p,'val',c,seed,t,'old',fit,Path(p['old_models'][fit]['path'])))
                    for arm in ['real','shuffled']:
                        for rnd in p['checkpoints']:
                            jobs.append(prepare(root,p,'val',c,seed,t,'val_'+arm,fit,root/'checkpoints'/f'{arm}_f{fit}_r{rnd}.coord',f'_r{rnd}'))
    assert len(jobs)==p['validation_native'],len(jobs)
    rows=execute(root,jobs,workers,'validation');refs=[r for r in rows if r['arm']=='fixed'];scores=[]
    for arm in ['real','shuffled']:
        for rnd in p['checkpoints']:
            sub=[r for r in rows if r['arm']=='val_'+arm and r['name'].endswith(f'_r{rnd}')]
            scores.append(dict(arm=arm,round=rnd,**objective(sub,refs,p['guard_tolerance_pct'])))
    selected={arm:min([s for s in scores if s['arm']==arm and s['feasible']],key=lambda x:(x['mean_loss'],x['round'])) for arm in ['real','shuffled']}
    models={arm:[dict(fit=f,path=str(root/'checkpoints'/f"{arm}_f{f}_r{selected[arm]['round']}.coord"),
                sha256=env.sha(root/'checkpoints'/f"{arm}_f{f}_r{selected[arm]['round']}.coord")) for f in range(3)] for arm in selected}
    oldscore=objective([r for r in rows if r['arm']=='old'],refs,p['guard_tolerance_pct'])
    env.dump(root/'selection.json',dict(created_epoch=time.time(),selected=selected,scores=scores,models=models,old_score=oldscore,
         protocol_sha256=env.sha(root/'protocol.json'),validation_results_sha256=env.sha(root/'validation-results.json')))
    print('SELECTED',{a:dict(round=x['round'],mean_loss=x['mean_loss'],max_regression_pct=x['max_regression_pct']) for a,x in selected.items()},flush=True)


def test(root,p,workers):
    selection=read(root/'selection.json');jobs=[]
    specs=[(a,0,None) for a in base.BASELINES]
    for f in range(3):
        specs.extend([('fixed',f,None),('old',f,Path(p['old_models'][f]['path'])),
                      ('learned',f,Path(selection['models']['real'][f]['path'])),
                      ('shuffled',f,Path(selection['models']['shuffled'][f]['path'])),
                      ('last',f,root/'checkpoints'/f"real_f{f}_r{p['rounds']}.coord")])
    for seed in p['test_seeds']:
        for t in TOPOS:
            for c in p['test_cases']:
                for a,f,m in specs:jobs.append(prepare(root,p,'test',c,seed,t,a,f,m))
    assert len(jobs)==p['test_native'],len(jobs)
    execute(root,jobs,workers,'test')
    print('TEST_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--stage',choices=['smoke','train','validate','test','all'],required=True)
    ap.add_argument('--workers',type=int,default=20);args=ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=args.root.resolve();p=init(root)
    for stage in ['smoke','train','validate','test']:
        if args.stage in [stage,'all']:globals()[stage](root,p,args.workers)


if __name__=='__main__':main()
