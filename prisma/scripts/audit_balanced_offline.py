#!/usr/bin/env python3
"""Verify native metrics, frozen inputs, ARS updates, and validation-only selection."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import itertools
import math
from pathlib import Path

import balanced_offline_evaluation as run
from balanced_offline_evaluation import np,env,read,FAMILIES,TOPOS,METRICS


@lru_cache(maxsize=None)
def sha(path):
    return env.sha(Path(path))


def audit_row(r):
    folder=Path(r['folder']);j=read(folder/'job.json');status=read(folder/'status.json')
    assert status['state']=='complete' and status['returncode']==0
    assert j['protocol_sha256']==sha(str(folder.parents[1]/'protocol.json'))
    for key in ['binary','config','traffic','topology','model','coord_model','plan']:
        if j.get(key):assert sha(j[key])==j[key+'_sha256'],(r['name'],key)
    assert j['balanced_arm']==r['arm']
    assert j['extra']['ONLINE_ENABLE']==j['extra']['FAC_ENABLE']=='0'
    values=np.loadtxt(folder/'fct_output_file.txt',dtype=np.int64).reshape(-1,8)
    assert len(values)==r['offered']==r['completed'] and np.all(values[:,6]>0)
    fct=values[:,6]/1000
    errors=[abs(float(np.percentile(fct,q))-r[m]) for q,m in [(95,'p95_us'),(99,'p99_us')]]
    assert max(errors)<1e-8 and abs(float(fct.mean())-r['mean_us'])<1e-8
    if j.get('plan'):
        # The event/dependency audit already runs for every simulation; verify CCT
        # independently from the raw start+duration columns here.
        cct=(int(np.max(values[:,5]+values[:,6]))-run.base.coll.START_NS)/1000
        assert abs(cct-r['makespan_us'])<1e-8
    if j.get('coord_model'):
        assert j['extra']['RL_PATH_RESELECT']=='0'
        a=read(folder/'transitions.csv.coordinated-audit.json')
        assert a['parameter_updates']==0
        assert np.array_equal(a['theta'],list(map(float,Path(j['coord_model']).read_text().split()[1:])))
        assert a['balance']==a['work']==a['conditional']==1
        actor=read(folder/'transitions.csv.audit.json')
        assert actor['unfinished_episodes']==actor.get('path_changes',0)==actor.get('path_reselections',0)==0
        assert sum(x['decisions'] for x in a['leaves'])==actor['terminals']
    return dict(flows=len(values),max_error=max(errors),phase=r['phase'],name=r['name'])


def score_independently(rows,refs):
    lookup={(run.base.identity(r),r['fit']):r for r in refs};cells=defaultdict(list)
    for r in rows:
        b=lookup[run.base.identity(r),r['fit']]
        for m in METRICS:cells[r['family'],r['topology_name'],m].append(float(np.log(r[m])-np.log(b[m])))
    assert len(cells)==24
    means=np.array([sum(v)/len(v) for v in cells.values()]);over=np.maximum(means-np.log(1.005),0)
    return dict(loss=float(means.mean()+2*over.mean()+max(over)),mean_loss=float(means.mean()),
                max_regression_pct=float(100*np.expm1(max(means))),feasible=bool(max(means)<=np.log(1.005)+1e-12))


def training(root,p):
    updates=0;max_error=0.
    previous={(a,f):run.base.INITIAL.copy() for a in ['real','shuffled'] for f in range(3)}
    for rnd in range(p['rounds']):
        rows=read(root/f'train-r{rnd}-results.json');refs=read(root/f'train-r{rnd}-references-results.json')
        rec=read(root/f'train-round{rnd}.json')
        assert env.sha(root/f'train-r{rnd}-results.json')==rec['results_sha256']
        for u in rec['updates']:
            arm,fit=u['arm'],u['fit'];assert np.array_equal(previous[arm,fit],u['before'])
            rewards=[]
            for d in range(4):
                for sign in [-1,1]:
                    sub=[r for r in rows if r['arm']=='train_'+arm and r['fit']==fit and r['name'].endswith(f'_d{d}_z{sign}')]
                    assert len(sub)==16
                    rewards.append(-score_independently(sub,refs)['loss'])
            error=float(np.max(np.abs(np.array(rewards)-u['raw_rewards'])));max_error=max(max_error,error);assert error<1e-12
            rng=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000)
            directions=rng.normal(size=(4,8));assert np.array_equal(directions,u['directions'])
            permutation=np.arange(8) if arm=='real' else np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000+500000).permutation(8)
            assert np.array_equal(permutation,u['permutation'])
            used=np.array(u['raw_rewards'])[permutation].reshape(4,2)
            top=sorted(range(4),key=lambda k:-max(used[k]))[:2];assert top==u['selected_directions']
            delta=sum((used[d,1]-used[d,0])*directions[d] for d in top)/2
            delta*=p['step'][rnd]/max(.001,float(np.std(used[top])))
            delta*=min(1.,p['normalized_update_cap']/max(1e-12,float(np.linalg.norm(delta))))
            assert np.allclose(delta,u['normalized_delta'],rtol=0,atol=1e-12)
            after=np.clip(np.array(u['before'])+delta*run.base.SCALE,run.base.LOWER,run.base.UPPER)
            assert np.allclose(after,u['after'],rtol=0,atol=1e-12)
            assert env.sha(u['model'])==u['model_sha256']
            assert np.array_equal(list(map(float,Path(u['model']).read_text().split()[1:])),u['after'])
            previous[arm,fit]=np.array(u['after']);updates+=1
    return dict(updates=updates,max_reward_recalculation_error=max_error,all_updates_reproduced=True)


def selection(root,p):
    rows=read(root/'validation-results.json');sel=read(root/'selection.json');refs=[r for r in rows if r['arm']=='fixed']
    assert env.sha(root/'validation-results.json')==sel['validation_results_sha256']
    results=[]
    for arm in ['real','shuffled']:
        for rnd in p['checkpoints']:
            s=score_independently([r for r in rows if r['arm']=='val_'+arm and r['name'].endswith(f'_r{rnd}')],refs)
            saved=next(x for x in sel['scores'] if x['arm']==arm and x['round']==rnd)
            for key in ['loss','mean_loss','max_regression_pct']:assert abs(s[key]-saved[key])<1e-10
            assert s['feasible']==saved['feasible'];results.append(dict(arm=arm,round=rnd,**s))
        feasible=[r for r in results if r['arm']==arm and r['feasible']]
        chosen=min(feasible,key=lambda r:(r['mean_loss'],r['round']))
        assert chosen['round']==sel['selected'][arm]['round']
    for r in read(root/'test-results.json'):
        j=read(Path(r['folder'])/'job.json')
        assert j['selection_sha256']==env.sha(root/'selection.json')
        assert Path(j['traffic']).stat().st_mtime>=sel['created_epoch']
        if r['arm'] in ['learned','shuffled']:
            model=sel['models']['real' if r['arm']=='learned' else 'shuffled'][r['fit']]
            assert j['coord_model_sha256']==model['sha256']
    return dict(scores=results,test_generated_after_selection=True,selection_reproduced=True)


def statistics(root):
    """Check paired point estimates and enumerate the small exact bootstrap law."""
    rows=read(root/'test-results.json');summary=read(root.parent/'analysis/summary.json')
    def weights(n):
        result=[]
        for counts in itertools.product(range(n+1),repeat=n):
            if sum(counts)!=n:continue
            prob=math.factorial(n)/(n**n)
            for count in counts:prob/=math.factorial(count)
            result.append((np.array(counts,dtype=float)/n,prob))
        assert abs(sum(p for w,p in result)-1)<1e-12
        return result
    seed_weights=weights(4);fit_weights=weights(3);checks=[]
    for q in summary['comparisons']:
        if q['actor'] not in ['learned','last'] or q['baseline'] not in ['fixed','old','conweave','shuffled'] or q['topology'] is not None:continue
        if (q['scope'],q['family']) not in [('familiar',None),('familiar','alltoall'),('familiar','allreduce'),('unseen','alltoall'),('unseen','allreduce')]:continue
        metric=q['metric']
        sub=[r for r in rows if r[metric] is not None and r['unseen']==(q['scope']=='unseen') and (q['family'] is None or r['family']==q['family'])]
        refs={(run.base.identity(r),r['fit']):r for r in sub if r['arm']==q['baseline']}
        groups=defaultdict(list)
        for r in sub:
            if r['arm']!=q['actor']:continue
            ref=refs[run.base.identity(r),0 if q['baseline']=='conweave' else r['fit']]
            groups[r['seed'],r['fit'],r['family']].append(math.log(r[metric]/ref[metric]))
        families=sorted({key[2] for key in groups})
        matrix=np.array([[sum(sum(groups[s,f,fam])/len(groups[s,f,fam]) for fam in families)/len(families) for f in range(3)] for s in sorted({key[0] for key in groups})])
        point=100*math.expm1(float(matrix.mean()));assert abs(point-q['change_pct'])<1e-9
        law=sorted((float(sw@matrix@fw),sp*fp) for sw,sp in seed_weights for fw,fp in fit_weights)
        cumulative=np.cumsum([p for v,p in law]);values=np.array([v for v,p in law])
        ci=[float(100*np.expm1(values[np.searchsorted(cumulative,t)])) for t in [.025,.975]]
        checks.append(dict(actor=q['actor'],baseline=q['baseline'],scope=q['scope'],family=q['family'],metric=metric,
            independently_recomputed_change_pct=point,exact_discrete_bootstrap_ci95_pct=ci,
            monte_carlo_ci95_pct=q['ci95_pct'],max_endpoint_difference_pct=max(abs(a-b) for a,b in zip(ci,q['ci95_pct']))))
    assert len(checks)==120
    env.dump(root.parent/'statistics-verification.json',checks)
    return dict(primary_comparisons=len(checks),all_point_estimates_verified=True,
                exact_bootstrap_count_patterns=350,max_monte_carlo_endpoint_difference_pct=max(x['max_endpoint_difference_pct'] for x in checks))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    root=args.root;p=read(root/'protocol.json');rows=[]
    assert env.sha(root/'binaries/scratch/network-load-balance')==p['binary_sha256']
    for filename,expected in p['library_sha256'].items():
        assert env.sha(root/'binaries'/filename)==expected
    for name in ['smoke-results.json','validation-results.json','test-results.json']:
        rows.extend(read(root/name))
    for rnd in range(p['rounds']):
        rows.extend(read(root/f'train-r{rnd}-results.json'));rows.extend(read(root/f'train-r{rnd}-references-results.json'))
    expected=sum(p[x+'_native'] for x in ['smoke','training','validation','test'])
    assert len(rows)==expected and len({r['name'] for r in rows})==expected
    assert set(p['train_seeds']).isdisjoint(p['validation_seeds']+p['test_seeds'])
    assert set(p['validation_seeds']).isdisjoint(p['test_seeds'])
    for filename in ['original-immutable-inputs.json','frozen-inputs.json']:
        for path,h in read(root.parent/filename).items():assert env.sha(path)==h
    flows=0;max_error=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(audit_row,rows),1):
            flows+=r['flows'];max_error=max(max_error,r['max_error'])
            if i%500==0:print('AUDITED',i,len(rows),flush=True)
    result=dict(runs=len(rows),flow_instances=flows,max_quantile_error_us=max_error,
        immutable_inputs_verified=True,all_native_metrics_verified=True,all_parameter_updates_zero=True,
        training=training(root,p),validation=selection(root,p),statistics=statistics(root))
    env.dump(root.parent/'verification.json',result)
    print('AUDIT_COMPLETE',len(rows),flows,flush=True)


if __name__=='__main__':main()
