#!/usr/bin/env python3
"""Reconstruct frozen decisions and quantify paired All-to-All interventions."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from diagnose_offline_alltoall import ARCHIVE,env

def hash8(key,seed):
    h=seed;mask=0xffffffff
    for k in [key&mask,key>>32]:
        k=k*0xcc9e2d51&mask;k=((k<<15)|(k>>17))&mask;k=k*0x1b873593&mask
        h^=k;h=((h<<13)|(h>>19))&mask;h=(h*5+0xe6546b64)&mask
    h^=8;h^=h>>16;h=h*0x85ebca6b&mask;h^=h>>13;h=h*0xc2b2ae35&mask;h^=h>>16
    return h%8

def trace(folder):
    j=json.loads((folder/'job.json').read_text())
    a=np.loadtxt(folder/'transitions.csv.candidates.csv',delimiter=',',skiprows=1).reshape(-1,8,29)
    ids=a[:,0,0].astype(int);assert np.all(a[:,:,0]==ids[:,None]);assert np.all(a[:,:,2]==np.arange(8))
    # Initial-path analysis only. A different selector can revisit a flow.
    assert len(set(ids))==len(ids)
    chosen=a[:,:,3].argmax(axis=1);assert np.all(a[:,:,3].sum(axis=1)==1)
    x=a[:,:,5:];n=len(a);ii=np.arange(n)
    v=np.array(list(map(float,Path(j['model']).read_text().split()[3:])))
    w=v[:768].reshape(24,32);bias=v[768:800];out=v[800:864].reshape(32,2);c=v[864:]
    score=np.maximum(0,x@w+bias)@(out[:,1]-out[:,0])+c[1]-c[0]
    error=float(abs(score-a[:,:,4]).max());assert error<1e-8
    # Native ties use tolerance 1e-12, so use its loop rather than plain argmin.
    best=np.zeros(n,dtype=int)
    for k in range(1,8):best=np.where(score[:,k]<score[ii,best]-1e-12,k,best)
    assert np.array_equal(best,chosen)
    starts={}
    for line in (folder/'collective-events.txt').read_text().splitlines():
        z=line.split()
        if z[0]=='start':starts[int(z[1])]=list(map(int,z[2:]))
    inc=np.zeros((8,8));outgoing=np.zeros((8,8));rates={};first_delays=[]
    for q,fid in enumerate(ids):
        t,s,d,sp,dp,b=starts[fid];sl=s//16;dl=d//16;assert sl!=dl
        key=((s+(sp<<16))<<32)+(d+(dp<<16));offset=hash8(key,128+sl)
        # BFS visits the same spine pointer order for every destination leaf.
        # Recover that stable order, not numeric spine IDs.
        for k in range(8):
            path=(k+offset)%8;serial=x[q,k,2]
            if path in rates:assert abs(rates[path]-serial)<1e-10
            rates[path]=serial
        path=(int(chosen[q])+offset)%8;inc[dl,path]+=b;outgoing[sl,path]+=b
        first_delays.append((a[q,0,1]-t)/1000)
    rate=np.array([.008/rates[k] for k in range(8)]) # Gbps
    inload=inc/rate[None,:];outload=outgoing/rate[None,:]
    picked=x[ii,chosen]
    result=dict(name=j['name'],arm=j['arm'],fit=j['fit'],topology=j['topology_name'],ranks=j['ranks'],size=j['size'],seed=j['seed'],decisions=n,
        score_max_error=error,score_and_action_reconstruction=True,
        decision_us_quantiles=np.percentile((a[:,0,1]-2e9)/1000,[0,50,95,99,100]).tolist(),
        source_wait_us_quantiles=np.percentile(first_delays,[0,50,95,99,100]).tolist(),
        chosen_age_us_quantiles=np.percentile(picked[:,4]*1000,[0,50,95,99,100]).tolist(),
        known_fraction=float(picked[:,14].mean()),fast_fraction=float(np.mean(picked[:,2]<.012)) if j['topology_name']=='asymmetric' else None,
        incoming_max_over_mean=float(inload.max()/inload.mean()),outgoing_max_over_mean=float(outload.max()/outload.mean()),
        incoming_mean_leaf_max_over_mean=float(np.mean(inload.max(axis=1)/inload.mean(axis=1))),
        incoming_cv=float(inload.std()/inload.mean()),incoming_bytes=inc.tolist(),path_rate_gbps=rate.tolist(),
        chosen_remote_us=float(picked[:,1].mean()*1000),chosen_local_us=float(picked[:,0].mean()*1000))
    # Candidate-wise contribution spread is descriptive, not a performance attribution.
    effective=np.zeros(24)
    for feature in range(24):
        e=np.zeros(24);e[feature]=1
        effective[feature]=np.maximum(0,e@w+bias)@(out[:,1]-out[:,0])+c[1]-c[0]
    result['coefficients']=effective.tolist()
    result['term_spread_mean']=(x*effective).std(axis=1).mean(axis=0).tolist()
    result['same_state_action_change_without_feature']={}
    for group,features in {'remote':[1,5],'history':[7,8],'port':[8],'destination':[7],'serialization':[2,3]}.items():
        altered=score-(x[:,:,features]*effective[features]).sum(axis=2);b=np.zeros(n,dtype=int)
        for k in range(1,8):b=np.where(altered[:,k]<altered[ii,b]-1e-12,k,b)
        result['same_state_action_change_without_feature'][group]=float(np.mean(b!=chosen))
    return result

def aggregate(rows):
    result=[]
    for arm in sorted({r['arm'] for r in rows}):
        for topo in ['all','symmetric','asymmetric']:
            for ranks in [0,32,64]:
                sel=[r for r in rows if r['arm']==arm and (topo=='all' or r['topology_name']==topo) and (ranks==0 or r['ranks']==ranks)]
                if sel:result.append(dict(arm=arm,topology=topo,ranks=ranks,n=len(sel),**{m:float(100*np.expm1(np.mean([np.log(r[m]/r['reference_metrics'][m]) for r in sel]))) for m in ['p95_us','p99_us','makespan_us']}))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--traces',action='store_true');p.add_argument('--archive-all',action='store_true');args=p.parse_args();root=args.root
    rows=[json.loads(f.read_text()) for f in (root/'runs').glob('*/diagnosis-result.json')]
    env.dump(root/'aggregate.json',aggregate(rows))
    if args.traces or args.archive_all:
        folders=[ARCHIVE/'formal/runs'/f'main_alltoall_p64_b65536_s89001_{t}_{a}_f{fit}' for t in ['symmetric','asymmetric'] for a,fit in [('fixed',0),('rl',9101),('rl',9102),('rl',9103)]]
        folders += [Path(r['folder']) for r in rows if r['seed']==89001 and r['ranks']==64 and r['size']==65536 and r['arm'] not in ['reselect','replay']]
        if args.archive_all:
            original=json.loads((ARCHIVE/'analysis/collective-runs.json').read_text())
            folders=[ARCHIVE/'formal/runs'/r['name'] for r in original if r['family']=='alltoall' and r['arm'] in ['rl','fixed']]
        out=[]
        for f in folders:
            out.append(trace(f));print('TRACE',f.name,flush=True)
        env.dump(root/('archive-decision-analysis.json' if args.archive_all else 'decision-analysis.json'),out)
if __name__=='__main__':main()
