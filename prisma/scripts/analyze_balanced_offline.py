#!/usr/bin/env python3
"""Full held-out comparisons for balanced offline training; never selects on test."""
import argparse
from collections import defaultdict
from pathlib import Path
import json

import balanced_offline_evaluation as run
from balanced_offline_evaluation import np, env, FAMILIES, TOPOS, METRICS, read

ARMS=['ecmp','wecmp','conga','conweave','fixed','old','learned','shuffled','last']
LABELS=dict(zip(ARMS,['ECMP','Weighted ECMP','CONGA','ConWeave*','Fixed initial policy',
    'Previous training','Balanced training','Shuffled rewards','Final round 16']))
COLORS=dict(zip(ARMS,['#929ba2','#a89260','#9379a6','#c36943','#647a9b',
    '#bc879f','#087f86','#9b9a62','#704866']))


def scoped(rows,scope='familiar',family=None,topology=None):
    return [r for r in rows if (scope=='all' or r['unseen']==(scope=='unseen')) and
            (family is None or r['family']==family) and (topology is None or r['topology_name']==topology)]


def diagnose_search(root,p):
    records=[]
    for rnd in range(p['rounds']):
        for u in read(root/f'train-round{rnd}.json')['updates']:
            if u['arm']!='real':continue
            rewards=u['raw_rewards']
            records.append(dict(round=rnd+1,fit=u['fit'],best_reward=max(rewards),mean_reward=float(np.mean(rewards)),
                positive_candidates=sum(v>1e-12 for v in rewards),feasible_candidates=sum(v['feasible'] for v in u['diagnostics']),
                update_norm=float(np.linalg.norm(u['normalized_delta'])),parameters_changed=u['before']!=u['after']))
    signal=dict(real_updates=len(records),perturbation_candidates=len(records)*8,
        candidates_better_than_fixed=sum(x['positive_candidates'] for x in records),
        updates_with_any_better_candidate=sum(x['positive_candidates']>0 for x in records),
        feasible_candidates=sum(x['feasible_candidates'] for x in records),
        changed_parameter_centers=sum(x['parameters_changed'] for x in records),records=records,
        note='Zero reward is the matched fixed INITIAL. Perturbations are not combined ARS updates. Centers before/after each update are not evaluated on the same new batch; these counts do not measure the causal effect of each update.')
    env.dump(root.parent/'search-signal.json',signal)
    rows=read(root/'validation-results.json');refs={(run.base.identity(r),r['fit']):r for r in rows if r['arm']=='fixed'}
    cells=defaultdict(list)
    for r in rows:
        if r['arm']!='val_real' or not r['name'].endswith(f"_r{p['rounds']}"):continue
        ref=refs[run.base.identity(r),r['fit']]
        for m in METRICS:cells[r['family'],r['topology_name'],m,r['seed'],r['fit']].append(np.log(r[m]/ref[m]))
    effects=[dict(family=f,topology=t,metric=m,seed=s,fit=k,log_ratio=float(np.mean(v)),change_pct=float(100*np.expm1(np.mean(v)))) for (f,t,m,s,k),v in sorted(cells.items())]
    env.dump(root.parent/'validation-seed-effects.json',effects)
    return {k:v for k,v in signal.items() if k!='records'}


def compare(rows,actor,baseline,metric,scope='familiar',family=None,topology=None):
    subset=[r for r in scoped(rows,scope,family,topology) if r[metric] is not None]
    a=[r for r in subset if r['arm']==actor];b=[r for r in subset if r['arm']==baseline]
    if not a or not b:return None
    afits=sorted({r['fit'] for r in a});bfits=sorted({r['fit'] for r in b})
    seeds=sorted({r['seed'] for r in a});assert seeds==sorted({r['seed'] for r in b})
    families=sorted({r['family'] for r in a});assert families==sorted({r['family'] for r in b})
    def matrix(v,fits):
        groups=defaultdict(list)
        for r in v:groups[r['seed'],r['fit'],r['family']].append(np.log(r[metric]))
        return np.array([[np.mean([np.mean(groups[s,f,fam]) for fam in families]) for f in fits] for s in seeds])
    x=matrix(a,afits);y=matrix(b,bfits)
    paired=actor not in run.base.BASELINES and baseline not in run.base.BASELINES
    rng=np.random.default_rng(19560917)
    si=rng.integers(0,len(seeds),(10000,len(seeds)))
    fi=rng.integers(0,len(afits),(10000,len(afits)))
    fj=fi if paired else rng.integers(0,len(bfits),(10000,len(bfits)))
    draws=x[si[:,:,None],fi[:,None,:]].mean(axis=(1,2))-y[si[:,:,None],fj[:,None,:]].mean(axis=(1,2))
    return dict(actor=actor,baseline=baseline,metric=metric,scope=scope,family=family,topology=topology,
        change_pct=float(100*np.expm1(x.mean()-y.mean())),ci95_pct=(100*np.expm1(np.percentile(draws,[2.5,97.5]))).tolist(),
        paired_fit=paired,seeds=seeds,actor_fits=afits,baseline_fits=bfits)


def analyze(root):
    p=read(root/'protocol.json');selection=read(root/'selection.json');rows=read(root/'test-results.json')
    assert len(rows)==p['test_native'] and all(r['all_complete'] for r in rows)
    # Require every predeclared seed/condition/arm/fit, with no duplicates or omissions.
    keys=[(r['seed'],r['family'],r['ranks'],r['size'],r['topology_name'],r['arm'],r['fit']) for r in rows]
    expected={(s,c['family'],c['ranks'],c['size'],t,a,f) for s in p['test_seeds'] for c in p['test_cases'] for t in TOPOS for a in ARMS for f in ([0] if a in run.base.BASELINES else range(3))}
    assert len(keys)==len(set(keys)) and set(keys)==expected
    comparisons=[]
    for actor in ARMS:
        for baseline in ['conweave','fixed','old','shuffled']:
            if actor==baseline or (actor not in ['learned','last'] and baseline!='conweave'):continue
            for scope in ['familiar','unseen','all']:
                for family in [None]+FAMILIES:
                    for topology in [None]+TOPOS:
                        for metric in METRICS+['makespan_us']:
                            q=compare(rows,actor,baseline,metric,scope,family,topology)
                            if q:comparisons.append(q)
    condition=[]
    for a in ARMS:
        for t in TOPOS:
            for c in p['test_cases']:
                for m in METRICS+(['makespan_us'] if c['family'] in ['alltoall','allreduce'] else []):
                    selected=[r for r in rows if (r['arm'],r['topology_name'],r['family'],r['ranks'],r['size'])==(a,t,c['family'],c['ranks'],c['size'])]
                    condition.append(dict(arm=a,topology=t,family=c['family'],ranks=c['ranks'],size=c['size'],unseen=c.get('unseen',False),metric=m,geomean_us=float(np.exp(np.mean([np.log(r[m]) for r in selected])))))
    out=root.parent/'analysis';out.mkdir(exist_ok=True)
    familiar=scoped(rows,'familiar');fixed=[r for r in familiar if r['arm']=='fixed']
    test_guards=[]
    for actor in ['old','learned','shuffled','last']:
        for fit in [None,0,1,2]:
            sub=[r for r in familiar if r['arm']==actor and (fit is None or r['fit']==fit)]
            test_guards.append(dict(actor=actor,fit=fit,**run.objective(sub,fixed,p['guard_tolerance_pct'])))
    summary=dict(runs=len(rows),flow_instances=sum(r['offered'] for r in rows),all_complete=True,comparisons=comparisons,test_guards=test_guards,search_diagnostics=diagnose_search(root,p),
        protocol_sha256=env.sha(root/'protocol.json'),selection_sha256=env.sha(root/'selection.json'),
        test_results_sha256=env.sha(root/'test-results.json'),selection=selection,
        statistics='Geometric means of matched-condition ratios, equal family weights. Paired seed-block bootstrap and paired route-fit resampling for conditional actors; baseline algorithms have one fit. 10000 resamples; only4 independent traffic seeds, so intervals have limited resolution. No flow-level pseudo-replication; intervals exploratory and not simultaneous.')
    env.dump(out/'condition-results.json',condition)
    env.dump(out/'summary.json',summary)
    plot(out,summary,condition)
    for baseline in ['fixed','old','conweave']:
        for scope,fam in [('familiar',None),('familiar','alltoall'),('unseen','alltoall'),('familiar','allreduce')]:
            print(baseline,scope,fam,[(q['metric'],q['change_pct'],q['ci95_pct']) for q in comparisons if (q['actor'],q['baseline'],q['scope'],q['family'],q['topology'])==('learned',baseline,scope,fam,None) and q['metric'] in METRICS],flush=True)
    print('ANALYSIS_COMPLETE',flush=True)


def plot(out,summary,condition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    selected_round=summary['selection']['selected']['real']['round']
    LABELS['learned']=('Selected initial policy (r0)' if selected_round==0 else f'Balanced training (r{selected_round})')
    if summary['selection']['selected']['shuffled']['round']==0:
        LABELS['shuffled']='Shuffled control (r0)'
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.dpi':180,
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    def get(actor,baseline,metric,scope='familiar',family=None,topology=None):
        return next(q for q in summary['comparisons'] if (q['actor'],q['baseline'],q['metric'],q['scope'],q['family'],q['topology'])==(actor,baseline,metric,scope,family,topology))
    # Main attribution, including improvements and regressions, with uncertainty.
    groups=[('familiar',None,'All six families')]+[('familiar',f,{'mixed_fast':'Fast shift','alltoall':'All-to-All','allreduce':'All-Reduce'}.get(f,f.capitalize())) for f in FAMILIES]+[('unseen',f,'Unseen '+('All-to-All' if f=='alltoall' else 'All-Reduce')) for f in ['alltoall','allreduce']]
    panels=[('learned',f'Validation-selected policy (round {selected_round})','training_attribution'),
            ('last','Final round 16'+(' (also selected)' if selected_round==16 else ' (not selected)'),'final_round_attribution')]
    for actor,title,name in panels:
        fig,axes=plt.subplots(1,3,figsize=(16,5.4),layout='constrained')
        for ax,baseline in zip(axes,['fixed','old','conweave']):
            for i,(scope,fam,label) in enumerate(groups):
                for metric,color,offset in [('p95_us','#087f86',-.14),('p99_us','#c36943',.14)]:
                    q=get(actor,baseline,metric,scope,fam);v=q['change_pct'];lo,hi=q['ci95_pct']
                    ax.errorbar(v,i+offset,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=color,capsize=2,ms=4,label=metric[:3].upper() if i==0 else None)
            ax.set_yticks(range(len(groups)),[x[2] for x in groups]);ax.invert_yaxis();ax.axvline(0,c='#7d878d',lw=1)
            ax.set_xlabel('Completion time change (%)');ax.set_title('Versus '+LABELS[baseline]);ax.grid(axis='x',alpha=.15);ax.legend()
        fig.suptitle(title+': negative means faster; 95% bootstrap intervals',fontweight='bold')
        save(fig,name)
    # All algorithms and the final checkpoint, for both tail metrics and topologies.
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    show=[a for a in ARMS if a!='conweave']
    vals=[get(a,'conweave',m,'familiar',f,t)['change_pct'] for a in show for m in METRICS for f in FAMILIES for t in TOPOS]
    lim=max(10,max(abs(v) for v in vals));norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    for i,t in enumerate(TOPOS):
        for j,m in enumerate(METRICS):
            ax=axes[i,j];a=np.array([[get(actor,'conweave',m,'familiar',f,t)['change_pct'] for f in FAMILIES] for actor in show])
            im=ax.imshow(a,cmap='RdBu_r',norm=norm,aspect='auto')
            for row in range(len(show)):
                for col in range(6):ax.text(col,row,f'{a[row,col]:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(a[row,col])>.7*lim else '#1d2930')
            ax.set_yticks(range(len(show)),[LABELS[x] for x in show]);ax.set_xticks(range(6),[x[2] for x in groups[1:7]],rotation=30,ha='right');ax.set_title(t.capitalize()+' / '+m[:3].upper())
    fig.colorbar(im,ax=axes,label='Change vs ConWeave (%)',shrink=.75)
    fig.suptitle('Every algorithm and traffic family; common color scale; negative means faster',fontweight='bold');save(fig,'all_algorithms')
    for family in ['alltoall','allreduce']:
        cs=sorted({(r['ranks'],r['size'],r['unseen']) for r in condition if r['family']==family})
        initial_selected=selected_round==0 and summary['selection']['selected']['shuffled']['round']==0
        curve_arms=[a for a in ARMS if not initial_selected or a not in ['learned','shuffled']]
        if initial_selected:
            # Merge only after checking actual measurements, not merely labels.
            fixed={(r['topology'],r['ranks'],r['size'],r['metric']):r['geomean_us'] for r in condition if r['family']==family and r['arm']=='fixed'}
            for r in condition:
                if r['family']==family and r['arm'] in ['learned','shuffled']:
                    assert np.isclose(r['geomean_us'],fixed[r['topology'],r['ranks'],r['size'],r['metric']],rtol=1e-12,atol=1e-9)
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for i,t in enumerate(TOPOS):
            for j,m in enumerate(METRICS+['makespan_us']):
                ax=axes[i,j]
                for a in curve_arms:
                    y=[next(r['geomean_us']/1000 for r in condition if (r['arm'],r['topology'],r['family'],r['ranks'],r['size'],r['metric'])==(a,t,family,n,b,m)) for n,b,u in cs]
                    label='Initial = selected = shuffled (r0)' if a=='fixed' and initial_selected else LABELS[a]
                    color=COLORS['learned'] if a=='fixed' and initial_selected else COLORS[a]
                    ax.plot(range(len(cs)),y,label=label,c=color,marker='o',ms=3,lw=2 if a in ['learned','conweave','fixed'] else 1,alpha=1 if a in ['learned','conweave','fixed'] else .7)
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_xticks(range(len(cs)),[f'{n} / {b//1024}'+('*' if u else '') for n,b,u in cs],rotation=35,ha='right')
                ax.set_xlabel('Ranks / KiB');ax.set_ylabel(t.capitalize()+' time (ms, log scale)');ax.grid(alpha=.16);ax.set_title(m[:3].upper() if m!='makespan_us' else 'Collective completion')
        for j in range(3):
            low=min(ax.get_ylim()[0] for ax in axes[:,j]);high=max(ax.get_ylim()[1] for ax in axes[:,j])
            for ax in axes[:,j]:ax.set_ylim(low,high)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4 if initial_selected else 5,fontsize=8)
        fig.suptitle(('All-to-All' if family=='alltoall' else 'All-Reduce')+': every condition; * = unseen 80 ranks',fontweight='bold');save(fig,'absolute_'+family)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm,color in [('real','#087f86'),('shuffled','#9b9a62')]:
        scores=[s for s in summary['selection']['scores'] if s['arm']==arm]
        axes[0].plot([s['round'] for s in scores],[100*np.expm1(s['mean_loss']) for s in scores],marker='o',c=color,label=arm)
        axes[1].plot([s['round'] for s in scores],[s['max_regression_pct'] for s in scores],marker='o',c=color,label=arm)
        chosen=summary['selection']['selected'][arm]
        axes[0].scatter(chosen['round'],100*np.expm1(chosen['mean_loss']),s=150,facecolors='none',edgecolors=color,lw=2)
    axes[0].axhline(0,c='#7d878d',lw=1);axes[0].set_ylabel('Validation mean tail change vs fixed (%)')
    axes[1].axhline(.5,c='#c36943',ls='--',label='0.5% eligibility limit');axes[1].set_ylabel('Worst family / topology / metric change (%)')
    for ax in axes:ax.set_xlabel('Offline training rounds');ax.set_xticks([0,4,8,12,16]);ax.grid(alpha=.15);ax.legend()
    fig.suptitle('Validation before test generation; large hollow circles mark selected checkpoints',fontweight='bold');save(fig,'validation_selection')
    scores=summary['selection']['scores']
    limit=max(1,max(abs(100*np.expm1(c['log_ratio'])) for s in scores for c in s['cells']))
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    contexts=[(f,t) for f in FAMILIES for t in TOPOS]
    short=['Uniform','Storage','Fast shift','Bursty','All-to-All','All-Reduce']
    fig,axes=plt.subplots(2,2,figsize=(16,7.5),layout='constrained')
    for i,arm in enumerate(['real','shuffled']):
        selected=[s for s in scores if s['arm']==arm]
        for j,m in enumerate(METRICS):
            ax=axes[i,j]
            a=np.array([[100*np.expm1(next(c['log_ratio'] for c in s['cells'] if (c['family'],c['topology'],c['metric'])==(f,t,m))) for f,t in contexts] for s in selected])
            im=ax.imshow(a,cmap='RdBu_r',norm=norm,aspect='auto')
            for row in range(len(selected)):
                for col in range(len(contexts)):
                    value=a[row,col]
                    ax.text(col,row,f'{value:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(value)>.7*limit else '#1d2930')
                    if value>.5+1e-10:ax.add_patch(Rectangle((col-.49,row-.49),.98,.98,fill=False,edgecolor='#303a40',lw=1))
            ax.set_yticks(range(len(selected)),[s['round'] for s in selected]);ax.set_ylabel('Offline rounds')
            ax.set_xticks(range(12),[short[FAMILIES.index(f)]+'\n'+('sym' if t=='symmetric' else 'asym') for f,t in contexts],rotation=45,ha='right',fontsize=8)
            ax.set_title(('Real rewards' if arm=='real' else 'Shuffled rewards')+' / '+m[:3].upper())
    fig.colorbar(im,ax=axes,label='Validation change vs fixed (%)',shrink=.75)
    fig.suptitle('Every validation constraint: outlined cells exceed 0.5% regression; negative means faster',fontweight='bold')
    save(fig,'validation_cells')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();analyze(args.root)


if __name__=='__main__':main()
