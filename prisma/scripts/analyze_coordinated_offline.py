#!/usr/bin/env python3
"""Full-condition, seed-block comparisons of causal frozen routing policies."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter,LogLocator
from coordinated_offline_evaluation import env,FAMILIES,FLAGS
LABELS={'ecmp':'ECMP','wecmp':'Weighted ECMP','conga':'CONGA','conweave':'ConWeave*','cable':'Original CABLE','previous_learned':'Previous learned','reserve_only':'Previous reservations','accounting':'Causal count control','rotor':'Offset allocation','work':'Progress reservations','combined':'Both, fixed','conditional':'Conditional, fixed','learned':'Conditional, learned','shuffled':'Reward-shuffled'}
COLORS=dict(zip(LABELS,['#c0c2c6','#ad9b6b','#927fa6','#bd6237','#526582','#6886a3','#477566','#a5b199','#918542','#94afb3','#5b83b1','#ab7697','#097f83','#9d4c67']))

def matrix(rows,arm,metric,scope,family,topology):
    selected=[r for r in rows if r['arm']==arm and r[metric] is not None and (scope=='all' or r['unseen']==(scope=='unseen')) and (family is None or r['family']==family) and (topology is None or r['topology_name']==topology)]
    if not selected:return None
    seeds=sorted({r['seed'] for r in selected});fits=sorted({r['fit'] for r in selected});families=sorted({r['family'] for r in selected})
    return np.array([[np.mean([np.mean([np.log(r[metric]) for r in selected if (r['seed'],r['fit'],r['family'])==(s,f,fam)]) for fam in families]) for f in fits] for s in seeds])

def compare(rows,actor,baseline,metric,scope='familiar',family=None,topology=None):
    a=matrix(rows,actor,metric,scope,family,topology);b=matrix(rows,baseline,metric,scope,family,topology)
    if a is None or b is None:return None
    assert a.shape[0]==b.shape[0]
    paired=(actor in list(FLAGS)+['learned','shuffled'] and baseline in list(FLAGS)+['learned','shuffled'] and a.shape[1]==b.shape[1])
    rng=np.random.default_rng(19360917);si=rng.integers(0,len(a),(10000,len(a)));fa=rng.integers(0,a.shape[1],(10000,a.shape[1]));fb=fa if paired else rng.integers(0,b.shape[1],(10000,b.shape[1]))
    draws=a[si[:,:,None],fa[:,None,:]].mean(axis=(1,2))-b[si[:,:,None],fb[:,None,:]].mean(axis=(1,2))
    return dict(actor=actor,baseline=baseline,metric=metric,scope=scope,family=family,topology=topology,paired_fit=paired,change_pct=float(100*np.expm1(a.mean()-b.mean())),ci95_pct=(100*np.expm1(np.percentile(draws,[2.5,97.5]))).tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stage',choices=['screen','test'],default='test');args=ap.parse_args();root=args.root
    rows=json.loads((root/(args.stage+'-results.json')).read_text());assert all(r['all_complete'] and r['analysis_ok'] for r in rows)
    arms=[a for a in LABELS if any(r['arm']==a for r in rows)];comparisons=[]
    for actor in arms:
        for baseline in ['conweave','conga','ecmp','reserve_only','cable','conditional','combined','shuffled','accounting']:
            if baseline not in arms or baseline==actor:continue
            # All algorithms versus ConWeave; detailed attribution for new policies.
            if baseline!='conweave' and actor not in list(FLAGS)+['learned']:continue
            for scope in ['familiar','unseen','all']:
                for family in [None]+FAMILIES:
                    for m in ['p95_us','p99_us','makespan_us']:
                        q=compare(rows,actor,baseline,m,scope,family)
                        if q:comparisons.append(q)
    for actor in list(FLAGS)+['learned','reserve_only','previous_learned']:
        if actor not in arms:continue
        for topo in ['symmetric','asymmetric']:
            for scope in ['familiar','unseen','all']:
                for family in [None,'alltoall','allreduce']:
                    for m in ['p95_us','p99_us']:
                        q=compare(rows,actor,'conweave',m,scope,family,topo)
                        if q:comparisons.append(q)
    out=root.parent/('analysis' if args.stage=='test' else 'screen-analysis');out.mkdir(exist_ok=True)
    summary=dict(stage=args.stage,runs=len(rows),flow_instances=sum(r['offered'] for r in rows),all_complete=True,comparisons=comparisons,protocol_sha256=env.sha(root/'protocol.json'))
    def get(actor,base,m,scope='familiar',family=None,topology=None):return next(q for q in comparisons if (q['actor'],q['baseline'],q['metric'],q['scope'],q['family'],q['topology'])==(actor,base,m,scope,family,topology))
    if args.stage=='test':
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
        def save(fig,name):
            for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
            plt.close(fig)
        show=['ecmp','wecmp','conga','cable','previous_learned','reserve_only','accounting','rotor','work','combined','conditional','learned','shuffled']
        fig,axes=plt.subplots(1,3,figsize=(16,6.5),layout='constrained')
        for ax,fam,scope,title in zip(axes,[None,'alltoall','alltoall'],['familiar','familiar','unseen'],['Six families, equal weight','All-to-All: developed conditions','All-to-All: unseen 80 ranks']):
            for i,a in enumerate(show):
                for m,c,offset,l in [('p95_us','#127f84',-.14,'P95'),('p99_us','#bd633c',.14,'P99')]:
                    q=get(a,'conweave',m,scope,fam);v=q['change_pct'];lo,hi=q['ci95_pct'];ax.errorbar(v,i+offset,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',ms=3.7,color=c,capsize=2,label=l if i==0 else None)
            ax.axvline(0,c='#899197',lw=1);ax.set_yticks(range(len(show)),[LABELS[a] for a in show]);ax.invert_yaxis();ax.grid(axis='x',alpha=.17);ax.set_xlabel('Change vs ConWeave (%)');ax.set_title(title);ax.legend()
        fig.suptitle('Independent test seeds: negative means faster; all policies retained',fontweight='bold');save(fig,'conweave_comparisons')
        actors=['reserve_only','rotor','work','combined','conditional','learned','shuffled'];fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
        vals=[get(a,'conweave',m,'familiar',f)['change_pct'] for a in actors for m in ['p95_us','p99_us'] for f in FAMILIES];lim=max(10,max(abs(v) for v in vals))
        for ax,m in zip(axes,['p95_us','p99_us']):
            values=np.array([[get(a,'conweave',m,'familiar',f)['change_pct'] for f in FAMILIES] for a in actors]);im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
            for i,row in enumerate(values):
                for j,v in enumerate(row):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>.7*lim else '#1b2635')
            ax.set_xticks(range(6),['Uniform','Storage','Fast shift','Bursty','All-to-All','All-Reduce'],rotation=35,ha='right');ax.set_yticks(range(len(actors)),[LABELS[a] for a in actors]);ax.set_title(m[:3].upper()+' change vs ConWeave (%)')
        fig.colorbar(im,ax=axes,label='Positive = slower',shrink=.8);fig.suptitle('Every traffic family, developed conditions with new seeds',fontweight='bold');save(fig,'family_effects')
        for family in ['alltoall','allreduce']:
            conditions=sorted({(r['ranks'],r['size'],r['unseen']) for r in rows if r['family']==family});fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
            for i,topo in enumerate(['symmetric','asymmetric']):
                for k,m in enumerate(['p95_us','p99_us','makespan_us']):
                    ax=axes[i,k]
                    for arm in arms:
                        y=[float(np.exp(np.mean([np.log(r[m]/1000) for r in rows if (r['family'],r['topology_name'],r['arm'],r['ranks'],r['size'])==(family,topo,arm,n,b)]))) for n,b,u in conditions]
                        ax.plot(range(len(y)),y,label=LABELS[arm],color=COLORS[arm],lw=2.2 if arm in ['learned','conweave'] else 1,alpha=1 if arm in ['learned','conweave','conditional'] else .6,marker='o',ms=3 if arm in ['learned','conweave'] else 2)
                    ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'));ax.yaxis.set_minor_locator(LogLocator(base=10,subs=[2,5]));ax.yaxis.set_minor_formatter(FuncFormatter(lambda v,p:f'{v:g}'));ax.set_xticks(range(len(conditions)),[f'{n} / {b//1024}'+('*' if u else '') for n,b,u in conditions],rotation=40,ha='right');ax.set_ylabel(topo.capitalize()+' time (ms, log scale)');ax.grid(alpha=.15);ax.set_xlabel('Ranks / KiB per '+('peer' if family=='alltoall' else 'tensor'));ax.set_title({'p95_us':'P95 flow completion','p99_us':'P99 flow completion','makespan_us':'Collective completion'}[m])
            for column in range(3):
                limits=[ax.get_ylim() for ax in axes[:,column]];shared=(min(x[0] for x in limits),max(x[1] for x in limits))
                for ax in axes[:,column]:ax.set_ylim(shared)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8);fig.suptitle(('All-to-All' if family=='alltoall' else 'All-Reduce')+': all algorithms and conditions (* = held-out conditions)',fontweight='bold');save(fig,'absolute_'+family)
        sel=json.loads((root/'selection.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        for arm,c in [('real','#097f83'),('shuffled','#9d4c67')]:
            v=sorted([q for q in sel['scores'] if q['arm']==arm],key=lambda q:q['round']);axes[0].plot([q['round'] for q in v],[q['loss'] for q in v],marker='o',c=c,label=arm);q=sel['selected'][arm];axes[0].scatter(q['round'],q['loss'],s=130,facecolors='none',edgecolors=c)
        axes[0].set_xlabel('Offline training rounds');axes[0].set_ylabel('Validation objective (lower is better)');axes[0].legend();axes[0].grid(alpha=.15)
        bases=['conweave','reserve_only','conditional','combined','shuffled']
        for i,b in enumerate(bases):
            for m,c,offset,l in [('p95_us','#127f84',-.13,'P95'),('p99_us','#bd633c',.13,'P99')]:
                q=get('learned',b,m);v=q['change_pct'];lo,hi=q['ci95_pct'];axes[1].errorbar(v,i+offset,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',c=c,label=l if i==0 else None)
        axes[1].set_yticks(range(len(bases)),[LABELS[b] for b in bases]);axes[1].invert_yaxis();axes[1].axvline(0,c='#899197',lw=1);axes[1].set_xlabel('Learned / comparator − 1 (%)');axes[1].legend();axes[1].grid(axis='x',alpha=.15);fig.suptitle('Validation selection and independent-test attribution',fontweight='bold');save(fig,'learning_attribution')
        summary['selection']=sel;summary['figures']={f.name:env.sha(f) for f in out.iterdir() if f.suffix in ['.png','.pdf']}
    env.dump(out/'summary.json',summary)
    for a in list(FLAGS)+(['learned','shuffled'] if args.stage=='test' else []):
        print(a,{fam:{m:get(a,'conweave',m,'familiar',fam) for m in ['p95_us','p99_us']} for fam in [None,'alltoall','allreduce']})
if __name__=='__main__':main()
