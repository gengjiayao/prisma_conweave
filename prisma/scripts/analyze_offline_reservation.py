#!/usr/bin/env python3
"""Paired family-balanced held-out comparisons for frozen reservation policies."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from offline_reservation_evaluation import FAMILIES,FEATURES,env,loss

LABELS={'ecmp':'ECMP','wecmp':'Weighted ECMP','conga':'CONGA','conweave':'ConWeave*','fixed':'Original fixed rule','cable':'Original CABLE','deleted':'Three-term deletion','mechanism':'Reservations + spreading, fixed','reserve_only':'Reservations only, fixed','spread_only':'Spreading only, fixed','learned':'Offline learned','shuffled':'Reward-shuffled training'}
PAIRED={'mechanism','reserve_only','spread_only','shuffled'}
COLORS={'learned':'#147d82','cable':'#334b78','mechanism':'#d48938','shuffled':'#9e5674','conga':'#8374a3','conweave':'#8b6654','ecmp':'#9b9fa3','wecmp':'#b7ad75','fixed':'#567054','deleted':'#839456','reserve_only':'#689daf','spread_only':'#bb837b'}

def array(rows,arm,metric,scope,family=None,topology=None):
    selected=[r for r in rows if r['arm']==arm and (scope=='all' or r['unseen']==(scope=='unseen')) and (family is None or r['family']==family) and (topology is None or r['topology_name']==topology) and r[metric] is not None]
    if not selected:return None
    seeds=sorted({r['seed'] for r in selected});fits=sorted({r['fit'] for r in selected});fams=sorted({r['family'] for r in selected})
    out=np.zeros((len(seeds),len(fits)))
    for i,seed in enumerate(seeds):
        for j,fit in enumerate(fits):
            cells=[]
            for fam in fams:
                v=[np.log(r[metric]) for r in selected if (r['seed'],r['fit'],r['family'])==(seed,fit,fam)];assert v
                cells.append(np.mean(v))
            out[i,j]=np.mean(cells)
    return out

def compare(rows,base,metric,scope,family=None,topology=None,actor='learned'):
    a=array(rows,actor,metric,scope,family,topology);b=array(rows,base,metric,scope,family,topology)
    if a is None or b is None:return None
    assert a.shape[0]==b.shape[0]==4
    paired=(base in PAIRED and actor=='learned') or (actor=='deleted' and base=='cable')
    rng=np.random.default_rng(9842026);si=rng.integers(0,4,(10000,4));fa=rng.integers(0,a.shape[1],(10000,a.shape[1]));fb=fa if paired else rng.integers(0,b.shape[1],(10000,b.shape[1]))
    draws=a[si[:,:,None],fa[:,None,:]].mean(axis=(1,2))-b[si[:,:,None],fb[:,None,:]].mean(axis=(1,2))
    ci=100*np.expm1(np.percentile(draws,[2.5,97.5]));point=100*np.expm1(a.mean()-b.mean())
    return dict(actor=actor,baseline=base,metric=metric,scope=scope,family=family,topology=topology,change_pct=float(point),ci95_pct=ci.tolist(),paired_fit=bool(paired))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    rows=json.loads((root/'test-results.json').read_text());assert len(rows)==2704 and all(r['all_complete'] and r['analysis_ok'] for r in rows)
    protocol=json.loads((root/'protocol.json').read_text());selected=json.loads((root/'selection.json').read_text())
    comparisons=[]
    for scope in ['familiar','unseen','all']:
        for metric in ['p95_us','p99_us','makespan_us']:
            for base in LABELS:
                if base=='learned':continue
                for family in [None]+FAMILIES:
                    result=compare(rows,base,metric,scope,family)
                    if result:comparisons.append(result)
    for base in ['cable','mechanism','shuffled','conga','conweave']:
        for metric in ['p95_us','p99_us']:
            for topo in ['symmetric','asymmetric']:
                for family in [None]+FAMILIES:
                    result=compare(rows,base,metric,'all',family,topo)
                    if result:comparisons.append(result)
    # Extra fixed-mechanism comparisons separate mechanism gains from training.
    mechanisms=[]
    for actor in ['mechanism','reserve_only','spread_only','deleted']:
        for base in ['cable','fixed']:
            for fam in [None]+FAMILIES:
                for m in ['p95_us','p99_us']:
                    q=compare(rows,base,m,'familiar',fam,actor=actor)
                    if q:mechanisms.append(q)
    out=root.parent/'analysis';out.mkdir(exist_ok=True)
    def get(base,m,scope='familiar',family=None,topology=None):
        return next(x for x in comparisons if (x['baseline'],x['metric'],x['scope'],x['family'],x['topology'])==(base,m,scope,family,topology))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.8),layout='constrained');bases=['cable','mechanism','shuffled','reserve_only','spread_only','deleted','fixed','conga','conweave','wecmp','ecmp']
    for ax,scope in zip(axes,['familiar','unseen']):
        for m,col,offset,l in [('p95_us','#147d82',-.14,'P95'),('p99_us','#bd6639',.14,'P99')]:
            for i,b in enumerate(bases):
                q=get(b,m,scope);v=q['change_pct'];lo,hi=q['ci95_pct'];ax.errorbar(v,i+offset,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',ms=4,color=col,capsize=2,label=l if i==0 else None)
        ax.set_yticks(range(len(bases)),[LABELS[b] for b in bases]);ax.invert_yaxis();ax.axvline(0,color='#7b8790',lw=1);ax.grid(axis='x',alpha=.17);ax.set_xlabel('Offline learned / comparator − 1 (%)');ax.set_title('New seeds, familiar sizes / ranks' if scope=='familiar' else 'New seeds + unseen sizes / ranks\nCollectives only');ax.legend(loc='best')
    fig.suptitle('Held-out P95/P99: paired traffic seeds and training-fit bootstrap',fontweight='bold');save(fig,'paired_tail_comparisons')
    bases=['cable','mechanism','shuffled','conga','conweave'];fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    allvals=[get(b,m,'all',f)['change_pct'] for b in bases for m in ['p95_us','p99_us'] for f in FAMILIES];lim=max(10,np.max(np.abs(allvals)))
    for ax,m in zip(axes,['p95_us','p99_us']):
        vals=np.array([[get(b,m,'all',f)['change_pct'] for f in FAMILIES] for b in bases]);im=ax.imshow(vals,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
        for i,line in enumerate(vals):
            for j,v in enumerate(line):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>.7*lim else '#172435')
        ax.set_xticks(range(6),['Uniform','Storage','Fast shift','Bursty','All-to-All','All-Reduce'],rotation=35,ha='right');ax.set_yticks(range(len(bases)),[LABELS[b] for b in bases]);ax.set_title(m[:3].upper()+' change (%)')
    fig.colorbar(im,ax=axes,label='Positive = learned policy is slower',shrink=.8);fig.suptitle('Every traffic family, including losses; all held-out conditions',fontweight='bold');save(fig,'family_effects')
    # Absolute collective curves: every rank/size condition and all algorithms.
    for family in ['alltoall','allreduce']:
        conditions=sorted({(r['ranks'],r['size'],r['unseen']) for r in rows if r['family']==family});fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
        for i,topo in enumerate(['symmetric','asymmetric']):
            for k,metric in enumerate(['p95_us','p99_us','makespan_us']):
                ax=axes[i,k]
                for arm,label in LABELS.items():
                    ys=[]
                    for ranks,size,unseen in conditions:
                        v=[r[metric]/1000 for r in rows if (r['family'],r['topology_name'],r['arm'],r['ranks'],r['size'])==(family,topo,arm,ranks,size)];ys.append(float(np.exp(np.mean(np.log(v)))))
                    ax.plot(range(len(ys)),ys,label=label,color=COLORS[arm],lw=2.4 if arm=='learned' else 1.1,alpha=1 if arm in ['learned','cable','mechanism'] else .65,marker='o',ms=4 if arm=='learned' else 2)
                ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'));ax.set_xticks(range(len(conditions)),[f'{p} / {b//1024}'+('*' if u else '') for p,b,u in conditions],rotation=35,ha='right');ax.set_ylabel(topo.capitalize()+' time (ms)');ax.set_xlabel('Ranks / KiB per peer' if family=='alltoall' else 'Ranks / KiB per tensor');ax.grid(alpha=.15);ax.set_title({'p95_us':'P95 flow completion','p99_us':'P99 flow completion','makespan_us':'Collective completion'}[metric])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
        fig.suptitle(f'{family}: all algorithms, sizes and topologies (* = unseen training condition)',fontweight='bold');save(fig,'absolute_'+family)
    val=json.loads((root/'validation-results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for arm,col in [('real','#147d82'),('shuffled','#9e5674')]:
        means=[];lines=[]
        for fit in range(3):
            line=[loss([r for r in val if r['arm']=='val_'+arm and r['fit']==fit and r['name'].endswith(f'_r{rnd}')]) for rnd in [0,4,8]];lines.append(line);axes[0].plot([0,4,8],100*np.expm1(np.array(line)-line[0]),color=col,alpha=.25,lw=1)
        average=np.mean(lines,axis=0);axes[0].plot([0,4,8],100*np.expm1(average-average[0]),color=col,marker='o',lw=2,label=arm)
        rnd=selected['selected'][arm]['round'];axes[0].scatter(rnd,100*np.expm1(selected['selected'][arm]['loss']-average[0]),s=100,facecolors='none',edgecolors=col,zorder=5)
    axes[0].set_xlabel('Offline training rounds');axes[0].set_ylabel('Validation objective change vs initializer (%)');axes[0].set_xticks([0,4,8]);axes[0].legend();axes[0].grid(alpha=.15)
    for i,b in enumerate(['cable','mechanism','shuffled']):
        for k,m in enumerate(['p95_us','p99_us']):
            q=get(b,m);v=q['change_pct'];lo,hi=q['ci95_pct'];axes[1].errorbar(v,i+(k-.5)*.18,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=['#147d82','#bd6639'][k],label=m[:3].upper() if i==0 else None)
    axes[1].set_yticks(range(3),[LABELS[b] for b in ['cable','mechanism','shuffled']]);axes[1].invert_yaxis();axes[1].axvline(0,c='#84909a',lw=1);axes[1].set_xlabel('Held-out change (%)');axes[1].legend();axes[1].grid(axis='x',alpha=.15);fig.suptitle('Training attribution: validation selection and untouched test results',fontweight='bold');save(fig,'learning_attribution')
    summary=dict(runs=len(rows),flow_instances=sum(r['offered'] for r in rows),all_complete=True,selected=selected,
        protocol_amendment=json.loads((root/'protocol-amendment.json').read_text()) if (root/'protocol-amendment.json').exists() else None,
        protocol_sha256=env.sha(root/'protocol.json'),comparisons=comparisons,mechanism_comparisons=mechanisms,
        figures={f.name:env.sha(f) for f in out.iterdir() if f.suffix in ['.png','.pdf']})
    env.dump(out/'summary.json',summary)
    print('SUMMARY',len(rows),summary['flow_instances'])
    for b in ['cable','mechanism','shuffled','conga','conweave']:
        print(b,{m:get(b,m) for m in ['p95_us','p99_us']})
if __name__=='__main__':main()
