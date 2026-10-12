#!/usr/bin/env python3
"""Paired statistics and vector figures for the frozen collective campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ARMS=['ecmp','wecmp','drill','conga','letflow','conweave','fixed','rl']
LABEL=dict(zip(ARMS,['ECMP','WECMP','DRILL','CONGA','LetFlow','ConWeave*','Fixed score','CABLE']))
COLOR=dict(zip(ARMS,['#888888','#AA4499','#CCAA33','#228833','#44BBCC','#EE6677','#BB9911','#225588']))
MARK=dict(zip(ARMS,['x','v','s','D','P','^','h','o']))
METRICS=['makespan_us','p95_us','p99_us']
FAMILIES=['alltoall','allreduce'];TOPOS=['symmetric','asymmetric'];RANKS=[32,64]
SIZES={'alltoall':[1024,8192,65536],'allreduce':[131072,1048576,8388608]}
SEEDS=list(range(89001,89009));FITS=[9101,9102,9103]


def analyze(rows):
    expected=2*2*3*8*2*10
    assert len(rows)==expected and len({r['name'] for r in rows})==expected
    assert all(r['native_ok'] and r['analysis_ok'] and r['all_complete'] for r in rows), 'Report failed/incomplete runs before computing uncensored tails'
    lookup={(r['family'],r['ranks'],r['size'],r['seed'],r['topology_name'],r['arm'],r['fit']):r for r in rows}
    assert len(lookup)==expected
    rng=np.random.RandomState(314159);bs=rng.randint(0,8,(10000,8));bf=rng.randint(0,3,(10000,3))
    def stat(family,arm,metric,topos=TOPOS,ranks=RANKS,sizes=None,ratio=True,reference=None):
        arr=np.zeros((8,3))
        for i,seed in enumerate(SEEDS):
            for j,fit in enumerate(FITS):
                vals=[]
                for t in topos:
                    for n in ranks:
                        for b in sizes or SIZES[family]:
                            baseline=lookup[family,n,b,seed,t,arm,fit if arm=='rl' else 0][metric]
                            if ratio:
                                v=lookup[family,n,b,seed,t,'rl',fit][metric]/baseline
                            else:
                                ref=lookup[family,n,b,seed,t,reference,fit if reference=='rl' else 0][metric] if reference else 1.
                                v=baseline/ref
                            assert v>0;vals.append(np.log(v))
                arr[i,j]=np.mean(vals)
        boot=arr[bs[:,:,None],bf[:,None,:]].mean(axis=(1,2));point=float(np.exp(arr.mean()))
        ci=np.exp(np.percentile(boot,[2.5,97.5])).tolist()
        return dict(value=point,ci95=ci,delta_pct=100*(point-1) if ratio else None,
                    delta_ci95=[100*(x-1) for x in ci] if ratio else None)
    summary=dict(runs=len(rows),all_complete=True,flow_instances=sum(r['offered'] for r in rows),
                 dependency_edges_checked=sum(r['dependency_edges_checked'] for r in rows),
                 distinct_plans=len({r['plan_sha256'] for r in rows}),families={},slices={})
    for f in FAMILIES:
        results={arm:{m:stat(f,arm,m) for m in METRICS} for arm in ARMS[:-1]}
        admit=all(results[a]['makespan_us']['value']<=.95 and results[a]['makespan_us']['ci95'][1]<1
                  and results[a]['p95_us']['value']<=1 and results[a]['p99_us']['value']<=1
                  for a in ['conga','conweave'])
        summary['families'][f]=dict(comparisons=results,admit_to_paper=admit)
        for t in TOPOS:
            for n in RANKS:
                key=f'{f}/{t}/{n}'
                summary['slices'][key]={arm:{m:stat(f,arm,m,[t],[n]) for m in METRICS} for arm in ARMS[:-1]}
    return summary,stat


def figures(out,summary,stat):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
                         'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,
                         'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',pad_inches=.05,dpi=200)
        plt.close(fig)
    # Common linear ranges depend on every method and its complete 95% interval.
    def ratio_limits(axes,bounds):
        lo=max(0.,np.floor((min(bounds)-.025)/.1)*.1)
        hi=np.ceil((max(bounds)+.025)/.1)*.1
        for ax in np.asarray(axes).flat:
            ax.set_ylim(lo,hi);ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1 if hi-lo<=.9 else (.2 if hi-lo<=1.8 else .5)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
            ax.axhline(1.,color='#707780',lw=.8,ls=':',zorder=1)
        return [float(lo),float(hi)]
    line_points=[];axis_limits={};overview_bounds=[1.]
    # Eight panels retain both worker counts, workload families and topology types.
    fig,axes=plt.subplots(2,4,figsize=(10,4.9))
    for col,(family,n) in enumerate((f,n) for f in FAMILIES for n in RANKS):
        for row,t in enumerate(TOPOS):
            ax=axes[row,col]
            for arm in ARMS:
                vals=[stat(family,arm,'makespan_us',[t],[n],[b],False,reference='ecmp') for b in SIZES[family]]
                y=np.array([v['value'] for v in vals]);ci=np.array([v['ci95'] for v in vals])
                overview_bounds.extend(ci.ravel())
                ax.plot(range(3),y,color=COLOR[arm],marker=MARK[arm],ms=4,
                        markerfacecolor='none' if arm=='fixed' else COLOR[arm],
                        linestyle='--' if arm=='fixed' else '-',lw=1.8 if arm=='rl' else 1.1,label=LABEL[arm],
                        zorder=5 if arm in ['rl','fixed'] else 2)
                ax.fill_between(range(3),ci[:,0],ci[:,1],color=COLOR[arm],alpha=.10,lw=0)
            ax.grid(axis='y',color='#e7ebef',lw=.5);ax.set_axisbelow(True)
            ax.set_xticks(range(3),['1','8','64'] if family=='alltoall' else ['128','1024','8192'])
            ax.set_title(f"{'All-to-All' if family=='alltoall' else 'Ring All-Reduce'} / {n} ranks\n{t.capitalize()}")
            if col==0:ax.set_ylabel('Job time / ECMP job time\n(lower is better)')
            if row==1:ax.set_xlabel('Message size per peer (KiB)' if family=='alltoall' else 'Input size per rank (KiB)')
    axis_limits['collective_completion']=ratio_limits(axes,overview_bounds)
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.025))
    fig.tight_layout(rect=[0,0,1,.955],w_pad=1.3,h_pad=1.4);save(fig,'collective_completion')
    # The complete paired outcome map: all seven comparators, three metrics, eight groups.
    matrix=[];labels=[]
    for f in FAMILIES:
        for n in RANKS:
            for t in TOPOS:
                matrix.append([stat(f,a,m,[t],[n])['delta_pct'] for a in ARMS[:-1] for m in METRICS])
                labels.append(f"{'A2A' if f=='alltoall' else 'All-Reduce'} / {n} / {'Sym.' if t=='symmetric' else 'Asym.'}")
    data=np.array(matrix);lim=max(10,np.ceil(np.abs(data).max()/10)*10)
    fig,ax=plt.subplots(figsize=(10,3.75));im=ax.imshow(data,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    ax.set_yticks(range(8),labels);ax.set_xticks(range(21),['Job','P95','P99']*7);ax.tick_params(length=0)
    for j,a in enumerate(ARMS[:-1]):
        ax.text(3*j+1,-.85,LABEL[a],ha='center',va='bottom',fontsize=8)
        if j:ax.axvline(3*j-.5,c='white',lw=2)
    for i in range(8):
        for j in range(21):
            v=data[i,j];label='0' if abs(v)<.5 else f'{v:+.0f}'
            ax.text(j,i,label,ha='center',va='center',fontsize=7,color='white' if abs(v)>.58*lim else '#20242a')
    ax.axhline(3.5,c='white',lw=2);fig.colorbar(im,ax=ax,pad=.02,fraction=.024,label='CABLE change (%); lower is better')
    fig.tight_layout();save(fig,'collective_outcomes')
    # Direct confidence intervals for the paper admission and same-information control.
    fig,axes=plt.subplots(1,3,figsize=(7,2.65),sharey=True)
    for mi,m in enumerate(METRICS):
        ax=axes[mi];ax.axvline(0,color='#666',lw=.8);ax.grid(axis='x',color='#e7ebef',lw=.6);ax.set_axisbelow(True)
        for fi,f in enumerate(FAMILIES):
            vals=[stat(f,a,m) for a in ['conga','conweave','fixed']]
            x=np.array([v['delta_pct'] for v in vals]);ci=np.array([v['delta_ci95'] for v in vals])
            ax.errorbar(x,np.arange(3)+(fi-.5)*.22,xerr=[x-ci[:,0],ci[:,1]-x],fmt=['o','s'][fi],
                        color=['#225588','#CC6677'][fi],capsize=2,ms=4,lw=1,label=['All-to-All','Ring All-Reduce'][fi])
        ax.set_title(['Job completion','Transfer P95','Transfer P99'][mi]);ax.set_xlabel('CABLE change (%)')
        ax.set_yticks(range(3),['CONGA','ConWeave*','Fixed score']);ax.set_ylim(2.4,-.4)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,frameon=False,
               loc='upper center',bbox_to_anchor=(.55,1.025))
    fig.tight_layout(rect=[0,0,1,.88]);save(fig,'collective_intervals')
    # Final-size paper panels: all ranks, topologies, message sizes and algorithms.
    for family in FAMILIES:
        fig,axes=plt.subplots(2,2,figsize=(7,4.1),sharex=True,sharey=True)
        family_bounds=[1.]
        for i,n in enumerate(RANKS):
            for j,t in enumerate(TOPOS):
                ax=axes[i,j]
                for arm in ARMS:
                    vals=[stat(family,arm,'makespan_us',[t],[n],[b],False,reference='ecmp') for b in SIZES[family]]
                    y=np.array([v['value'] for v in vals]);ci=np.array([v['ci95'] for v in vals])
                    family_bounds.extend(ci.ravel())
                    line_points.extend(dict(figure='collective_'+family,family=family,ranks=n,topology=t,size=b,arm=arm,reference='ecmp',value=v['value'],ci95=v['ci95']) for b,v in zip(SIZES[family],vals))
                    ax.plot(range(3),y,color=COLOR[arm],marker=MARK[arm],ms=4.5,
                            markerfacecolor='none' if arm=='fixed' else COLOR[arm],
                            linestyle='--' if arm=='fixed' else '-',lw=1.8 if arm=='rl' else 1.1,
                            label=LABEL[arm],zorder=5 if arm in ['rl','fixed'] else 2)
                    ax.fill_between(range(3),ci[:,0],ci[:,1],color=COLOR[arm],alpha=.10,lw=0)
                ax.grid(axis='y',color='#e7ebef',lw=.5);ax.set_axisbelow(True)
                ax.set_title(f'{n} ranks / {t.capitalize()}')
                ax.set_xticks(range(3),['1','8','64'] if family=='alltoall' else ['128','1024','8192'])
                if j==0:ax.set_ylabel('Job time / ECMP job time\n(lower is better)')
                if i==1:ax.set_xlabel('Message size per peer (KiB)' if family=='alltoall' else 'Input size per rank (KiB)')
        axis_limits['collective_'+family]=ratio_limits(axes,family_bounds)
        fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=4,frameon=False,
                   loc='upper center',bbox_to_anchor=(.5,1.03),columnspacing=1.5)
        fig.tight_layout(rect=[0,0,1,.89],w_pad=1.4,h_pad=1.2);save(fig,'collective_'+family)

    (out/'collective-axis-points.json').write_text(json.dumps(dict(axis_limits=axis_limits,points=line_points),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--experiment-dir',type=Path);sources.add_argument('--compact-data',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.experiment_dir/'main-results.json' if a.experiment_dir else a.compact_data).read_text())
    summary,stat=analyze(rows);a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'collective-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Raw audit occurs during collection; verify its evidence files remain unchanged.
    audit=[]
    for r in rows if a.experiment_dir else []:
        folder=Path(r['folder'])
        for name,key in [('fct_output_file.txt','fct_sha256'),('collective-events.txt','event_sha256')]:
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==r[key]
        audit.append({k:r[k] for k in ['name','offered','completed','dependency_edges_checked','fct_sha256','event_sha256']})
    (a.output_dir/'collective-audit.json').write_text(json.dumps(audit,indent=1)+'\n')
    compact=[{k:v for k,v in r.items() if k not in ['folder','actor_audit']} for r in rows]
    (a.output_dir/'collective-runs.json').write_text(json.dumps(compact,indent=1)+'\n')
    figures(a.output_dir,summary,stat)
    print(json.dumps(summary['families'],indent=2))


if __name__=='__main__':main()
