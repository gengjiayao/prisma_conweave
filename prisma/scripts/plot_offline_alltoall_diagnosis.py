#!/usr/bin/env python3
"""Full-result diagnostic figures, with explicit paired baselines and units."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from diagnose_offline_alltoall import ARCHIVE,env
from analyze_offline_alltoall import aggregate

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);root=a.parse_args().root
    rows=json.loads((root/'results.json').read_text())+json.loads((root/'remote-decomposition/results.json').read_text())
    assert len(rows)==900 and all(r['all_complete'] and r['analysis_ok'] for r in rows)
    agg=aggregate(rows);env.dump(root/'combined-aggregate.json',agg)
    old=[r for r in json.loads((ARCHIVE/'analysis/collective-runs.json').read_text()) if r['family']=='alltoall']
    oldidx={(r['ranks'],r['size'],r['seed'],r['topology_name'],r['arm'],r['fit']):r for r in old}
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained')
    cells=[(t,p) for t in ['symmetric','asymmetric'] for p in [32,64]]
    for ax,base in zip(axes,['conga','fixed']):
        vals=[]
        for t,p in cells:
            line=[]
            for size in [1024,8192,65536]:
                z=[np.log(r['p99_us']/oldidx[p,size,r['seed'],t,base,0]['p99_us']) for r in old if r['arm']=='rl' and (r['ranks'],r['size'],r['topology_name'])==(p,size,t)]
                line.append(100*np.expm1(np.mean(z)))
            vals.append(line)
        im=ax.imshow(vals,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-65,vcenter=0,vmax=70),aspect='auto')
        for i,line in enumerate(vals):
            for k,v in enumerate(line):ax.text(k,i,f'{v:+.1f}%',ha='center',va='center',color='white' if abs(v)>48 else '#182435')
        ax.set_xticks(range(3),['1 KiB','8 KiB','64 KiB']);ax.set_yticks(range(4),['Symmetric / 32','Symmetric / 64','Asymmetric / 32','Asymmetric / 64'])
        ax.set_xlabel('Bytes per peer');ax.set_title('CABLE vs '+('CONGA' if base=='conga' else 'same-information fixed rule'))
    fig.colorbar(im,ax=axes,label='P99 change (%) — positive is slower',shrink=.8)
    fig.suptitle('Original All-to-All results: all 8 traffic seeds and 3 frozen models',fontweight='bold');save(fig,'original_breakdown')
    names={'local_only':'Local queue only','no_queue_trend_destination':'Remove remote queue, trend + destination history',
        'no_queue_trend':'Remove remote queue + trend','no_queue':'Remove remote queue only','no_trend':'Remove remote trend only',
        'no_destination':'Remove destination history','no_port':'Remove port history','no_history':'Remove both history terms',
        'no_serialization':'Remove explicit serialization terms','feedback20':'Faster feedback: 20 us','reselect':'Frozen actor + safe reselection','no_remote':'Remove full remote delay + trend'}
    arms=list(names);fig,(ax,heat)=plt.subplots(1,2,figsize=(13,6.8),gridspec_kw={'width_ratios':[1.8,1]},layout='constrained')
    for m,color,offset,label in [('p95_us','#187f87',-.13,'P95'),('p99_us','#d36a31',.13,'P99')]:
        ys=np.arange(len(arms))+offset;vs=[next(r[m] for r in agg if r['arm']==arm and r['topology']=='all' and r['ranks']==0) for arm in arms]
        ax.scatter(vs,ys,c=color,s=36,label=label,zorder=3)
        for v,y in zip(vs,ys):ax.plot([0,v],[y,y],c=color,alpha=.4,lw=1)
    ax.set_yticks(range(len(arms)),[names[x] for x in arms]);ax.invert_yaxis();ax.axvline(0,c='#88909a',lw=1);ax.grid(axis='x',alpha=.16)
    ax.set_xlabel('Change vs the same original CABLE model (%)');ax.legend(loc='lower right');ax.set_title('Paired interventions: P95 and P99')
    matrix=[[next(r['p99_us'] for r in agg if r['arm']==arm and r['topology']==t and r['ranks']==p) for t,p in cells] for arm in arms]
    im=heat.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-60,vcenter=0,vmax=max(75,np.max(matrix))),aspect='auto')
    for i,line in enumerate(matrix):
        for k,v in enumerate(line):heat.text(k,i,f'{v:+.0f}',ha='center',va='center',fontsize=9,color='white' if v>80 or v<-45 else '#182435')
    heat.set_xticks(range(4),['Sym.\n32','Sym.\n64','Asym.\n32','Asym.\n64']);heat.set_yticks([]);heat.set_title('P99 change by topology and rank count (%)')
    fig.suptitle('Offline diagnosis: 2 traffic seeds, 3 frozen models; all message sizes retained',fontweight='bold');save(fig,'paired_interventions')
    traces=json.loads((root/'decision-analysis.json').read_text())+json.loads((root/'remote-decomposition/decision-analysis.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(10.8,8.1),layout='constrained')
    fig.set_constrained_layout_pads(hspace=.12,wspace=.04)
    for i,t in enumerate(['symmetric','asymmetric']):
        for k,arm in enumerate(['rl','local_only','no_queue_trend_destination']):
            x=next(x for x in traces if x['arm']==arm and x['fit']==9101 and x['topology']==t)
            loads=np.array(x['incoming_bytes']);rate=np.array(x['path_rate_gbps']);ideal=loads.sum()/8/rate.sum();ratio=loads/rate[None,:]/ideal
            ax=axes[i,k];im=ax.imshow(ratio,cmap='YlOrRd',vmin=0,vmax=2.5)
            ax.set_xticks([0,3,7],['1','4','8']);ax.set_yticks([0,3,7],['1','4','8']);ax.set_xlabel('Spine order' if i else '');ax.set_ylabel(t.capitalize()+'\nDestination leaf')
            ax.set_title(({'rl':'Original CABLE','local_only':'Local queue only','no_queue_trend_destination':'Remove three terms'}[arm])+f'\nPeak / ideal: {ratio.max():.2f}x',fontsize=10)
    fig.colorbar(im,ax=axes,label='Assigned bytes / capacity-proportional ideal',shrink=.8)
    fig.suptitle('Destination-side path concentration: 64 ranks, 64 KiB / peer\nFixed example: traffic seed 89001, model 9101',fontweight='bold');save(fig,'destination_path_load')
    archive_traces=json.loads((root/'archive-decision-analysis.json').read_text());assert len(archive_traces)==384
    points=[]
    for x in archive_traces:
        r=oldidx[x['ranks'],x['size'],x['seed'],x['topology'],x['arm'],x['fit']]
        bound=float((np.array(x['incoming_bytes'])/np.array(x['path_rate_gbps'])[None,:]).max()*8/1e6)
        points.append(dict(name=x['name'],arm=x['arm'],ranks=x['ranks'],topology=x['topology'],bound_ms=bound,cct_ms=r['makespan_us']/1000,p99_ms=r['p99_us']/1000,ratio=r['makespan_us']/1000/bound))
    env.dump(root/'path-service-bounds.json',points)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,t in zip(axes,['symmetric','asymmetric']):
        for arm,col,label in [('rl','#c25736','CABLE'),('fixed','#267b93','Fixed rule')]:
            p=[r for r in points if r['topology']==t and r['arm']==arm]
            ax.scatter([r['bound_ms'] for r in p],[r['cct_ms'] for r in p],c=col,s=18,alpha=.55,label=label)
        ax.plot([.1,150],[.1,150],ls='--',c='#75818d',lw=1,label='Payload service lower bound')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.1,150);ax.set_ylim(.1,150);ax.set_title(t.capitalize());ax.set_xlabel('Busiest destination link: payload service time (ms)');ax.set_ylabel('Measured collective completion time (ms)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('All 384 original CABLE / fixed runs: assigned work predicts completion time',fontweight='bold');save(fig,'service_time_evidence')
    summary=dict(new_runs=len(rows),new_flow_instances=sum(r['offered'] for r in rows),original_runs=len(old),reconstructed_original_runs=len(points),
                 core_runs=612,decomposition_runs=288,all_complete=True,
                 cct_over_payload_bound_quantiles=np.percentile([r['ratio'] for r in points],[0,50,95,100]).tolist(),
                 figures={f.name:env.sha(f) for f in figdir.iterdir()},aggregates=agg)
    env.dump(root/'summary.json',summary)
if __name__=='__main__':main()
