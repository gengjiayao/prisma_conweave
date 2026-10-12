#!/usr/bin/env python3
"""Paired held-out online-routing results, with explicit learning-off controls."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, NullLocator, FuncFormatter
FAMS=['uniform','storage','shift','bursty','alltoall','allreduce'];TOPOS=['symmetric','asymmetric']
ARMS=['ecmp','wecmp','drill','conga','letflow','conweave','fixed','cable','online_off','online']
LAB={'ecmp':'ECMP','wecmp':'WECMP','drill':'DRILL','conga':'CONGA','letflow':'LetFlow','conweave':'ConWeave*','fixed':'Fixed score','cable':'Frozen CABLE','online_off':'Learning off','online':'Online-QR'}
FLAB=dict(zip(FAMS,['Uniform','Storage','Traffic shift','Bursty incast','All-to-All','Ring All-Reduce']))
COL=dict(zip(ARMS,['#92979f','#AA4499','#B99D37','#228833','#55B5CC','#E86A79','#997A00','#7785c2','#d97706','#124E78']))
MARK=dict(zip(ARMS,['x','v','s','D','P','^','h','>','s','o']))

def analyze(rows,p):
 assert len(rows)==p['primary_runs']==1152 and len({r['name'] for r in rows})==1152
 assert all(r['analysis_ok'] and r['all_complete'] for r in rows),'Incomplete run: report failure and censored tails before aggregation'
 lookup={(r['family'],r['topology_name'],r['seed'],r['arm'],r['fit']):r for r in rows};seeds=p['test_seeds'];rng=np.random.RandomState(567891)
 bs=rng.randint(0,len(seeds),(10000,len(seeds)));bf=rng.randint(0,3,(10000,3));bc=rng.randint(0,3,(10000,3))
 def get(f,t,s,a,k,m,phase):
  r=lookup[f,t,s,a,k if a in ['online','online_off','cable'] else 0]
  if m=='job_us' and phase is None:return float(np.exp(np.mean([np.log(v['job_us']) for v in r['phases'].values()])))
  return r[m] if phase is None else r['phases'][str(phase)][m]
 def stat(arm,metric,families=FAMS,topos=TOPOS,phase=None,numerator='online',denominator=None):
  # Default comparison is online/comparator. For line curves specify method/ECMP.
  den=denominator or arm;num=numerator
  paired_fits={num,den}<=set(['online','online_off'])
  arr=np.empty((len(seeds),3,3))
  for si,s in enumerate(seeds):
   for k in range(3):
    for j in range(3):
     vals=[np.log(get(f,t,s,num,k,metric,phase)/get(f,t,s,den,k if paired_fits else j,metric,phase)) for f in families for t in topos]
     arr[si,k,j]=np.mean(vals)
  ci=np.exp(np.percentile(arr[bs[:,:,None,None],bf[:,None,:,None],bc[:,None,None,:]].mean(axis=(1,2,3)),[2.5,97.5]));point=float(np.exp(arr.mean()))
  return dict(value=point,ci95=ci.tolist(),delta_pct=100*(point-1),delta_ci95=(100*(ci-1)).tolist())
 summary=dict(runs=len(rows),all_complete=True,flow_instances=sum(r['offered'] for r in rows),families={},overall={},per_topology={},step=json.loads((Path(p['_task'])/'selection.json').read_text())['step'])
 for a in ARMS[:-1]:summary['overall'][a]={m:stat(a,m) for m in ['p95_us','p99_us']}
 for f in FAMS:summary['families'][f]={a:{m:stat(a,m,[f]) for m in ['p95_us','p99_us']} for a in ARMS[:-1]}
 for t in TOPOS:summary['per_topology'][t]={a:{m:stat(a,m,topos=[t]) for m in ['p95_us','p99_us']} for a in ARMS[:-1]}
 summary['collective_cct']={f:{a:stat(a,'job_us',[f]) for a in ARMS[:-1]} for f in ['alltoall','allreduce']}
 summary['collective_overlap']={a:{f:sum(v['job_us']>150000 for r in rows if r['arm']==a and r['family']==f for k,v in r['phases'].items() if int(k)<5) for f in ['alltoall','allreduce']} for a in ARMS}
 summary['phase_comparisons']={f:{str(k):{a:{m:stat(a,m,[f],phase=k) for m in ['p95_us','p99_us']} for a in ['online_off','cable','conga','conweave']} for k in range(3 if f in ['shift','bursty'] else 6)} for f in ['shift','bursty','alltoall','allreduce']}
 online=[r for r in rows if r['arm']=='online'];off=[r for r in rows if r['arm']=='online_off']
 summary['online_updates']=sum(v['updates'] for r in online for v in r['online_audit']['leaves'])
 summary['off_updates']=sum(v['updates'] for r in off for v in r['online_audit']['leaves'])
 summary['online_path_changes']=sum(r['actor_audit']['path_changes'] for r in online)
 summary['online_reselections']=sum(r['actor_audit']['path_reselections'] for r in online)
 summary['off_path_changes']=sum(r['actor_audit']['path_changes'] for r in off)
 summary['online_leaves_with_update_before_last_decision']=sum(v['first_update_ns']>0 and v['first_update_ns']<v['last_decision_ns'] for r in online for v in r['online_audit']['leaves'])
 return summary,stat

def activity_plot(out,rows):
 """Fixed examples: first test seed, first exploration seed, asymmetric fabric."""
 chosen=[next(r for r in rows if r['arm']=='online' and r['family']==f and r['seed']==93001 and r['fit']==0 and r['topology_name']=='asymmetric') for f in ['shift','allreduce']]
 fig,axes=plt.subplots(2,2,figsize=(9,4.5),sharex='col')
 for j,r in enumerate(chosen):
  bynode={}
  with (Path(r['folder'])/'transitions.csv.online-updates.csv').open() as f:
   for u in csv.DictReader(f):
    t=(int(u['update_ns'])-2000000000)/1e6;norm=np.sqrt(sum(float(v)**2 for k,v in u.items() if k.startswith('w')))
    bynode.setdefault(int(u['node']),[]).append((t,norm))
  arrays=[np.asarray(v) for v in bynode.values()];grid=np.linspace(0,max(v[-1,0] for v in arrays),500);norms=[];counts=[]
  for v in arrays:
   ix=np.searchsorted(v[:,0],grid,side='right');counts.append(ix);norms.append(np.r_[0,v[:,1]][ix])
  norms=np.asarray(norms);counts=np.asarray(counts)
  axes[0,j].plot(grid,counts.sum(axis=0)/1000,c=COL['online'],lw=1.7)
  axes[1,j].plot(grid,np.median(norms,axis=0),c=COL['online'],lw=1.7,label='Median leaf')
  axes[1,j].fill_between(grid,norms.min(axis=0),norms.max(axis=0),color=COL['online'],alpha=.15,label='Range across 8 leaves')
  axes[0,j].set_title(FLAB[r['family']]);axes[1,j].set_xlabel('Time since first traffic release (ms)')
  boundaries=[50,100] if r['family']=='shift' else [150*k for k in range(1,6)]
  for ax in axes[:,j]:
   for b in boundaries:ax.axvline(b,c='#999',ls=':',lw=.7)
   ax.set_xlim(0,grid[-1]);ax.grid(axis='y',c='#e5e9ef',lw=.5);ax.set_axisbelow(True)
 axes[0,0].set_ylabel('Published updates (thousands)');axes[1,0].set_ylabel('Learned weight L2 norm')
 fig.legend(*axes[1,0].get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.55,1.02),frameon=False)
 fig.tight_layout(rect=[0,0,1,.94])
 for ext in ['pdf','png']:fig.savefig(out/f'online_training_timeline.{ext}',bbox_inches='tight',pad_inches=.06,dpi=180)
 plt.close(fig)

def absolute_plots(out,rows,p):
 """Absolute tail latencies complement relative plots; bootstrap runs, not flows."""
 lookup={(r['family'],r['topology_name'],r['seed'],r['arm'],r['fit']):r for r in rows}
 rng=np.random.RandomState(567892);bs=rng.randint(0,6,(10000,6));bf=rng.randint(0,3,(10000,3));results={}
 for metric in ['p95_us','p99_us']:
  fig,axes=plt.subplots(2,3,figsize=(12,6.2))
  for ax,f in zip(axes.flat,FAMS):
   for ti,t in enumerate(TOPOS):
    points=[];intervals=[]
    for arm in ARMS:
     arr=np.array([[np.log(lookup[f,t,s,arm,k if arm in ['cable','online','online_off'] else 0][metric]/1000) for k in range(3)] for s in p['test_seeds']])
     value=float(np.exp(arr.mean()));ci=np.exp(np.percentile(arr[bs[:,:,None],bf[:,None,:]].mean(axis=(1,2)),[2.5,97.5]));points.append(value);intervals.append(ci)
     results.setdefault(f,{}).setdefault(t,{}).setdefault(arm,{})[metric]=dict(value_ms=value,ci95_ms=ci.tolist())
    points=np.array(points);intervals=np.array(intervals)
    ax.errorbar(np.arange(10)+(ti-.5)*.18,points,yerr=[points-intervals[:,0],intervals[:,1]-points],fmt=['o','s'][ti],ms=3.5,color=['#124E78','#d66b7b'][ti],capsize=1.8,label=t.capitalize())
   ax.set_title(FLAB[f]);ax.set_xticks(range(10),[LAB[a] for a in ARMS],rotation=55,ha='right');ax.set_yscale('log');ax.grid(axis='y',which='both',c='#e5e9ef',lw=.5);ax.set_axisbelow(True);ax.axvspan(8.65,9.35,color='#124E78',alpha=.05)
   lo,hi=ax.get_ylim();ticks=MaxNLocator(nbins=4).tick_values(lo,hi);ax.set_yticks([v for v in ticks if lo<=v<=hi]);ax.set_ylim(lo,hi);ax.yaxis.set_minor_locator(NullLocator());ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.3g}'))
   ax.set_ylabel(f"Geometric mean P{95 if metric=='p95_us' else 99} FCT (ms, log scale)")
  fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False)
  fig.tight_layout(rect=[0,0,1,.96],h_pad=1.1,w_pad=1.6)
  for ext in ['pdf','png']:fig.savefig(out/f'online_absolute_{metric[:3]}.{ext}',bbox_inches='tight',pad_inches=.06,dpi=180)
  plt.close(fig)
 (out/'absolute-tail-summary.json').write_text(json.dumps(results,indent=2)+'\n')

def plots(out,summary,stat):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,n):
  for e in ['pdf','png']:fig.savefig(out/f'{n}.{e}',bbox_inches='tight',pad_inches=.06,dpi=180)
  plt.close(fig)
 # Keep every method, family, topology and percentile.
 fig,axes=plt.subplots(2,2,figsize=(10,5.8),sharex=True);bounds=[1.]
 for i,m in enumerate(['p95_us','p99_us']):
  for j,t in enumerate(TOPOS):
   ax=axes[i,j]
   for arm in ARMS:
    vals=[stat('ecmp',m,[f],[t],numerator=arm,denominator='ecmp') for f in FAMS];y=np.array([v['value'] for v in vals]);ci=np.array([v['ci95'] for v in vals]);bounds.extend(ci.ravel())
    ax.plot(range(6),y,label=LAB[arm],color=COL[arm],marker=MARK[arm],ms=4,lw=2 if arm=='online' else 1.1,ls='--' if arm in ['online_off','fixed'] else '-',markerfacecolor='none' if arm in ['online_off','fixed'] else COL[arm],zorder=5 if arm=='online' else 2)
    ax.fill_between(range(6),ci[:,0],ci[:,1],color=COL[arm],alpha=.08,lw=0)
   ax.set_title(f'{t.capitalize()} / P{95 if i==0 else 99}');ax.grid(axis='y',c='#e5e9ef',lw=.5);ax.set_axisbelow(True);ax.axhline(1,c='#888',ls=':',lw=.8)
   ax.set_xticks(range(6),['Uniform','Storage','Shift','Bursty','All-to-All','All-Reduce'],rotation=20)
   if j==0:ax.set_ylabel('FCT / ECMP FCT (lower is better)')
 for ax in axes.flat:ax.set_ylim(max(0,np.floor((min(bounds)-.02)*10)/10),np.ceil((max(bounds)+.02)*10)/10)
 fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=5,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False)
 fig.tight_layout(rect=[0,0,1,.9],h_pad=1.4,w_pad=1.5);save(fig,'online_all_algorithms')
 # Full paired map retains adverse results; negative is faster.
 matrix=[];labels=[]
 for f in FAMS:
  for t in TOPOS:
   matrix.append([stat(a,m,[f],[t])['delta_pct'] for a in ARMS[:-1] for m in ['p95_us','p99_us']]);labels.append(FLAB[f]+' / '+('Sym.' if t=='symmetric' else 'Asym.'))
 matrix=np.array(matrix);lim=max(10,np.ceil(abs(matrix).max()/10)*10)
 fig,ax=plt.subplots(figsize=(11,5.2));im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim));ax.set_yticks(range(12),labels);ax.set_xticks(range(18),['P95','P99']*9);ax.tick_params(length=0)
 for j,a in enumerate(ARMS[:-1]):
  ax.text(2*j+.5,-.8,LAB[a].replace('Frozen ','Frozen\n').replace('Learning ','Learning\n').replace('Fixed ','Fixed\n'),ha='center',va='bottom')
  if j:ax.axvline(2*j-.5,color='white',lw=2)
 for i in range(12):
  for j in range(18):
   v=matrix[i,j];ax.text(j,i,('0' if abs(v)<.05 else f'{v:+.1f}'),ha='center',va='center',fontsize=7,color='white' if abs(v)>.60*lim else '#222')
  if i%2:ax.axhline(i+.5,c='white',lw=1.3)
 fig.colorbar(im,ax=ax,fraction=.022,pad=.02,label='Online-QR FCT change (%); lower is better');fig.tight_layout();save(fig,'online_paired_map')
 # Learning attribution, CABLE transfer and fixed-rule baseline, including all families.
 fig,axes=plt.subplots(1,3,figsize=(10,3.9),sharey=True)
 groups=[None]+FAMS
 for j,a in enumerate(['online_off','cable','fixed']):
  ax=axes[j];ax.axvline(0,c='#888',lw=.8);ax.grid(axis='x',c='#e5e9ef',lw=.6);ax.set_axisbelow(True)
  for mi,m in enumerate(['p95_us','p99_us']):
   vals=[stat(a,m,FAMS if f is None else [f]) for f in groups];x=np.array([v['delta_pct'] for v in vals]);ci=np.array([v['delta_ci95'] for v in vals])
   ax.errorbar(x,np.arange(len(groups))+(mi-.5)*.2,xerr=[x-ci[:,0],ci[:,1]-x],fmt=['o','s'][mi],color=['#124E78','#d66b7b'][mi],capsize=2,ms=4,label=['P95','P99'][mi])
  ax.set_title('Compared with '+LAB[a]);ax.set_xlabel('Online-QR FCT change (%)');ax.set_yticks(range(len(groups)),['All conditions']+[FLAB[f] for f in FAMS]);ax.set_ylim(6.5,-.5)
 fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False,bbox_to_anchor=(.57,1.01));fig.tight_layout(rect=[0,0,1,.92],w_pad=1.4);save(fig,'online_learning_attribution')
 selected=['ecmp','conga','conweave','fixed','cable','online_off','online']
 for name,families,nphase in [('online_phase_adaptation',['shift','bursty'],3),('online_collective_adaptation',['alltoall','allreduce'],6)]:
  fig,axes=plt.subplots(2,2,figsize=(9,5),sharex=True);bounds=[1.]
  for j,f in enumerate(families):
   for i,m in enumerate(['p95_us','p99_us']):
    ax=axes[i,j]
    for a in selected:
     vals=[stat('ecmp',m,[f],phase=k,numerator=a,denominator='ecmp') for k in range(nphase)];y=np.array([v['value'] for v in vals]);ci=np.array([v['ci95'] for v in vals]);bounds.extend(ci.ravel())
     ax.plot(range(nphase),y,label=LAB[a],color=COL[a],marker=MARK[a],ls='--' if a in ['online_off','fixed'] else '-',lw=2 if a=='online' else 1.1,ms=4);ax.fill_between(range(nphase),ci[:,0],ci[:,1],color=COL[a],alpha=.09,lw=0)
    ax.set_title(f'{FLAB[f]} / P{95 if i==0 else 99}');ax.grid(axis='y',c='#e5e9ef',lw=.5);ax.set_axisbelow(True);ax.axhline(1,c='#888',ls=':',lw=.8)
    ax.set_xticks(range(nphase),[str(k+1) for k in range(nphase)])
    if j==0:ax.set_ylabel('FCT / ECMP FCT')
    if i==1:ax.set_xlabel('50 ms arrival phase' if nphase==3 else 'Collective job (cold start at job 1)')
  for ax in axes.flat:ax.set_ylim(max(0,np.floor((min(bounds)-.02)*10)/10),np.ceil((max(bounds)+.02)*10)/10)
  fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.5,1.015),frameon=False);fig.tight_layout(rect=[0,0,1,.89],h_pad=1.1,w_pad=1.2);save(fig,name)
 # Collective completion is distinct from transfer tail latency.
 fig,axes=plt.subplots(1,2,figsize=(9,2.8),sharey=True);bounds=[1.]
 for j,f in enumerate(['alltoall','allreduce']):
  ax=axes[j]
  for a in ARMS:
   vals=[stat('ecmp','job_us',[f],phase=k,numerator=a,denominator='ecmp') for k in range(6)];y=np.array([v['value'] for v in vals]);ci=np.array([v['ci95'] for v in vals]);bounds.extend(ci.ravel())
   ax.plot(range(1,7),y,label=LAB[a],color=COL[a],marker=MARK[a],ls='--' if a in ['online_off','fixed'] else '-',lw=2 if a=='online' else 1.1,ms=4);ax.fill_between(range(1,7),ci[:,0],ci[:,1],color=COL[a],alpha=.07,lw=0)
  ax.set_title(FLAB[f]);ax.set_xticks(range(1,7));ax.set_xlabel('Collective job');ax.axhline(1,c='#888',ls=':',lw=.8);ax.grid(axis='y',c='#e5e9ef',lw=.5);ax.set_axisbelow(True)
 axes[0].set_ylabel('Job CCT / ECMP CCT')
 axes[0].set_ylim(max(0,np.floor((min(bounds)-.02)*10)/10),np.ceil((max(bounds)+.02)*10)/10)
 fig.legend(*axes[0].get_legend_handles_labels(),ncol=5,loc='upper center',bbox_to_anchor=(.5,1.08),frameon=False);fig.tight_layout(rect=[0,0,1,.86]);save(fig,'online_collective_completion')

def main():
 a=argparse.ArgumentParser();a.add_argument('--experiment-dir',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);args=a.parse_args();task=args.experiment_dir;p=json.loads((task/'protocol.json').read_text());p['_task']=str(task);rows=json.loads((task/'main-results.json').read_text());out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 summary,stat=analyze(rows,p);(out/'online-summary.json').write_text(json.dumps(summary,indent=2)+'\n');plots(out,summary,stat);activity_plot(out,rows);absolute_plots(out,rows,p)
 manifest={n.name:hashlib.sha256(n.read_bytes()).hexdigest() for n in out.iterdir() if n.suffix in ['.pdf','.png']};(out/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
