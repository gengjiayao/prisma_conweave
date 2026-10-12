#!/usr/bin/env python3
"""Audit paired native runs, export reproducible statistics and publication figures."""
import argparse,hashlib,json,sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
A=argparse.ArgumentParser();source=A.add_mutually_exclusive_group(required=True)
source.add_argument('--experiment-dir',type=Path);source.add_argument('--compact-data',type=Path)
A.add_argument('--paper-dir',type=Path,required=True)
a=A.parse_args();paper=a.paper_dir;task=a.experiment_dir or paper/'data/replot'
task.mkdir(parents=True,exist_ok=True)
if a.experiment_dir:
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
 import path_rollout as env
else:
 env=SimpleNamespace(sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest())
FAM=['uniform','storage','mixed','mixed_fast'];TOP=['symmetric','asymmetric'];LOAD=[.2,.35,.5];SEEDS=list(range(88001,88009));FITS=[9101,9102,9103]
ARMS=['ecmp','wecmp','drill','conga','letflow','conweave','fixed','rl']
LABEL={'ecmp':'ECMP','wecmp':'Weighted ECMP','drill':'DRILL','conga':'CONGA','letflow':'LetFlow','conweave':'ConWeave (adapted)','fixed':'Fixed score','rl':'CABLE','start':'Earlier checkpoint','no_remote':'Remote terms removed','no_history':'History terms removed','neutral':'Zero score'}
COL=dict(zip(ARMS,['#888888','#AA4499','#DDCC77','#228833','#66CCEE','#EE6677','#CCBB44','#225588']))
MARK=dict(zip(ARMS,['x','v','s','D','P','^','h','o']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.4,'savefig.facecolor':'white'})
figdir=paper/'figures';datadir=paper/'data';figdir.mkdir(exist_ok=True);datadir.mkdir(exist_ok=True)
rows=[]
if a.experiment_dir:
 for stage,expected in [('main',1920),('ablate',640)]:
  r=json.loads((task/f'{stage}-results.json').read_text());assert len(r)==expected
  rows+=r
else:
 rows=json.loads(a.compact_data.read_text())
assert len({r['name'] for r in rows})==2560
# Recompute every reported full-run percentile directly from the raw completion log.
# env.load independently reconciles flow keys, sizes, and starts against offered traces.
audit=[]
for r in rows if a.experiment_dir else []:
 assert r.get('analysis_ok'),(r['name'],r.get('error'))
 assert r['native_ok'] and r['all_complete'],('Campaign contains incomplete runs: report separately before plotting',r['name'])
 folder=Path(r['folder']);_,flows=env.load(folder)
 assert env.sha(folder/'fct_output_file.txt')==r['fct_sha256']
 assert env.sha(r['binary'])==r['binary_sha256']
 assert env.sha(r['topology'])==r['topology_sha256']
 if r['model']:assert env.sha(r['model'])==r['model_sha256']
 vals=np.array([f['fct_us'] for f in flows.values()])
 for q in [50,95,99,99.9]:assert np.isclose(np.percentile(vals,q),r[f'p{q:g}_us'],rtol=1e-12,atol=1e-8)
 assert len(flows)==r['offered']
 audit.append(dict(name=r['name'],offered=r['offered'],fct_sha256=r['fct_sha256']))
lookup={(r['topology_name'],r['load'],r['family'],r['traffic_seed'],r['arm'],r['fit']):r for r in rows}
assert len(lookup)==len(rows)
def get(t,l,f,s,arm,k):return lookup[t,l,f,s,arm,k if arm in ['rl','start','no_remote','no_history'] else 0]
def value(r,metric):
 if ':' in metric:g,m=metric.split(':');return r['groups'][g][m]
 return r[metric]
# Deterministic two-way block bootstrap: resample eight traffic-seed blocks and
# three paired optimizer-fit identities. Workloads and selected topologies/loads stay fixed.
rng=np.random.RandomState(271828);bs=rng.randint(0,8,(10000,8));bf=rng.randint(0,3,(10000,3))
def interval(logarray):
 assert logarray.shape==(8,3)
 draws=logarray[bs[:,:,None],bf[:,None,:]].mean(axis=(1,2))
 return np.exp(logarray.mean()),np.exp(np.percentile(draws,[2.5,97.5])).tolist()
def stat(arm,metric,topos=TOP,loads=LOAD,families=FAM,ratio=False,reference=None):
 arr=np.zeros((8,3))
 for si,s in enumerate(SEEDS):
  for ki,k in enumerate(FITS):
   v=[]
   for t in topos:
    for l in loads:
     for f in families:
      base=value(get(t,l,f,s,arm,k),metric)
      num=value(get(t,l,f,s,'rl',k),metric) if ratio else 1.
      assert base and (not ratio or num),(t,l,f,s,arm,metric)
      assert not (ratio and reference)
      ref=value(get(t,l,f,s,reference,k),metric) if reference else 1.
      v.append(np.log(num/base if ratio else base/ref))
   arr[si,ki]=np.mean(v)
 mean,ci=interval(arr)
 return {'value':mean,'ci95':ci,'delta_pct':100*(mean-1) if ratio else None}
summary=dict(runs=len(rows),main_runs=1920,ablation_runs=640,all_complete=True,flow_instances=sum(r['offered'] for r in rows),
             main_flow_instances=sum(r['offered'] for r in rows if r['phase']=='main'),
             unique_traces=len({r['traffic_sha256'] for r in rows}),ratios={},controls={},per_topology={},per_load={},workloads={})
for arm in ARMS[:-1]:
 summary['ratios'][arm]={m:stat(arm,m,ratio=True) for m in ['p95_us','p99_us','mean_us','makespan_us']}
for t in TOP:
 summary['per_topology'][t]={arm:{m:stat(arm,m,[t],ratio=True) for m in ['p95_us','p99_us','makespan_us']} for arm in ARMS[:-1]}
for l in LOAD:
 summary['per_load'][str(l)]={arm:{m:stat(arm,m,loads=[l],ratio=True) for m in ['p95_us','p99_us']} for arm in ARMS[:-1]}
for arm in ['start','no_remote','no_history','neutral','fixed']:
 summary['controls'][arm]={m:stat(arm,m,loads=[.35],ratio=True) for m in ['p95_us','p99_us','mean_us','makespan_us']}
for t in TOP:
 summary['workloads'][t]={f:{arm:{m:stat(arm,m,[t],families=[f],ratio=True) for m in ['p95_us','p99_us']} for arm in ARMS[:-1]} for f in FAM}
summary['size_groups']={g:{arm:{m:stat(arm,f'{g}:{m}',ratio=True) for m in ['p95_us','p99_us']} for arm in ['conga','conweave','fixed']} for g in ['small','medium','large']}
(task/'analysis-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(datadir/'current_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if a.experiment_dir:(task/'raw-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
# A portable compact dataset for independent plot/statistic reproduction.
compact=[{k:r[k] for k in ['name','phase','topology_name','load','family','traffic_seed','arm','fit','offered','completed','native_ok','all_complete','p95_us','p99_us','mean_us','makespan_us','groups','traffic_sha256','fct_sha256','model_sha256','topology_sha256','binary_sha256']} for r in rows]
(datadir/'current_runs.json').write_text(json.dumps(compact,indent=1)+'\n')
manifest={}
def save(fig,name,claim):
 fig.savefig(figdir/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
 fig.savefig(figdir/f'{name}.png',bbox_inches='tight',pad_inches=.03,dpi=180)
 plt.close(fig);manifest[name]={'claim':claim,'data':'data/current_runs.json','pdf_sha256':env.sha(figdir/f'{name}.pdf')}
def clean(ax):ax.grid(axis='y',color='#e4e7eb',lw=.6);ax.set_axisbelow(True)
# Paired ECMP-normalized FCT exposes differences without logarithmic compression.
# Every panel uses the same linear range, determined by all methods' full intervals.
line_points=[]
def ratio_limits(axes,bounds):
 lo=max(0.,np.floor((min(bounds)-.025)/.1)*.1)
 hi=np.ceil((max(bounds)+.025)/.1)*.1
 for ax in np.asarray(axes).flat:
  ax.set_ylim(lo,hi);ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.1 if hi-lo<=.9 else (.2 if hi-lo<=1.8 else .5)))
  ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
  ax.axhline(1.,color='#707780',lw=.8,ls=':',zorder=1)
 return [float(lo),float(hi)]
tail_bounds=[1.]

fig,axes=plt.subplots(2,2,figsize=(7,4.25),sharex=True,sharey=True)
for j,t in enumerate(TOP):
 for i,m in enumerate(['p95_us','p99_us']):
  ax=axes[i,j];clean(ax)
  for arm in ARMS:
   vals=[stat(arm,m,[t],[l],reference='ecmp') for l in LOAD]
   y=np.array([v['value'] for v in vals]);ci=np.array([v['ci95'] for v in vals])
   tail_bounds.extend(ci.ravel())
   line_points.extend(dict(figure='current_tails',topology=t,metric=m,arm=arm,load=l,reference='ecmp',value=v['value'],ci95=v['ci95']) for l,v in zip(LOAD,vals))
   ax.plot(LOAD,y,label=LABEL[arm],color=COL[arm],marker=MARK[arm],ms=5.5 if arm=='fixed' else 4,markerfacecolor='none' if arm=='fixed' else COL[arm],linestyle='--' if arm=='fixed' else '-',lw=2 if arm=='rl' else 1.2,zorder=6 if arm=='fixed' else (5 if arm=='rl' else 2))
   ax.fill_between(LOAD,ci[:,0],ci[:,1],color=COL[arm],alpha=.10,linewidth=0)
  ax.set_xticks(LOAD,['20%','35%','50%']);ax.set_title(f'({chr(97+2*i+j)}) {t.capitalize()} / P{95 if i==0 else 99}')
  if j==0:ax.set_ylabel('FCT / ECMP FCT\n(lower is better)')
  if i==1:ax.set_xlabel('Nominal host load')
tail_limits=ratio_limits(axes,tail_bounds)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,bbox_to_anchor=(.5,1.03),columnspacing=1.3)
fig.tight_layout(rect=[0,0,1,.91],h_pad=1.2,w_pad=1.1)
save(fig,'current_tails','P95/P99 divided by paired ECMP FCT; common linear axis, ECMP=1; geometric means and 95% paired-block intervals')
# Workload comparison includes every family and comparator; colors preserve zero.
from matplotlib.colors import TwoSlopeNorm
comp=ARMS[:-1];mat=[];yt=[]
for t in TOP:
 for f in FAM:
  mat.append([100*(stat(arm,m,[t],families=[f],ratio=True)['value']-1) for arm in comp for m in ['p95_us','p99_us']]);yt.append(('Sym. ' if t=='symmetric' else 'Asym. ')+{'uniform':'Uniform','storage':'Storage','mixed':'Mixed hotspot','mixed_fast':'Fast hotspot'}[f])
mat=np.array(mat);lim=max(5,np.ceil(np.max(np.abs(mat))/10)*10)
fig,ax=plt.subplots(figsize=(7,3.25));im=ax.imshow(mat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
ax.set_yticks(range(8),yt);ax.set_xticks(np.arange(14),['P95','P99']*7);ax.tick_params(length=0)
for j,arm in enumerate(comp):
 ax.text(j*2+.5,-.87,LABEL[arm].replace('Weighted ECMP','Weighted\nECMP').replace('ConWeave (adapted)','ConWeave\n(adapted)').replace('Fixed score','Fixed\nscore'),ha='center',va='bottom',fontsize=8)
 if j:ax.axvline(j*2-.5,c='white',lw=2)
for i in range(8):
 for j in range(14):ax.text(j,i,('0' if abs(mat[i,j])<.5 else f'{mat[i,j]:+.0f}'),ha='center',va='center',fontsize=8,color='white' if abs(mat[i,j])>lim*.60 else '#222222')
ax.axhline(3.5,c='white',lw=3);fig.colorbar(im,ax=ax,fraction=.026,pad=.02,label='CABLE FCT change (%)')
fig.tight_layout();save(fig,'current_workloads','Paired P95/P99 percentage change across all loads for each workload/topology and seven comparators; negative favors CABLE')
# Learning and coefficient removal with paired intervals, no averaging before ratios.
fig,axes=plt.subplots(2,2,figsize=(7,3.35),sharey='row')
for i,controls in enumerate([['fixed','start'],['no_remote','no_history','neutral']]):
 for j,t in enumerate(TOP):
  ax=axes[i,j];ax.axvline(0,color='#666666',lw=.8);ax.grid(axis='x',color='#e4e7eb',lw=.6);ax.set_axisbelow(True)
  for mi,m in enumerate(['p95_us','p99_us']):
   vals=[stat(arm,m,[t],[.35],ratio=True) for arm in controls];y=np.arange(len(controls))+(mi-.5)*.2
   x=np.array([v['value'] for v in vals])*100-100;ci=np.array([v['ci95'] for v in vals])*100-100
   ax.errorbar(x,y,xerr=np.stack([x-ci[:,0],ci[:,1]-x]),fmt='o' if mi==0 else 's',color=['#225588','#EE6677'][mi],ms=4,capsize=2,lw=1,label=['P95','P99'][mi])
  ax.set_title(t.capitalize()+(' / score training' if i==0 else ' / input removal'))
  ax.set_yticks(range(len(controls)),[LABEL[x] for x in controls]);ax.set_ylim(-.45,len(controls)-.55)
  if i==1:ax.set_xlabel('CABLE FCT change vs. control (%)')
for i in range(2):axes[i,0].invert_yaxis()
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.60,1.03))
fig.tight_layout(rect=[0,0,1,.95],w_pad=1.5,h_pad=1.3)
save(fig,'current_controls','Current-wire earlier checkpoint, fixed, neutral and frozen coefficient-group removal controls at35% load; negative favors full CABLE; separate scales expose small learning gains')
# Size-stratified ratios against the two feedback baselines; every topology/size/metric.
fig,axes=plt.subplots(2,3,figsize=(7,3.9),sharey=True)
for i,t in enumerate(TOP):
 for j,g in enumerate(['small','medium','large']):
  ax=axes[i,j];ax.axvline(0,color='#666666',lw=.8);ax.grid(axis='x',color='#e4e7eb',lw=.6)
  for mi,m in enumerate(['p95_us','p99_us']):
   vals=[stat(arm,f'{g}:{m}',[t],ratio=True) for arm in ['conga','conweave','fixed']]
   x=np.array([v['value'] for v in vals])*100-100;ci=np.array([v['ci95'] for v in vals])*100-100
   ax.errorbar(x,np.arange(3)+(mi-.5)*.2,xerr=np.stack([x-ci[:,0],ci[:,1]-x]),fmt=['o','s'][mi],color=['#225588','#EE6677'][mi],ms=3.5,capsize=2,lw=1,label=['P95','P99'][mi])
  ax.set_yticks(range(3),['CONGA','ConWeave\n(adapted)','Fixed score']);ax.invert_yaxis();ax.set_title(f'{t.capitalize()} / '+{'small':'≤8 KB','medium':'8–64 KB','large':'>64 KB'}[g])
  if i==1:ax.set_xlabel('CABLE FCT change (%)')
axes[0,2].legend(loc='best',frameon=False);fig.tight_layout(w_pad=1.2,h_pad=1.3)
save(fig,'current_sizes','P95/P99 within three flow-size groups, two topologies, and CONGA/ConWeave/fixed controls; all loads')
# End-to-end costs and completion, both topologies and all algorithms.
fig,axes=plt.subplots(1,3,figsize=(7,2.55),gridspec_kw={'width_ratios':[1,1,1.1]})
for j,m in enumerate(['mean_us','makespan_us']):
 ax=axes[j];clean(ax)
 for ti,t in enumerate(TOP):
  vals=[stat(arm,m,[t]) for arm in ARMS];x=np.arange(8)+(ti-.5)*.2
  y=np.array([v['value'] for v in vals])/1000;ci=np.array([v['ci95'] for v in vals])/1000
  ax.errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt=['o','^'][ti],color=['#225588','#EE6677'][ti],ms=3.5,capsize=1.5,lw=.8,label=t.capitalize())
 ax.set_xticks(range(8),['ECMP','WECMP','DRILL','CONGA','LetFlow','CW*','Fixed','CABLE'],rotation=65,ha='right');ax.set_title(['Mean FCT','Batch completion'][j]);ax.set_ylabel('Time (ms)')
ax=axes[2];counts=[sum(r['offered'] for r in rows if r['arm']==arm and r['phase']=='main') for arm in ARMS];x=np.arange(8)
ax.bar(x,np.array(counts)/1e6,color=[COL[arm] for arm in ARMS],width=.65)
ax.set_xticks(x,['ECMP','WECMP','DRILL','CONGA','LetFlow','CW*','Fixed','CABLE'],rotation=65,ha='right');ax.set_ylabel('Completed instances\n(millions)');ax.set_title('All offered flows complete');clean(ax)
axes[0].legend(frameon=False,fontsize=7.5);fig.tight_layout(w_pad=1.2)
save(fig,'current_completion','Mean and batch completion time with intervals; completed instances rather than unique flows, CABLE includes three fits')
# Per-fit consistency, paired workload ratios at35% load, appendix.
fit_bounds=[0.]
fig,axes=plt.subplots(1,2,figsize=(7,2.55),sharey=True)
for j,t in enumerate(TOP):
 ax=axes[j];ax.axhline(0,c='#888888',lw=.8);clean(ax)
 for ki,k in enumerate(FITS):
  vals=[]
  for f in FAM:
   vals.append(100*(np.exp(np.mean([np.log(get(t,.35,f,s,'rl',k)['p99_us']/get(t,.35,f,s,'conweave',0)['p99_us']) for s in SEEDS]))-1))
  fit_bounds.extend(vals)
  line_points.extend(dict(figure='current_fits',topology=t,fit=k,family=f,reference='conweave',delta_pct=v) for f,v in zip(FAM,vals))
  ax.plot(range(4),vals,marker=['o','s','^'][ki],color=['#225588','#EE6677','#228833'][ki],label=f'Fit {ki+1}')
 ax.set_xticks(range(4),['Uniform','Storage','Mixed hotspot','Fast hotspot'],rotation=20);ax.set_title(t.capitalize())
fit_limits=[float(5*np.floor((min(fit_bounds)-1)/5)),float(5*np.ceil((max(fit_bounds)+1)/5))]
for ax in axes:
 ax.set_ylim(*fit_limits);ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
axes[0].set_ylabel('P99 change vs. ConWeave* (%)');axes[1].legend(frameon=False);fig.tight_layout()
save(fig,'current_fits','All three trained fits shown separately,35% load, workload and topology transfer')
(datadir/'line_axis_points.json').write_text(json.dumps(dict(reference='paired ECMP for tails; adapted ConWeave for fit changes',tail_limits=tail_limits,fit_limits=fit_limits,points=line_points),indent=2)+'\n')
# Claim macros are generated from audited paired statistics.
mac={'MainRuns':'1,920','AblationRuns':'640','TotalRuns':'2,560','FlowInstances':f"{summary['flow_instances']:,}"}
for arm,key in [('conga','Conga'),('conweave','Conweave'),('ecmp','Ecmp'),('fixed','Fixed')]:
 for m,n in [('p95_us','Pninetyfive'),('p99_us','Pninetynine'),('makespan_us','Batch')]:
  v=summary['ratios'][arm][m];mac[key+n]=f"{abs(v['delta_pct']):.1f}";mac[key+n+'Signed']=f"{v['delta_pct']:+.1f}"
for arm,key in [('start','Start'),('no_remote','Remote'),('no_history','History')]:
 for m,n in [('p95_us','Pninetyfive'),('p99_us','Pninetynine')]:mac[key+n+'Signed']=f"{summary['controls'][arm][m]['delta_pct']:+.1f}"
(datadir/'current_results.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in mac.items()))
if (datadir/'current_diagram_manifest.json').exists():manifest.update(json.loads((datadir/'current_diagram_manifest.json').read_text()))
(datadir/'current_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['runs','flow_instances','ratios','controls']},indent=2))
