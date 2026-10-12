#!/usr/bin/env python3
"""All-condition tails, causal learning attribution and adaptation figures."""
import argparse,json,csv,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator, FuncFormatter
from analyze_online_routing import FAMS,TOPOS,FLAB
ARMS=['ecmp','wecmp','drill','conga','letflow','conweave','fixed','cable','online','hold_off','hold_on','ac_off','ac_ack','ac_on']
LAB=dict(zip(ARMS,['ECMP','WECMP','DRILL','CONGA','LetFlow','ConWeave*','Fixed score','Frozen CABLE','Online-QR','Initial / off','Initial / on','Reselect / off','ACK-A2C','Flow-A2C']))
COL=dict(zip(ARMS,['#9b9da0','#aa4499','#c2a44f','#228833','#66ccee','#cc6677','#999933','#726ac4','#0099aa','#c9ac86','#aa7733','#ee9977','#bb5599','#14567a']))

def analyze(task,rows,p):
 assert len(rows)==p['primary_runs'] and len({r['name'] for r in rows})==len(rows)
 assert all(r['analysis_ok'] and r['all_complete'] for r in rows),'Incomplete runs must be reported before uncensored tail aggregation'
 for r in rows:
  if 'fac_audit' in r:
   a=r['fac_audit'];assert r['arm'] in ['hold_on','hold_off'] or 'path_changes' in r['actor_audit'];r['weighted_cost_us']=(a['area_ns']+2*a['excess95_ns']+4*a['excess99_ns'])/r['offered']/1000
 lookup={(r['family'],r['topology_name'],r['seed'],r['arm'],r['fit']):r for r in rows};seeds=p['test_seeds'];fits=p['fits'];rng=np.random.RandomState(318691);B=10000
 bs=rng.randint(len(seeds),size=(B,len(seeds)));bf={k:rng.randint(k,size=(B,k)) for k in [1,2,3]};bd={k:rng.randint(k,size=(B,k)) for k in [1,2,3]}
 def get(f,t,s,a,k,m,phase):
  r=lookup[f,t,s,a,k]
  if phase>=0:return r['phases'][str(phase)][m]
  if m=='job_us':return float(np.exp(np.mean([np.log(v[m]) for v in r['phases'].values()])))
  return r[m]
 @lru_cache(None)
 def stat(num,den,m,families=tuple(FAMS),topos=tuple(TOPOS),phase=-1):
  kn,kd=fits.get(num,1),fits.get(den,1);paired=num in ['ac_on','ac_off','hold_on','hold_off','ac_ack'] and den in ['ac_on','ac_off','hold_on','hold_off','ac_ack']
  arr=np.array([[[np.mean([np.log(get(f,t,s,num,k,m,phase)/get(f,t,s,den,k if paired else j,m,phase)) for f in families for t in topos]) for j in range(kd)] for k in range(kn)] for s in seeds])
  draws=arr[bs[:,:,None,None],bf[kn][:,None,:,None],bd[kd][:,None,None,:]].mean(axis=(1,2,3));ci=np.exp(np.percentile(draws,[2.5,97.5]));val=float(np.exp(arr.mean()))
  return dict(value=val,ci95=ci.tolist(),delta_pct=100*(val-1),delta_ci95=(100*(ci-1)).tolist())
 summary=dict(runs=len(rows),flow_instances=sum(r['offered'] for r in rows),all_complete=True,selection=json.loads((task/'selection.json').read_text()),calibration=json.loads((task/'calibration.json').read_text()),overall={},families={},topologies={},attribution={},phases={},collective_cct={})
 for a in ARMS[:-1]:summary['overall'][a]={m:stat('ac_on',a,m) for m in ['p95_us','p99_us']}
 for f in FAMS:summary['families'][f]={a:{m:stat('ac_on',a,m,(f,)) for m in ['p95_us','p99_us']} for a in ARMS[:-1]}
 for t in TOPOS:summary['topologies'][t]={a:{m:stat('ac_on',a,m,tuple(FAMS),(t,)) for m in ['p95_us','p99_us']} for a in ARMS[:-1]}
 for num,den in [('ac_on','ac_off'),('hold_on','hold_off'),('ac_on','hold_on'),('ac_off','hold_off'),('ac_on','ac_ack')]:
  summary['attribution'][num+'/'+den]={g:{m:stat(num,den,m,tuple(FAMS) if g=='overall' else (g,)) for m in ['p95_us','p99_us']} for g in ['overall']+FAMS}
 for f in ['shift','bursty','alltoall','allreduce']:
  summary['phases'][f]={str(k):{a:{m:stat('ac_on',a,m,(f,),tuple(TOPOS),k) for m in ['p95_us','p99_us']} for a in ['ac_off','cable','online','conweave']} for k in range(6)}
 for f in ['alltoall','allreduce']:summary['collective_cct'][f]={a:stat('ac_on',a,'job_us',(f,)) for a in ARMS[:-1]}
 summary['reward_objective']={f:stat('ac_on','ac_off','weighted_cost_us',tuple(FAMS) if f=='overall' else (f,)) for f in ['overall']+FAMS}
 summary['activity']={}
 for a in ['ac_on','ac_off','hold_on','hold_off','ac_ack']:
  rr=[r for r in rows if r['arm']==a];aa=[r['fac_audit'] for r in rr];leaves=[v for r in aa for v in r['leaves']]
  summary['activity'][a]=dict(runs=len(rr),updates=sum(v['updates'] for v in leaves),actor_updates=sum(r['actor_parameter_updates'] for r in rr),decisions=sum(v['decisions'] for v in leaves),trained_decisions=sum(v['trained_decisions'] for v in leaves),decisions_after_first_actor_update=sum(r['decisions_after_first_actor_update'] for r in rr),wire_bytes=sum(r['wire_bytes'] for r in aa),path_changes=sum(r['actor_audit'].get('path_changes',0) for r in rr),leaves_with_update_before_last_decision=sum(v['first_update_ns']>=0 and v['first_update_ns']<v['last_decision_ns'] for v in leaves),total_leaves=len(leaves))
 return summary,stat,lookup

def figures(out,summary,stat,lookup,p):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 def save(fig,name):
  for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.08)
  plt.close(fig)
 def errors(ax,vals,ys,color,marker,label):
  x=np.array([v['delta_pct'] for v in vals]);ci=np.array([v['delta_ci95'] for v in vals]);ax.errorbar(x,ys,xerr=np.maximum(0,[x-ci[:,0],ci[:,1]-x]),fmt=marker,color=color,label=label,capsize=2,ms=4,lw=1.2)
 groups=['overall']+FAMS
 comparisons=[('ac_on','ac_off','Learning with reselection'),('hold_on','hold_off','Learning: initial path only'),('ac_on','ac_ack','Active-flow vs. ACK reward')]
 fig,axs=plt.subplots(1,3,figsize=(11,4),sharey=True)
 for ax,(num,den,title) in zip(axs,comparisons):
  for i,m in enumerate(['p95_us','p99_us']):errors(ax,[stat(num,den,m,tuple(FAMS) if f=='overall' else (f,)) for f in groups],np.arange(7)+(i-.5)*.22,['#14567a','#cc6677'][i],['o','s'][i],m[:3].upper())
  ax.axvline(0,c='#777',lw=.8);ax.grid(axis='x',c='#e7ebef',lw=.6);ax.set_axisbelow(True);ax.set_title(title);ax.set_xlabel('FCT change relative to control (%)');ax.set_yticks(range(7),['All conditions']+[FLAB[f] for f in FAMS]);ax.set_ylim(6.5,-.5)
 fig.legend(*axs[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False,bbox_to_anchor=(.57,1.02));fig.tight_layout(rect=[0,0,1,.94]);save(fig,'learning_attribution')
 # All methods / both topologies, explicit direction and identical symmetric scale.
 matrix=np.array([[stat('ac_on',a,m,(f,),(t,))['delta_pct'] for a in ARMS[:-1] for m in ['p95_us','p99_us']] for f in FAMS for t in TOPOS]);lim=max(10,np.ceil(abs(matrix).max()/10)*10)
 fig,ax=plt.subplots(figsize=(15,5.5));im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim));ax.set_yticks(range(12),[FLAB[f]+' / '+('Sym.' if t=='symmetric' else 'Asym.') for f in FAMS for t in TOPOS]);ax.set_xticks(range(26),['95','99']*13);ax.tick_params(length=0)
 for j,a in enumerate(ARMS[:-1]):
  ax.text(2*j+.5,-.85,LAB[a].replace(' / ','\n').replace('Frozen ','Frozen\n').replace('Fixed ','Fixed\n'),ha='center',va='bottom',fontsize=8)
  if j:ax.axvline(2*j-.5,c='white',lw=2)
 for i in range(12):
  for j in range(26):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=6.5,color='white' if abs(matrix[i,j])>.62*lim else '#222')
  if i%2:ax.axhline(i+.5,c='white',lw=1.5)
 ax.set_xlabel('P95 / P99 for each comparator');fig.colorbar(im,ax=ax,fraction=.018,pad=.015,label='Flow-A2C FCT change (%); negative is faster');fig.tight_layout();save(fig,'all_condition_comparisons')
 # Absolute tail latencies (no ratios hidden), topology panels and all methods.
 for m in ['p95_us','p99_us']:
  fig,axs=plt.subplots(2,3,figsize=(13,6.8))
  for ax,f in zip(axs.flat,FAMS):
   for ti,t in enumerate(TOPOS):
    vals=[];los=[];his=[]
    for a in ARMS:
     vv=[lookup[f,t,s,a,k][m]/1000 for s in p['test_seeds'] for k in range(p['fits'].get(a,1))];vals.append(np.exp(np.mean(np.log(vv))));los.append(min(vv));his.append(max(vv))
    vals=np.array(vals);ax.errorbar(np.arange(len(ARMS))+(ti-.5)*.24,vals,yerr=[vals-los,his-vals],color=['#14567a','#cc6677'][ti],fmt=['o','s'][ti],ms=3,capsize=1.5,lw=.8,label=t.capitalize())
   ax.set_yscale('log');lo,hi=ax.get_ylim();ticks=(LogLocator(base=10,subs=[1,2,5]).tick_values(lo,hi) if hi/lo>4 else MaxNLocator(nbins=4).tick_values(lo,hi));ax.set_yticks([v for v in ticks if lo<=v<=hi]);ax.set_ylim(lo,hi);ax.yaxis.set_minor_locator(NullLocator());ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));ax.set_title(FLAB[f]);ax.set_xticks(range(len(ARMS)),[LAB[a] for a in ARMS],rotation=65,ha='right');ax.set_ylabel(m[:3].upper()+' FCT (ms; log scale)');ax.grid(axis='y',which='both',color='#e7ebef',lw=.5);ax.set_axisbelow(True)
  fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.suptitle('Points: geometric mean; whiskers: full range across traffic and learner seeds',y=.955,fontsize=9);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'absolute_'+m[:3])
 # Equal axes per 4-panel figure. Show all six chronological phases/jobs.
 selected=['ecmp','conga','conweave','cable','online','ac_off','ac_ack','ac_on']
 for name,families in [('stream_adaptation',['shift','bursty']),('collective_adaptation',['alltoall','allreduce'])]:
  fig,axs=plt.subplots(2,2,figsize=(10,5.5),sharex=True);bounds=[1.]
  for j,f in enumerate(families):
   for i,m in enumerate(['p95_us','p99_us']):
    ax=axs[i,j]
    for a in selected:
     v=[stat(a,'ecmp',m,(f,),tuple(TOPOS),k) for k in range(6)];y=np.array([x['value'] for x in v]);ci=np.array([x['ci95'] for x in v]);bounds.extend(ci.ravel());ax.plot(range(1,7),y,label=LAB[a],color=COL[a],lw=2 if a=='ac_on' else 1,marker='o' if a=='ac_on' else '.',ms=4,ls='--' if a=='ac_off' else '-');ax.fill_between(range(1,7),ci[:,0],ci[:,1],color=COL[a],alpha=.07,lw=0)
    ax.set_title(FLAB[f]+' / '+m[:3].upper());ax.axhline(1,c='#888',ls=':',lw=.8);ax.grid(axis='y',c='#e7ebef',lw=.5);ax.set_ylabel('FCT / ECMP FCT');ax.set_xticks(range(1,7));ax.set_axisbelow(True)
    if i==1:ax.set_xlabel('50 ms arrival phase' if name=='stream_adaptation' else 'Collective job')
  lo=max(0,np.floor((min(bounds)-.02)*10)/10);hi=np.ceil((max(bounds)+.02)*10)/10
  for ax in axs.flat:ax.set_ylim(lo,hi)
  fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=4,loc='upper center',frameon=False);fig.tight_layout(rect=[0,0,1,.91]);save(fig,name)
 # Job completion differs from per-transfer FCT; retain both collective families.
 fig,axs=plt.subplots(1,2,figsize=(10,3.2),sharey=True);bounds=[1.]
 for ax,f in zip(axs,['alltoall','allreduce']):
  for a in selected:
   vv=[stat(a,'ecmp','job_us',(f,),tuple(TOPOS),k) for k in range(6)];y=np.array([v['value'] for v in vv]);ci=np.array([v['ci95'] for v in vv]);bounds.extend(ci.ravel());ax.plot(range(1,7),y,label=LAB[a],c=COL[a],lw=2 if a=='ac_on' else 1,marker='o' if a=='ac_on' else '.',ms=4,ls='--' if a=='ac_off' else '-');ax.fill_between(range(1,7),ci[:,0],ci[:,1],color=COL[a],alpha=.08,lw=0)
  ax.set_title(FLAB[f]);ax.set_xlabel('Collective job');ax.set_xticks(range(1,7));ax.axhline(1,c='#888',ls=':',lw=.8);ax.grid(axis='y',c='#e7ebef',lw=.5);ax.set_axisbelow(True)
 axs[0].set_ylabel('Job CCT / ECMP CCT');axs[0].set_ylim(max(0,np.floor((min(bounds)-.02)*10)/10),np.ceil((max(bounds)+.02)*10)/10);fig.legend(*axs[0].get_legend_handles_labels(),ncol=4,loc='upper center',frameon=False);fig.tight_layout(rect=[0,0,1,.87]);save(fig,'collective_completion')
 # Audit the preselected first held-out seed/fit, not best-looking examples.
 fig,axs=plt.subplots(3,2,figsize=(10,6.3),sharex='col')
 for j,f in enumerate(['shift','allreduce']):
  r=lookup[f,'asymmetric',p['test_seeds'][0],'ac_on',0];by={}
  for u in csv.DictReader((Path(r['folder'])/'transitions.csv.fac-updates.csv').open()):
   weights=np.array([float(u[f'a{k}']) for k in range(352)]);by.setdefault(int(u['node']),[]).append(((int(u['publish_ns'])-2000000000)/1e6,weights))
  for node,seq in by.items():
   ts=[v[0] for v in seq];norm=[np.linalg.norm(v[1]-seq[0][1]) for v in seq];axs[0,j].plot(ts,range(1,len(ts)+1),lw=.9,alpha=.7);axs[1,j].plot(ts,norm,lw=.9,alpha=.7)
  tr=json.loads(Path(json.loads((Path(r['folder'])/'job.json').read_text())['traffic_metadata']).read_text());first=min(q['release_ns'] for q in tr['flows']);agg={}
  for h in csv.DictReader((Path(r['folder'])/'transitions.csv.fac-host.csv').open()):agg[int(h['end_ns'])]=agg.get(int(h['end_ns']),0)+int(h['active'])
  axs[2,j].plot([(t-2000000000)/1e6 for t in sorted(agg)],[agg[t] for t in sorted(agg)],c='#14567a',lw=1.2);axs[0,j].set_title(FLAB[f]+' / asymmetric');axs[2,j].set_xlabel('Time since traffic release (ms)')
  for ax in axs[:,j]:
   for t in ([50*k for k in range(1,6)] if f=='shift' else [150*k for k in range(1,6)]):ax.axvline(t,c='#bbb',ls=':',lw=.6)
   ax.grid(axis='y',c='#e7ebef',lw=.5);ax.set_axisbelow(True)
 axs[0,0].set_ylabel('Updates per leaf');axs[1,0].set_ylabel('Actor distance from first update');axs[2,0].set_ylabel('Unfinished flows');fig.tight_layout();save(fig,'training_timeline')

def main():
 a=argparse.ArgumentParser();a.add_argument('--experiment-dir',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);args=a.parse_args();p=json.loads((args.experiment_dir/'protocol.json').read_text());rows=json.loads((args.experiment_dir/'main-results.json').read_text());args.output_dir.mkdir(parents=True,exist_ok=True);summary,stat,lookup=analyze(args.experiment_dir,rows,p);(args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');figures(args.output_dir,summary,stat,lookup,p);(args.output_dir/'figure-manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output_dir.iterdir() if f.suffix in ['.pdf','.png']},indent=2)+'\n');print(json.dumps(summary['overall'],indent=2))
if __name__=='__main__':main()
