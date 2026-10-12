#!/usr/bin/env python3
"""Flow-A2C: frozen protocol, causal reward audit, matched learning controls."""
import argparse,csv,json,os,shutil,time,resource,random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import online_routing_evaluation as prev
from online_routing_evaluation import env,old,coll,np,BASE,FAMILIES,BASELINES

FAC_ARMS=['ac_on','ac_off','hold_on','hold_off','ac_ack']

def init(task):
 task.mkdir(parents=True,exist_ok=True)
 for d in ['models','topologies','traces','plans','source','runs']:(task/d).mkdir(exist_ok=True)
 binary=task/'binaries/scratch/network-load-balance'
 if not binary.exists():
  shutil.copytree(BASE/'binaries',task/'binaries',symlinks=True)
  shutil.copy2(env.NS/'build-cable/scratch/network-load-balance',binary)
  shutil.copy2(env.NS/'build-cable/libns3.19-point-to-point-optimized.so',task/'binaries/libns3.19-point-to-point-optimized.so')
  shutil.copy2(env.NS/'build-cable/libns3.19-network-optimized.so',task/'binaries/libns3.19-network-optimized.so')
 env.configure(task,binary)
 for n in ['fixed.txt','rl_9101.txt','rl_9102.txt','rl_9103.txt']:
  if not (task/'models'/n).exists():shutil.copy2(BASE/'paper/models'/n,task/'models'/n)
 for f in (BASE/'experiment/topologies').glob('*.txt'):
  if not (task/'topologies'/f.name).exists():shutil.copy2(f,task/'topologies'/f.name)
 if not (task/'protocol.json').exists():
  p=dict(created_epoch=time.time(),method='Flow-A2C: per-leaf on-policy 8-window actor-critic with global active-flow cost received through real packets',
   state='20 causal per-candidate inputs: local queue/work, received remote queue report/age/trend, link service, reservations, observed age, sent and unacked bytes, ACK progress/rate, gap, previous path, last received global active count. No final size or future FCT.',
   reward='Negative active-flow age integral plus 2x excess above development reference P95 and 4x excess above development reference P99. Integral equals sum FCT; excess is a surrogate, not a quantile.',
   learner='Independent per-leaf 20x16 tanh actor and critic; 8-step discounted bootstrapped returns, gamma=exp(-1ms/200ms), centered standardized advantage; Adam, actor/critic update L2 caps .10/.50. Model fixed within rollout; forwarding continues during report/200us compute wait. Gap actions omitted from updates.',
   initializer='Zero actor/critic output layer; Gaussian hidden features with std .15. Explicit queue/reservation score prior and stochastic .01 uniform exploration, identical for learning-on/off. No offline training.',
   control='Source NIC sends exact local flow-time interval integrals every1ms; leaf sums16host reports and sends7real broadcasts via spine; every leaf waits8aggregates. 120B payload; actual injection and queue contention counted. No reliable retransmission; audits detect stalls.',
   timing='8ms frozen-policy rollouts; publish200us after last required report. Native inference cost not modeled; compute delay is a configured budget, not hardware measurement.',
   period_ns=1000000,rollout_windows=8,compute_ns=200000,temperature=16.,critic_step=.01,
   development_seeds=[95101,95102],development_families=['uniform','storage','shift'],step_grid=[.001,.003,.01],
   calibration='First run fixed-rule baselines on12development conditions. Tau95/Tau99 are geometric means of run P95/P99, in ns, frozen before learning-rate search.',
   selection='Choose one common actor step using equal-weight mean log P95/P99 across both routing modes and all12development conditions. ACK arm uses same step. No held-out tuning.',
   test_seeds=[96101,96102,96103,96104],families=FAMILIES,topologies=['symmetric','asymmetric'],primary_runs=1056,
   arms=BASELINES+['cable','online']+FAC_ARMS,fits=dict(cable=3,online=2,**{a:2 for a in FAC_ARMS}),
   common={**old.COMMON,'SIMULATOR_EXTRA_TIME':'2.0'},conweave=old.CW,
   traffic='Streams300ms: six50ms arrival phases. Uniform/storage load.35. Shift uniform.20,mixed.50,mixed_fast.35 twice; bursty .20,.50,.35 twice. Collectives six32rank jobs150ms apart: Alltoall8/64KiB alternating, RingAllreduce1MiB/rank, exact dependency DAG.',
   metrics='All offered-flow P95/P99 including cold start; per-phase tails and collective CCT. Equal-weight log ratios across family/topology with traffic-seed/learner-fit bootstrap. Report losses and incomplete runs.',
   binary_sha256=env.sha(binary),library_sha256=env.sha(task/'binaries/libns3.19-point-to-point-optimized.so'),network_library_sha256=env.sha(task/'binaries/libns3.19-network-optimized.so'))
  env.dump(task/'protocol.json',p)
  for f in [Path(__file__),env.NS/'src/network/model/flow-id-num-tag.cc',env.NS/'src/point-to-point/model/flow-actor-critic.h',env.NS/'src/point-to-point/model/flow-actor-critic.inc',env.NS/'src/point-to-point/model/rdma-hw.cc',env.NS/'src/point-to-point/model/switch-node.cc',env.NS/'src/point-to-point/model/route-learning.inc',env.NS/'scratch/network-load-balance.cc']:
   shutil.copy2(f,task/'source'/f.name)
 p=json.loads((task/'protocol.json').read_text());assert env.sha(binary)==p['binary_sha256'];assert env.sha(task/'binaries/libns3.19-point-to-point-optimized.so')==p['library_sha256'];return p

def traffic(task,family,seed):
 # Reuse the verified collective DAG; extend stream construction to six phases.
 meta=task/'traces'/f'{family}_s{seed}.json'
 if meta.exists():return json.loads(meta.read_text())
 if family in ['alltoall','allreduce']:return prev.traffic(task,family,seed)
 flows=[]
 if family=='bursty':
  rng=random.Random(seed);sizes=[4096,32768,524288];weights=[.6,.3,.1];mean=sum(a*b for a,b in zip(sizes,weights))
  for phase,load in enumerate([.2,.5,.35]*2):
   for burst in range(100):
    count=round(128*load*1e9*.0005/(8*mean));dest=rng.randrange(128)
    for i in range(count):
     src=rng.randrange(127);src+=src>=dest;dst=dest if rng.random()<.5 else rng.randrange(128)
     if src==dst:dst=(dst+1)%128
     flows.append(dict(src=src,dst=dst,bytes=rng.choices(sizes,weights)[0],release_ns=2000000000+phase*50000000+burst*500000+rng.randrange(20001),iteration=phase))
  flows.sort(key=lambda q:q['release_ns'])
 else:
  for phase in range(6):
   fam=family if family!='shift' else ['uniform','mixed','mixed_fast'][phase%3];load=.35 if family!='shift' else [.2,.5,.35][phase%3]
   path=old.make_trace(task,fam,seed+100000*phase,load)
   for line in path.read_text().splitlines()[1:]:
    s,d,pg,b,t=line.split();flows.append(dict(src=int(s),dst=int(d),bytes=int(b),release_ns=round(float(t)*1e9)+phase*50000000,iteration=phase))
 trace=meta.with_suffix('.txt');trace.write_text(str(len(flows))+'\n'+''.join(f"{q['src']} {q['dst']} 3 {q['bytes']} {q['release_ns']/1e9:.9f}\n" for q in flows))
 x=dict(family=family,seed=seed,traffic=str(trace),traffic_sha256=env.sha(trace),duration=.3,flows=flows,plan=None,plan_sha256=None);env.dump(meta,x);return x

def prepare(task,p,phase):
 if phase=='smoke':seeds=[95099];families=['storage','allreduce'];specs=[(a,0,.003) for a in FAC_ARMS]
 elif phase=='calibrate':seeds=p['development_seeds'];families=p['development_families'];specs=[('fixed',0,0)]
 elif phase=='dev':
  seeds=p['development_seeds'];families=p['development_families'];specs=[(a,0,s) for a in ['ac_on','hold_on'] for s in p['step_grid']]+[(a,0,.003) for a in ['ac_off','hold_off']]
 else:
  seeds=p['test_seeds'];families=p['families'];step=json.loads((task/'selection.json').read_text())['step']
  specs=[(a,0,0) for a in BASELINES]+[('cable',k,0) for k in range(3)]+[('online',k,.08) for k in range(2)]+[(a,k,step) for a in FAC_ARMS for k in range(2)]
 calibration=json.loads((task/'calibration.json').read_text()) if (task/'calibration.json').exists() else dict(tau95_ns=20000000,tau99_ns=50000000)
 jobs=[]
 for fam in families:
  for seed in seeds:
   tr=traffic(task,fam,seed)
   for topo in p['topologies']:
    for arm,fit,step in specs:
     name=f'{phase}_{fam}_s{seed}_{topo}_{arm}_f{fit}_lr{step:g}';folder=task/'runs'/name
     if folder.exists():jobs.append(folder);continue
     env.trace=lambda *a,_p=Path(tr['traffic']),**kw:_p
     tf=task/'topologies'/('leaf_spine_128_750M_sym.txt' if topo=='symmetric' else 'leaf_spine_128_100G_asym_OS2_asymmetric.txt')
     fac=arm in FAC_ARMS;native=fac or arm in ['cable','fixed','online'];reselect=arm in ['online','ac_on','ac_off','ac_ack']
     model=task/'models'/(f'rl_{9101+fit}.txt' if arm=='cable' else 'fixed.txt') if native else None
     ex={**p['common'],'TOPOLOGY_FILE':str(tf),'RANDOM_SEED':'100'}
     if tr['plan']:ex.update(COLLECTIVE_PLAN_FILE=tr['plan'],COLLECTIVE_EVENT_FILE=str(folder/'collective-events.txt'))
     if native:ex.update(RL_SWITCH_MODE='9' if reselect else '8',RL_PATH_RESELECT=str(int(reselect)),RL_GATE_MODE='0',RL_PRESSURE_ENABLE='0',RL_SWITCH_EPSILON='0',RL_SWITCH_MARGIN_NS='0',RL_SWITCH_INTERVAL_NS='50000',RL_SWITCH_MODEL=str(model),RL_SWITCH_OUTPUT=str(folder/'transitions.csv'))
     else:
      ex.update(RL_SWITCH_MODE='0',CONGA_ACK_FEEDBACK='0')
      if arm=='conweave':ex.update(p['conweave'])
     if arm=='online':ex.update(ONLINE_ENABLE='1',ONLINE_LEARN='1',ONLINE_STEP=str(step),ONLINE_SEED=str(97101+fit),ONLINE_EXPLORATION='.02',ONLINE_RESERVATION_WEIGHT='1',ONLINE_UPDATE_DELAY_NS='20000')
     if fac:ex.update(FAC_ENABLE='1',FAC_LEARN=str(int(arm not in ['ac_off','hold_off'])),FAC_SEED=str(97101+fit),FAC_REWARD_MODE=str(int(arm=='ac_ack')),FAC_ACTOR_STEP=str(step),FAC_CRITIC_STEP=str(p['critic_step']),FAC_TEMPERATURE=str(p['temperature']),FAC_PERIOD_NS=str(p['period_ns']),FAC_COMPUTE_NS=str(p['compute_ns']),FAC_ROLLOUT_WINDOWS=str(p['rollout_windows']),FAC_TAU95_NS=str(calibration['tau95_ns']),FAC_TAU99_NS=str(calibration['tau99_ns']))
     f=env.make(name,scenario='uniform',seed=seed,method='oracle' if native else arm,duration=tr['duration'],remote=int(native),feedback=int(native),period=200,quantum=1024 if native else 0,gap=0 if native else 100,extra=ex)
     j=json.loads((f/'job.json').read_text());j.update(arm=arm,fit=fit,step=step,family=fam,seed=seed,topology_name=topo,phase=phase,traffic_metadata=str(task/'traces'/f'{fam}_s{seed}.json'),topology=str(tf),topology_sha256=env.sha(tf),model=str(model) if model else None,model_sha256=env.sha(model) if model else None,protocol_sha256=env.sha(task/'protocol.json'))
     env.dump(f/'job.json',j);jobs.append(f)
 env.dump(task/f'{phase}-jobs.json',list(map(str,jobs)));return jobs

def audit_fac(folder,row):
 j=json.loads((folder/'job.json').read_text());ex=j['extra'];a=json.loads((folder/'transitions.csv.fac-audit.json').read_text());row['fac_audit']=a
 fcts=[int(x.split()[6]) for x in (folder/'fct_output_file.txt').read_text().splitlines()]
 assert a['active']==0 and a['starts']==a['finishes']==row['offered']
 assert a['area_ns']==sum(fcts)
 for q in [95,99]:assert a[f'excess{q}_ns']==sum(max(0,t-int(ex[f'FAC_TAU{q}_NS'])) for t in fcts)
 starts={};costs=[]
 for r in csv.DictReader((folder/'transitions.csv.fac-events.csv').open()):
  k=(int(r['host']),int(r['flow']));t=int(r['ns'])
  if r['kind']=='start':assert k not in starts;starts[k]=t
  else:costs.append(t-starts.pop(k))
 assert not starts and sorted(costs)==sorted(fcts)
 # Verify causal packet reception and no stale-policy batches; frozen weights exact.
 maxreceipt={};host={};globalcost=defaultdict(lambda:np.zeros(3,dtype=np.int64));hostleaf={};received_host=set();broadcasts={};received_leaf=defaultdict(set)
 fields=['area_ns','excess95_ns','excess99_ns','active','old95','old99','starts','finishes']
 for r in csv.DictReader((folder/'transitions.csv.fac-host.csv').open()):
  host[int(r['host']),int(r['seq'])]=tuple(int(r[k]) for k in fields);globalcost[int(r['seq'])]+=np.array([int(r[k]) for k in fields[:3]])
 for r in csv.DictReader((folder/'transitions.csv.fac-receive.csv').open()):
  node,seq,kind=int(r['leaf']),int(r['seq']),int(r['kind']);recv,end=int(r['receive_ns']),int(r['end_ns']);assert recv>end
  maxreceipt[node,seq]=max(maxreceipt.get((node,seq),0),recv)
  if kind==1:
   origin=int(r['origin']);assert tuple(int(r[k]) for k in fields)==host[origin,seq];hostleaf[origin]=node;received_host.add((origin,seq))
  else:
   origin=int(r['origin']);data=tuple(int(r[k]) for k in fields);assert data==broadcasts.setdefault((origin,seq),data);received_leaf[node,seq].add(origin)
 # All host-to-leaf reports must arrive; no silent reward-report loss.
 assert len(hostleaf)==128
 # Final packets can still be in flight at stop; every trained interval must be complete.
 sums=defaultdict(lambda:np.zeros(8,dtype=np.int64))
 for (h,seq),data in host.items():sums[hostleaf[h],seq]+=data
 for key,data in broadcasts.items():assert tuple(sums[key])==data
 versions={};updates={};nupdates=0;actorupdates=0;initial={};firstactor={};frame_counts=defaultdict(int)
 for r in csv.DictReader((folder/'transitions.csv.fac-updates.csv').open()):
  node,block=int(r['node']),int(r['block']);pub=int(r['publish_ns']);last=int(r['last_report_ns']);v=int(r['version'])
  assert pub-last==int(ex['FAC_COMPUTE_NS']);assert last==max(maxreceipt[node,k] for k in range(int(r['first_window']),int(r['last_window'])+1))
  for seq in range(int(r['first_window']),int(r['last_window'])+1):
   assert len(received_leaf[node,seq])==7
   assert all((h,seq) in received_host for h in hostleaf)
  assert v==(block+1 if a['learn'] else 0)
  weights=tuple(float(r[f'{prefix}{k}']) for prefix in ['a','c'] for k in range(352))
  assert all(np.isfinite(weights));assert 0<=float(r['actor_delta'])<=.10000001 and 0<=float(r['critic_delta'])<=.50000001
  if not a['learn']:
   assert float(r['actor_delta'])==float(r['critic_delta'])==0;assert weights==initial.setdefault(node,weights)
  if float(r['actor_delta'])>0:actorupdates+=1;firstactor.setdefault(node,pub)
  updates[node,block]=(v,pub,int(r['decisions']));versions.setdefault(node,[]).append((pub,v));nupdates+=1
 for r in csv.DictReader((folder/'transitions.csv.fac-frames.csv').open()):
  node,block,seq=int(r['node']),int(r['block']),int(r['window']);v,pub,n=updates[node,block];assert int(r['version'])==v-int(a['learn']);frame_counts[node,block]+=int(r['decisions'])
  if int(ex['FAC_REWARD_MODE'])==0:assert np.isclose(float(r['cost']),np.dot(globalcost[seq],[1,2,4])/6400000000.,rtol=1e-12)
 for key,(v,pub,n) in updates.items():assert n==frame_counts[key]
 assert nupdates==sum(v['blocks'] for v in a['leaves'])
 decisions=0;trained=defaultdict(int);later=0
 import bisect
 for r in csv.DictReader((folder/'transitions.csv.fac-decisions.csv').open()):
  node,t,v=int(r['node']),int(r['ns']),int(r['version']);past=versions[node];idx=bisect.bisect_right(past,(t,10**9));expected=past[idx-1][1] if idx else 0
  assert v==expected and 0<float(r['probability'])<=1;decisions+=1;later+=node in firstactor and t>firstactor[node]
 assert decisions==sum(v['decisions'] for v in a['leaves'])
 if (folder/'transitions.csv.boundaries.csv').exists():
  with (folder/'transitions.csv.boundaries.csv').open() as f:
   for b in csv.DictReader(f):
    if int(b['previous_port']):assert int(b['cumulative_ack'])>=int(b['sent_end']) and int(b['sequence'])>=int(b['sent_end'])
 else:assert j['arm'] in ['hold_on','hold_off']
 row.update(actor_parameter_updates=actorupdates,decisions_after_first_actor_update=later,fac_exact_integral=True,fac_causal_updates=True)
 return row

def run_one(folder):
 out=folder/'fac-result.json'
 if out.exists():return json.loads(out.read_text())
 j=json.loads((folder/'job.json').read_text());native=False;t=time.time()
 try:
  env.execute(folder);native=True;row=prev.audit(folder)
  if j['arm'] in FAC_ARMS:row=audit_fac(folder,row)
 except Exception as e:
  import traceback;traceback.print_exc();row={k:j[k] for k in ['name','arm','fit','step','family','seed','topology_name','phase']};row.update(native_ok=native,analysis_ok=False,all_complete=False,error=repr(e),folder=str(folder))
 row['wall_seconds']=time.time()-t;env.dump(out,row);print('AUDIT',row['name'],row['all_complete'],row.get('p99_us'),row.get('error',''),flush=True);return row

def finish(task,p,phase,rows):
 assert all(r['all_complete'] and r['analysis_ok'] for r in rows),'Failures retained; resolve before advancing'
 if phase=='calibrate':env.dump(task/'calibration.json',dict(created_epoch=time.time(),**{f'tau{q}_ns':round(float(np.exp(np.mean([np.log(r[f'p{q}_us']*1000) for r in rows])))) for q in [95,99]},results_sha256=env.sha(task/'calibrate-results.json')))
 if phase=='dev':
  scores=[dict(step=s,score=float(np.mean([.5*np.log(r['p95_us'])+.5*np.log(r['p99_us']) for r in rows if r['arm'] in ['ac_on','hold_on'] and r['step']==s]))) for s in p['step_grid']]
  chosen=min(scores,key=lambda r:(r['score'],r['step']));env.dump(task/'selection.json',dict(created_epoch=time.time(),step=chosen['step'],scores=scores,results_sha256=env.sha(task/'dev-results.json'),calibration_sha256=env.sha(task/'calibration.json')))

def main():
 a=argparse.ArgumentParser();a.add_argument('--experiment-dir',type=Path,required=True);a.add_argument('--phase',choices=['smoke','calibrate','dev','main'],required=True);a.add_argument('--workers',type=int,default=24);args=a.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 task=args.experiment_dir.resolve();p=init(task);jobs=prepare(task,p,args.phase);rows=[]
 if args.phase=='main':assert len(jobs)==p['primary_runs']
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for future in as_completed([pool.submit(run_one,f) for f in jobs]):
   rows.append(future.result());env.dump(task/f'{args.phase}-progress.json',dict(done=len(rows),total=len(jobs),complete=sum(r['all_complete'] for r in rows),errors=[r['name'] for r in rows if not r['analysis_ok']]))
 rows.sort(key=lambda r:r['name']);env.dump(task/f'{args.phase}-results.json',rows);finish(task,p,args.phase,rows);print('FINISHED',args.phase,len(rows),flush=True)
if __name__=='__main__':main()
