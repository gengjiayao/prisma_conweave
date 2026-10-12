#!/usr/bin/env python3
"""Causal online routing development and paired held-out evaluation."""
import argparse,csv,json,os,random,shutil,sys,time,resource,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import path_rollout as env
import nsdi_tail_evaluation as old
import collective_evaluation as coll
BASE=Path('/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/nsdi-refresh-20260917-aomgdw2h')
FAMILIES=['uniform','storage','shift','bursty','alltoall','allreduce']
BASELINES=['ecmp','wecmp','drill','conga','letflow','conweave','fixed']
FITS=[0,1,2]

def init(task):
    task.mkdir(parents=True,exist_ok=True)
    for d in ['binaries','models','topologies','traces','plans','source','runs']:(task/d).mkdir(exist_ok=True)
    binary=task/'binaries/scratch/network-load-balance'
    if not binary.exists():
        shutil.copytree(BASE/'binaries',task/'binaries',dirs_exist_ok=True,symlinks=True)
        shutil.copy2(env.NS/'build-cable/scratch/network-load-balance',binary)
        shutil.copy2(env.NS/'build-cable/libns3.19-point-to-point-optimized.so',task/'binaries/libns3.19-point-to-point-optimized.so')
    env.configure(task,binary)
    for n in ['fixed.txt']+[f'rl_{f}.txt' for f in [9101,9102,9103]]:
        dst=task/'models'/n
        if not dst.exists():shutil.copy2(BASE/'paper/models'/n,dst)
    for f in (BASE/'experiment/topologies').glob('*.txt'):
        if not (task/'topologies'/f.name).exists():shutil.copy2(f,task/'topologies'/f.name)
    if not (task/'protocol.json').exists():
        p=dict(created_epoch=time.time(),method='ACK-trained per-leaf online quantile contextual bandit with drained-ACK reselection',
            state='12 bounded inputs from local queue, received remote reports, link service times, recent work and decayed local path-assignment counts; no final flow size or simulator FCT.',
            target='log(sampled source-leaf data-to-cumulative-ACK latency / explicit fixed prior delay); quantile Huber losses at .5,.95,.99; no simulator completion callback training.',
            initializer='Zero learned coefficients, explicit fixed delay prior; no offline CABLE model initialization. Learning-off control has identical initial parameters, features, exploration, routing boundaries and feedback.',
            online='Per-leaf parameter updates during the same native run. Local state resets for every independent run, persists across phases/jobs inside it. Sender ACKs are original transport packets.',
            update_delay_ns=20000,sample_interval_ns=50000,exploration=.02,reservation_weight=1.,safe_reselection='Fresh packets only, all previous sent bytes cumulatively ACKed, at least50us between decisions; no forced gap or global oracle.',
            inference='Native C++ inference; computation latency not modeled. Training update publication delayed20us and accepted at most once/20us/leaf; this is a modeled budget, not a hardware benchmark.',
            development_seeds=[92001,92002],development_families=['uniform','storage','shift'],step_grid=[.005,.02,.08],
            selection='Choose common step by equal-weight geometric P95 and P99 across ALL development conditions; require complete runs. Test never changes hyperparameters.',
            test_seeds=list(range(93001,93007)),families=FAMILIES,topologies=['symmetric','asymmetric'],fits=FITS,
            primary_runs=1152,arms=BASELINES+['cable','online_off','online'],baselines=BASELINES,
            common={**old.COMMON,'SIMULATOR_EXTRA_TIME':'2.0'},conweave=old.CW,
            stream='150ms; uniform/storage load.35; shift=50ms uniform.20 then mixed.50 then mixed_fast.35; bursty=new size mix and synchronized arrivals in three load phases(.20,.50,.35).',
            collective='Six32-rank jobs at150ms intervals; AlltoAll alternates8/64KiBperpeer, ringAllReduce1MiBperrank; new host/rank placement everyjob; network-only DAG.',
            metrics='P95/P99 of all offered flows including cold start; per-arrival-phase/per-job tails; completion and makespan; paired log ratios with10000seed-block/learner-fit bootstrap.',
            incomplete='No silent filtering. If a method leaves flows incomplete report completion and unavailable uncensored quantiles; never select only completed-flow tails.',
            binary_sha256=env.sha(binary),library_sha256=env.sha(task/'binaries/libns3.19-point-to-point-optimized.so'))
        env.dump(task/'protocol.json',p)
        for f in [Path(__file__),env.NS/'src/point-to-point/model/online-routing.h',env.NS/'src/point-to-point/model/online-routing.inc',env.NS/'src/point-to-point/model/switch-node.cc',env.NS/'src/point-to-point/model/route-learning.inc',env.NS/'scratch/network-load-balance.cc']:
            shutil.copy2(f,task/'source'/f.name)
    p=json.loads((task/'protocol.json').read_text());assert env.sha(binary)==p['binary_sha256'];return p

def traffic(task,family,seed):
    meta=task/'traces'/f'{family}_s{seed}.json'
    if meta.exists():return json.loads(meta.read_text())
    flows=[];plan=None
    if family in ['alltoall','allreduce']:
        for iteration in range(6):
            size=([8192,65536][iteration%2] if family=='alltoall' else 1048576)
            x=coll.generate(family,32,size,seed+iteration*100000);offset=len(flows)
            for q in x['flows']:
                q['id']+=offset;q['parents']=[v+offset for v in q['parents']];q['release_ns']+=iteration*150000000;q['iteration']=iteration;flows.append(q)
        plan=task/'plans'/f'{family}_s{seed}.dag'
        plan.write_text(str(len(flows))+'\n'+''.join(' '.join(map(str,[q['id'],q['src'],q['dst'],3,q['bytes'],q['release_ns'],0,len(q['parents'])]+q['parents']))+'\n' for q in flows))
        duration=.8
    elif family=='bursty':
        rng=random.Random(seed);sizes=[4096,32768,524288];weights=[.6,.3,.1];mean=sum(a*b for a,b in zip(sizes,weights))
        for phase,load in enumerate([.2,.5,.35]):
            for burst in range(100):
                count=round(128*load*1e9*.0005/(8*mean));dest=rng.randrange(128)
                for i in range(count):
                    src=rng.randrange(127);src+=src>=dest
                    dst=dest if rng.random()<.5 else rng.randrange(128)
                    if src==dst:dst=(dst+1)%128
                    ns=2000000000+phase*50000000+burst*500000+rng.randrange(20001)
                    flows.append(dict(src=src,dst=dst,bytes=rng.choices(sizes,weights)[0],release_ns=ns,iteration=phase))
        flows.sort(key=lambda q:q['release_ns']);duration=.15
    else:
        for phase in range(3):
            fam=family if family!='shift' else ['uniform','mixed','mixed_fast'][phase]
            load=.35 if family!='shift' else [.2,.5,.35][phase]
            path=old.make_trace(task,fam,seed+100000*phase,load)
            for line in path.read_text().splitlines()[1:]:
                s,d,pg,b,t=line.split();flows.append(dict(src=int(s),dst=int(d),bytes=int(b),release_ns=round(float(t)*1e9)+phase*50000000,iteration=phase))
        duration=.15
    trace=meta.with_suffix('.txt');trace.write_text(str(len(flows))+'\n'+''.join(f"{q['src']} {q['dst']} 3 {q['bytes']} {q['release_ns']/1e9:.9f}\n" for q in flows))
    x=dict(family=family,seed=seed,traffic=str(trace),traffic_sha256=env.sha(trace),duration=duration,flows=flows,plan=str(plan) if plan else None,plan_sha256=env.sha(plan) if plan else None)
    env.dump(meta,x);return x

def prepare(task,p,phase):
    if phase=='smoke':seeds=[91999];families=['uniform','allreduce'];topos=p['topologies'];specs=[('ecmp',0,0),('cable',0,0),('online_off',0,.02),('online',0,.02)]
    elif phase=='dev':
        seeds=p['development_seeds'];families=p['development_families'];topos=p['topologies']
        specs=[('fixed',0,0),('cable',0,0),('online_off',0,.02)]+[('online',0,s) for s in p['step_grid']]
    else:
        seeds=p['test_seeds'];families=p['families'];topos=p['topologies'];step=json.loads((task/'selection.json').read_text())['step']
        specs=[(a,0,0) for a in BASELINES]+[(a,f,step) for a in ['cable','online_off','online'] for f in FITS]
    jobs=[]
    for fam in families:
        for seed in seeds:
            tr=traffic(task,fam,seed)
            for topo in topos:
                for arm,fit,step in specs:
                    name=f'{phase}_{fam}_s{seed}_{topo}_{arm}_f{fit}_lr{step:g}';folder=task/'runs'/name
                    if folder.exists():jobs.append(folder);continue
                    env.trace=lambda *a,_p=Path(tr['traffic']),**kw:_p
                    tf=task/'topologies'/('leaf_spine_128_750M_sym.txt' if topo=='symmetric' else 'leaf_spine_128_100G_asym_OS2_asymmetric.txt')
                    native=arm in ['cable','fixed','online','online_off'];online=arm in ['online','online_off']
                    model=task/'models'/(f'rl_{9101+fit}.txt' if arm=='cable' else 'fixed.txt') if native else None
                    ex={**p['common'],'TOPOLOGY_FILE':str(tf),'RANDOM_SEED':'100'}
                    if tr['plan']:ex.update(COLLECTIVE_PLAN_FILE=tr['plan'],COLLECTIVE_EVENT_FILE=str(folder/'collective-events.txt'))
                    if native:
                        ex.update(RL_SWITCH_MODE='9' if online else '8',RL_PATH_RESELECT='1' if online else '0',RL_GATE_MODE='0',RL_PRESSURE_ENABLE='0',RL_SWITCH_EPSILON='0',RL_SWITCH_MARGIN_NS='0',RL_SWITCH_INTERVAL_NS='50000',RL_SWITCH_MODEL=str(model),RL_SWITCH_OUTPUT=str(folder/'transitions.csv'))
                    else:
                        ex.update(RL_SWITCH_MODE='0',CONGA_ACK_FEEDBACK='0')
                        if arm=='conweave':ex.update(p['conweave'])
                    if online:ex.update(ONLINE_ENABLE='1',ONLINE_LEARN=str(int(arm=='online')),ONLINE_STEP=str(step),ONLINE_SEED=str(9401+fit),ONLINE_EXPLORATION=str(p['exploration']),ONLINE_RESERVATION_WEIGHT=str(p['reservation_weight']),ONLINE_UPDATE_DELAY_NS=str(p['update_delay_ns']))
                    f=env.make(name,scenario='uniform',seed=seed,method='oracle' if native else arm,duration=tr['duration'],remote=int(native),feedback=int(native),period=200,quantum=1024 if native else 0,gap=0 if native else 100,extra=ex)
                    j=json.loads((f/'job.json').read_text());j.update(arm=arm,fit=fit,step=step,family=fam,seed=seed,topology_name=topo,phase=phase,traffic_metadata=str(task/'traces'/f'{fam}_s{seed}.json'),topology=str(tf),topology_sha256=env.sha(tf),model=str(model) if model else None,model_sha256=env.sha(model) if model else None,protocol_sha256=env.sha(task/'protocol.json'))
                    env.dump(f/'job.json',j);jobs.append(f)
    env.dump(task/f'{phase}-jobs.json',list(map(str,jobs)));return jobs

def audit(folder):
    j=json.loads((folder/'job.json').read_text());tr=json.loads(Path(j['traffic_metadata']).read_text())
    for k in ['binary','config','traffic','topology','model']:
        if j.get(k):assert env.sha(j[k])==j[k+'_sha256']
    n=len(tr['flows']);starts={};ends={};key_id={};values={};dependencies=0
    if tr['plan']:
        assert env.sha(tr['plan'])==tr['plan_sha256']
        for line in (folder/'collective-events.txt').read_text().splitlines():
            x=line.split();i=int(x[1]);t=int(x[2]);assert 0<=i<n
            if x[0]=='start':
                assert i not in starts;starts[i]=t;s,d,sp,dp,size=map(int,x[3:]);q=tr['flows'][i]
                assert [s,d,size]==[q['src'],q['dst'],q['bytes']];key_id[s,d,sp,dp]=i
            else:assert i in starts and i not in ends;ends[i]=t
        for i,t in starts.items():
            q=tr['flows'][i];assert t>=q['release_ns']
            for parent in q['parents']:assert parent in ends and t>=ends[parent];dependencies+=1
    else:
        from collections import defaultdict
        sp=defaultdict(lambda:10000);dp=defaultdict(lambda:100)
        for i,q in enumerate(tr['flows']):s,d=q['src'],q['dst'];key_id[s,d,sp[s],dp[d]]=i;sp[s]+=1;dp[d]+=1;starts[i]=q['release_ns']
    for line in (folder/'fct_output_file.txt').read_text().splitlines():
        x=list(map(int,line.split()));i=key_id[tuple(x[:4])];q=tr['flows'][i];assert i not in values and x[4]==q['bytes'] and abs(x[5]-starts[i])<=1 and x[6]>0
        if tr['plan']:assert x[6]==ends[i]-starts[i]
        values[i]=x[6]/1000
    row={k:j[k] for k in ['name','arm','fit','step','family','seed','topology_name','phase','traffic_sha256','model_sha256','binary_sha256','config_sha256']}
    complete=len(values)==n;row.update(native_ok=True,analysis_ok=True,all_complete=complete,offered=n,completed=len(values),completion_fraction=len(values)/n,dependency_edges=dependencies,folder=str(folder),fct_sha256=env.sha(folder/'fct_output_file.txt'))
    times=list(values.values());row.update({f'p{q}_us':float(np.percentile(times,q)) if complete else None for q in [95,99]});row['mean_us']=float(np.mean(times)) if complete else None
    row['makespan_us']=max(starts[i]/1000+v-2e6 for i,v in values.items()) if complete else None
    row['phases']={}
    for phase in sorted({q['iteration'] for q in tr['flows']}):
        ids=[i for i,q in enumerate(tr['flows']) if q['iteration']==phase];done=[i for i in ids if i in values];vv=[values[i] for i in done];full=len(done)==len(ids)
        row['phases'][str(phase)]=dict(offered=len(ids),completed=len(done),p95_us=float(np.percentile(vv,95)) if full else None,p99_us=float(np.percentile(vv,99)) if full else None,job_us=max(starts[i]/1000+values[i]-(2e6+phase*150000) for i in ids) if full and tr['plan'] else None)
    af=folder/'transitions.csv.audit.json'
    if af.exists():
        a=json.loads(af.read_text());assert a['overrides']==a['late_packet_actions']==0
        if complete:assert a['unfinished_episodes']==0 and a['terminals']+a.get('path_reselections',0)==a['transitions']==sum(a['path_assignments'])
        row['actor_audit']=a
        if j['arm'] in ['online','online_off']:
            bounds=folder/'transitions.csv.boundaries.csv'
            with bounds.open() as f:
                for b in csv.DictReader(f):
                    if int(b['previous_port']):assert int(b['cumulative_ack'])>=int(b['sent_end']) and int(b['sequence'])>=int(b['sent_end'])
            oa=json.loads((folder/'transitions.csv.online-audit.json').read_text());row['online_audit']=oa
            assert all(v['updates']==0 for v in oa['leaves']) if j['arm']=='online_off' else sum(v['updates'] for v in oa['leaves'])>0
            rows=0
            with (folder/'transitions.csv.online-updates.csv').open() as f:
                for u in csv.DictReader(f):assert int(u['update_ns'])-int(u['ack_ns'])==20000 and float(u['delta_norm'])>0;rows+=1
            assert rows==sum(v['updates'] for v in oa['leaves'])
            row['online_log_sha256']=env.sha(folder/'transitions.csv.online-updates.csv')
    return row

def run_one(folder):
    out=folder/'online-result.json'
    if out.exists():return json.loads(out.read_text())
    j=json.loads((folder/'job.json').read_text());native=False
    try:
        env.execute(folder);native=True;row=audit(folder)
    except Exception as exc:
        row={k:j[k] for k in ['name','arm','fit','step','family','seed','topology_name','phase']};row.update(native_ok=native,analysis_ok=False,all_complete=False,error=repr(exc),folder=str(folder))
    env.dump(out,row);print('AUDIT',row['name'],row['all_complete'],row.get('p99_us'),row.get('error',''),flush=True);return row

def select(task,p,rows):
    scores=[]
    for step in p['step_grid']:
        rr=[r for r in rows if r['arm']=='online' and r['step']==step];valid=all(r['all_complete'] and r['analysis_ok'] for r in rr)
        score=float(np.mean([.5*np.log(r['p95_us'])+.5*np.log(r['p99_us']) for r in rr])) if valid else None
        scores.append(dict(step=step,valid=valid,score=score))
    valid=[x for x in scores if x['valid']];assert valid,'No valid development candidate'
    chosen=min(valid,key=lambda x:(x['score'],x['step']));env.dump(task/'selection.json',dict(created_epoch=time.time(),step=chosen['step'],scores=scores,development_results_sha256=env.sha(task/'dev-results.json'),protocol_sha256=env.sha(task/'protocol.json')))

def main():
    a=argparse.ArgumentParser();a.add_argument('--experiment-dir',type=Path,required=True);a.add_argument('--phase',choices=['smoke','dev','main'],required=True);a.add_argument('--workers',type=int,default=16);args=a.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    task=args.experiment_dir.resolve();p=init(task);jobs=prepare(task,p,args.phase);rows=[]
    if args.phase=='main':assert len(jobs)==p['primary_runs']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_one,f) for f in jobs]):
            rows.append(future.result());env.dump(task/f'{args.phase}-progress.json',dict(done=len(rows),total=len(jobs),complete=sum(r['all_complete'] for r in rows),errors=[r['name'] for r in rows if not r['analysis_ok']]))
    rows.sort(key=lambda r:r['name']);env.dump(task/f'{args.phase}-results.json',rows)
    if args.phase=='dev':select(task,p,rows)
    print('FINISHED',args.phase,len(rows),sum(r['all_complete'] for r in rows),flush=True)
if __name__=='__main__':main()
