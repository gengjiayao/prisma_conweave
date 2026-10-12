#!/usr/bin/env python3
"""Offline policy search and held-out tests of destination path reservations."""
import argparse,json,os,random,resource,shutil,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import collective_evaluation as coll
from collective_evaluation import env,np
import nsdi_tail_evaluation as old

BASE=Path('/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/nsdi-refresh-20260917-aomgdw2h')
FAMILIES=['uniform','storage','mixed_fast','bursty','alltoall','allreduce']
FEATURES=[1,2,3,5,7,8,22,23]
INITIAL=np.array([.25,0,0,0,0,.1,.05,.5])
LOWER=np.array([0,-1,-1,-1,0,0,0,0.]);UPPER=np.array([2,1,1,1,2,2,2,4.])
DIRECTION_SCALE=np.array([1,1,1,1,.5,.5,.2,1.])
NEW_ARMS=['mechanism','reserve_only','spread_only','learned','shuffled']

def export(theta,path):
    theta=np.asarray(theta);assert theta.shape==(8,) and np.isfinite(theta).all()
    w=np.zeros((24,32));b=np.zeros(32);v=np.zeros((32,2));c=np.zeros(2);h=0
    for feature,weight in [(0,1.)]+list(zip(FEATURES,theta)):
        w[feature,h]=1;v[h,1]=weight;h+=1
        if feature==5:w[feature,h]=-1;v[h,1]=-weight;h+=1
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('ROUTE_DQN_V1 24 32\n'+'\n'.join(' '.join(format(float(x),'.17g') for x in a.ravel()) for a in [w,b,v,c])+'\n')
    return str(path)

def scenarios(phase,rnd=0):
    cases=[dict(family=f,ranks=0,size=0,load=.35) for f in FAMILIES[:4]]
    if phase=='train':
        ranks=[32,64][(rnd//2)%2];size=[8192,65536][(rnd//4)%2]
        cases += [dict(family='alltoall',ranks=ranks,size=size),dict(family='allreduce',ranks=ranks,size=1048576)]
    elif phase=='smoke':cases=[cases[0],cases[1],dict(family='alltoall',ranks=32,size=8192),dict(family='allreduce',ranks=32,size=1048576)]
    else:
        cases += [dict(family='alltoall',ranks=p,size=b) for p in [32,64] for b in [8192,65536]]
        cases += [dict(family='allreduce',ranks=p,size=1048576) for p in [32,64]]
        if phase=='test':
            cases += [dict(family='alltoall',ranks=48,size=b,unseen=True) for b in [16384,131072]]
            cases += [dict(family='allreduce',ranks=48,size=3145728,unseen=True)]
    return cases

def init(root):
    root.mkdir(parents=True,exist_ok=True)
    for d in ['models','topologies','traces','plans','source','runs','checkpoints']:(root/d).mkdir(exist_ok=True)
    binary=root/'binaries/scratch/network-load-balance'
    if not binary.exists():
        shutil.copytree(BASE/'binaries',root/'binaries',symlinks=True)
        shutil.copy2(env.NS/'build-cable/scratch/network-load-balance',binary)
        for name in ['point-to-point','network']:shutil.copy2(env.NS/f'build-cable/libns3.19-{name}-optimized.so',root/f'binaries/libns3.19-{name}-optimized.so')
    env.configure(root,binary)
    for name in ['fixed.txt']+[f'rl_{fit}.txt' for fit in [9101,9102,9103]]:
        if not (root/'models'/name).exists():shutil.copy2(BASE/'paper/models'/name,root/'models'/name)
    for f in (BASE/'experiment/topologies').glob('*.txt'):
        if not (root/'topologies'/f.name).exists():shutil.copy2(f,root/'topologies'/f.name)
    if not (root/'models/initial.txt').exists():export(INITIAL,root/'models/initial.txt')
    for fit in range(3):
        target=root/'models'/f'deleted_{fit}.txt'
        if not target.exists():
            vals=np.array(list(map(float,(root/'models'/f'rl_{9101+fit}.txt').read_text().split()[3:])))
            w=vals[:768].reshape(24,32);w[3]+=w[1];w[[1,5,7]]=0
            target.write_text('ROUTE_DQN_V1 24 32\n'+' '.join(format(x,'.17g') for x in vals)+'\n')
    if not (root/'protocol.json').exists():
        p=dict(created_epoch=time.time(),method='Frozen initial-flow actor with source-leaf destination-prefix reservations and capacity-weighted near-best spreading',
          reservation='Add one count per locally assigned cross-rack flow; exponential1ms decay; port count times local1KB serialization in x22, destination-prefix/path count times received remote1KB serialization in x23. Group only by received prefix; unknown fallback is visible destination/32. No future size/completion input.',
          spread='Choose among score<=minimum+.016ms with probability proportional to local link capacity. Stateless SplitMix64 of observed flow key XOR(node<<32) XOR seed; same per-flow random numbers for matched controls.',
          score='Local delay coefficient fixed1; eight optimized coefficients for x1,x2,x3,x5,x7,x8,x22,x23. Destination history, port history and reservation penalties constrained nonnegative; signed trend/explicit serialization allowed.',
          theta_initial=INITIAL.tolist(),features=FEATURES,lower=LOWER.tolist(),upper=UPPER.tolist(),direction_scale=DIRECTION_SCALE.tolist(),
          training='Offline ARS paired parameter perturbations;3fits;real and reward-shuffled arms share initial state, direction seeds, budgets, and selection rule;8rounds,4directions,top2, paired common traffic and route randomness. No parameter updates in a simulation.',
          rounds=8,directions=4,top=2,sigma=[.12]*4+[.06]*4,step=[.15]*4+[.075]*4,update_cap=.35,
          training_seeds=list(range(98401,98409)),validation_seeds=[98501,98502],test_seeds=[98601,98602,98603,98604],
          optimizer_seeds=[98101,98102,98103],route_seeds=[98701,98702,98703],checkpoints=[0,4,8],
          reward='Negative equal-family mean loss. Stream loss=.2log(mean ms)+.4log(P95 ms)+.4log(P99 ms). Collective loss=.15log(mean ms)+.25log(P95 ms)+.25log(P99 ms)+.35log(CCT ms). No censored-flow scoring.',
          train_cases=[dict(round=r,topology=['symmetric','asymmetric'][r%2],seed=98401+r,cases=scenarios('train',r)) for r in range(8)],
          validation_cases=scenarios('val'),test_cases=scenarios('test'),families=FAMILIES,
          selection='One checkpoint shared by3fits per arm; minimize the predeclared objective over both topologies, both validation seeds and all cases, first averaging within each family. Checkpoint0 permitted. Freeze before test generation.',
          test_rule='No test tuning; report all seeds,fits,conditions,losses; test traffic created only after selection. Report familiar sizes/ranks and held-out48rank16/128KiB Alltoall and48rank3MiB Allreduce separately.',
          baselines=['ecmp','wecmp','conga','conweave','fixed'],controls=['cable','deleted','mechanism','reserve_only','spread_only','learned','shuffled'],fits=3,
          training_native=2304,validation_native=720,test_native=2704,smoke_native=72,
          common={**old.COMMON,'SIMULATOR_EXTRA_TIME':'2.0'},conweave=old.CW,
          traffic='50ms streams. Uniform/storage/mixed_fast load.35; bursty .5ms periods,0-20us jitter,50% hotspot,4/32/512KiB(.6/.3/.1). Collectives one real dependency DAG; no GPU compute.',
          topology='128hosts8leaf8spine;host1Gbps;symmetric8x.75Gbps or asymmetric4x1+4x.5Gbps;both6Gbps/leaf;DCQCN,IRN,PFCoff.',
          decay_ns=1000000,near_ms=.016,binary_sha256=env.sha(binary),
          library_sha256={f.name:env.sha(f) for f in (root/'binaries').glob('*.so')})
        env.dump(root/'protocol.json',p)
        for f in [Path(__file__),env.NS/'src/point-to-point/model/offline-path.h',env.NS/'src/point-to-point/model/offline-path.inc',env.NS/'src/point-to-point/model/route-learning.h',env.NS/'src/point-to-point/model/route-learning.inc',env.NS/'src/point-to-point/model/switch-node.cc',env.NS/'scratch/network-load-balance.cc']:
            shutil.copy2(f,root/'source'/f.name)
    p=json.loads((root/'protocol.json').read_text());assert env.sha(binary)==p['binary_sha256']
    for name,h in p['library_sha256'].items():assert env.sha(root/'binaries'/name)==h
    return p

def traffic(root,c,seed):
    fam=c['family'];name=f"{fam}_p{c['ranks']}_b{c['size']}_s{seed}";meta=root/'traces'/f'{name}.json'
    if meta.exists():return json.loads(meta.read_text())
    if fam in ['alltoall','allreduce']:
        x=coll.write_plan(root,fam,c['ranks'],c['size'],seed);env.dump(meta,x);return x
    if fam!='bursty':path=old.make_trace(root,fam,seed,c.get('load',.35))
    else:
        rng=random.Random(seed);sizes=[4096,32768,524288];weights=[.6,.3,.1];mean=sum(a*b for a,b in zip(sizes,weights));flows=[]
        for burst in range(100):
            count=round(128*.35*1e9*.0005/(8*mean));dest=rng.randrange(128)
            for _ in range(count):
                src=rng.randrange(127);src+=src>=dest;dst=dest if rng.random()<.5 else rng.randrange(128)
                if src==dst:dst=(dst+1)%128
                flows.append((2000000000+burst*500000+rng.randrange(20001),src,dst,rng.choices(sizes,weights)[0]))
        flows.sort();path=root/'traces'/f'{name}.txt'
        path.write_text(str(len(flows))+'\n'+''.join(f'{s} {d} 3 {b} {ns/1e9:.9f}\n' for ns,s,d,b in flows))
    x=dict(traffic=str(path),traffic_sha256=env.sha(path),plan=None,seed=seed,family=fam);env.dump(meta,x);return x

def prepare_one(root,p,phase,c,seed,topo,arm,fit,model=None,suffix=''):
    name=f"{phase}_{c['family']}_p{c['ranks']}_b{c['size']}_s{seed}_{topo}_{arm}_f{fit}{suffix}";folder=root/'runs'/name
    if folder.exists():return folder
    tr=traffic(root,c,seed);new=arm in NEW_ARMS or arm in ['train_real','train_shuffled','val_real','val_shuffled'];native=new or arm in ['cable','fixed','deleted']
    if model is None and native:model=root/'models'/(f'rl_{9101+fit}.txt' if arm=='cable' else f'deleted_{fit}.txt' if arm=='deleted' else 'initial.txt' if new else 'fixed.txt')
    tf=root/'topologies'/('leaf_spine_128_750M_sym.txt' if topo=='symmetric' else 'leaf_spine_128_100G_asym_OS2_asymmetric.txt')
    ex={**p['common'],'TOPOLOGY_FILE':str(tf),'ONLINE_ENABLE':'0','FAC_ENABLE':'0','OFFLINE_PATH_ENABLE':str(int(new))}
    if tr['plan']:ex.update(COLLECTIVE_PLAN_FILE=tr['plan'],COLLECTIVE_EVENT_FILE=str(folder/'collective-events.txt'))
    if native:
        ex.update(RL_SWITCH_MODE='8',RL_PATH_RESELECT='0',RL_GATE_MODE='0',RL_PRESSURE_ENABLE='0',RL_SWITCH_EPSILON='0',RL_SWITCH_MARGIN_NS='0',RL_SWITCH_MODEL=str(model),RL_SWITCH_OUTPUT=str(folder/'transitions.csv'))
    else:
        ex.update(RL_SWITCH_MODE='0',CONGA_ACK_FEEDBACK='0')
        if arm=='conweave':ex.update(p['conweave'])
    if new:ex.update(OFFLINE_PATH_RESERVATIONS=str(int(arm!='spread_only')),OFFLINE_PATH_SPREAD=str(int(arm!='reserve_only')),OFFLINE_PATH_DECAY_NS=str(p['decay_ns']),OFFLINE_PATH_NEAR_MS=str(p['near_ms']),OFFLINE_PATH_SEED=str(p['route_seeds'][fit]))
    env.trace=lambda *a,_p=Path(tr['traffic']),**kw:_p
    f=env.make(name,scenario='uniform',seed=seed,method='oracle' if native else arm,remote=int(native),feedback=int(native),period=200,quantum=1024 if native else 0,gap=0 if native else 100,extra=ex)
    j=json.loads((f/'job.json').read_text());j.update(arm=arm,fit=fit,family=c['family'],ranks=c['ranks'],size=c['size'],seed=seed,topology_name=topo,phase=phase,unseen=c.get('unseen',False),topology=str(tf),topology_sha256=env.sha(tf),model=str(model) if model else None,model_sha256=env.sha(model) if model else None,protocol_sha256=env.sha(root/'protocol.json'))
    if tr['plan']:j.update(plan=tr['plan'],plan_sha256=tr['plan_sha256'],plan_metadata=str(Path(tr['plan']).with_suffix('.json')))
    if phase=='test':
        j['selection_sha256']=env.sha(root/'selection.json');assert Path(tr['traffic']).stat().st_mtime>=json.loads((root/'selection.json').read_text())['created_epoch']
        if (root/'protocol-amendment.json').exists():j['protocol_amendment_sha256']=env.sha(root/'protocol-amendment.json')
    env.dump(f/'job.json',j);return f

def run_one(folder):
    out=folder/'result.json'
    if out.exists():return json.loads(out.read_text())
    j=json.loads((folder/'job.json').read_text())
    if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['state']!='complete':env.execute(folder)
    for k in ['binary','config','traffic','topology','model']:
        if j.get(k):assert env.sha(j[k])==j[k+'_sha256'],k
    if j.get('plan'):r=coll.audit(folder)
    else:
        summary,flows=env.load(folder);assert summary['completed']==summary['offered']
        values=np.array([f['fct_us'] for f in flows.values()]);r={k:j[k] for k in ['name','arm','fit','family','ranks','size','seed','topology_name','phase']}
        r.update(all_complete=True,analysis_ok=True,offered=len(values),completed=len(values),p95_us=float(np.percentile(values,95)),p99_us=float(np.percentile(values,99)),mean_us=float(values.mean()),makespan_us=None,folder=str(folder))
    r['unseen']=j['unseen'];r['wall_seconds']=json.loads((folder/'status.json').read_text())['elapsed_seconds']
    if (folder/'transitions.csv.audit.json').exists():
        audit=json.loads((folder/'transitions.csv.audit.json').read_text());assert audit['unfinished_episodes']==audit.get('path_changes',0)==audit.get('path_reselections',0)==0
        assert audit['transitions']==audit['terminals']==sum(audit['path_assignments']);r['actor_audit']=audit
    if j['extra']['OFFLINE_PATH_ENABLE']=='1':
        a=json.loads((folder/'transitions.csv.offline-audit.json').read_text());assert a['parameter_updates']==0
        assert sum(x['decisions'] for x in a['leaves'])==r['actor_audit']['terminals'];r['offline_audit']=a
    assert r['all_complete'] and r['analysis_ok'];env.dump(out,r);return r

def execute(root,jobs,workers,label):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run_one,x) for x in jobs]):
            rows.append(f.result());env.dump(root/f'{label}-progress.json',dict(done=len(rows),total=len(jobs)))
    rows.sort(key=lambda r:r['name']);env.dump(root/f'{label}-results.json',rows);return rows

def loss(rows):
    groups=defaultdict(list)
    for r in rows:
        assert r['all_complete'];v=np.log(np.array([r['mean_us'],r['p95_us'],r['p99_us']])/1000)
        value=float(np.dot(v,[.2,.4,.4])) if r['makespan_us'] is None else float(np.dot(v,[.15,.25,.25])+.35*np.log(r['makespan_us']/1000))
        groups[r['family']].append(value)
    return float(np.mean([np.mean(v) for v in groups.values()]))

def train(root,p,workers):
    theta={(a,f):INITIAL.copy() for a in ['real','shuffled'] for f in range(3)}
    for a,f in theta:export(theta[a,f],root/'checkpoints'/f'{a}_f{f}_r0.txt')
    for rnd in range(8):
        record=root/f'train-round{rnd}.json'
        if record.exists():
            for u in json.loads(record.read_text())['updates']:theta[u['arm'],u['fit']]=np.array(u['after'])
            continue
        case=p['train_cases'][rnd];jobs=[];owners={};directions={};permutations={}
        for fit in range(3):
            rng=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000);directions[fit]=rng.normal(size=(4,8))*DIRECTION_SCALE
            permutations[fit]=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000+500000).permutation(8)
        for arm in ['real','shuffled']:
            for fit in range(3):
                for d in range(4):
                    for sign in [-1,1]:
                        proposed=np.clip(theta[arm,fit]+sign*p['sigma'][rnd]*directions[fit][d],LOWER,UPPER)
                        model=export(proposed,root/'models'/f'train_{arm}_f{fit}_r{rnd}_d{d}_z{sign}.txt')
                        for c in case['cases']:
                            folder=prepare_one(root,p,'train',c,case['seed'],case['topology'],'train_'+arm,fit,model,f'_r{rnd}_d{d}_z{sign}')
                            jobs.append(folder);owners[folder.name]=(arm,fit,d,sign)
        rows=execute(root,jobs,workers,f'train-r{rnd}');groups=defaultdict(list)
        for r in rows:groups[owners[r['name']]].append(r)
        updates=[]
        for arm in ['real','shuffled']:
            for fit in range(3):
                rewards=np.array([-loss(groups[arm,fit,d,s]) for d in range(4) for s in [-1,1]])
                permutation=np.arange(8) if arm=='real' else permutations[fit];used=rewards[permutation].reshape(4,2)
                top=np.argsort(-used.max(axis=1),kind='stable')[:2];std=max(.001,float(used[top].std()))
                delta=p['step'][rnd]*np.mean((used[top,1]-used[top,0])[:,None]*directions[fit][top],axis=0)/std
                delta*=min(1.,p['update_cap']/max(1e-12,float(np.linalg.norm(delta))))
                before=theta[arm,fit].copy();theta[arm,fit]=np.clip(before+delta,LOWER,UPPER)
                model=export(theta[arm,fit],root/'checkpoints'/f'{arm}_f{fit}_r{rnd+1}.txt')
                updates.append(dict(arm=arm,fit=fit,before=before.tolist(),after=theta[arm,fit].tolist(),raw_rewards=rewards.tolist(),permutation=permutation.tolist(),directions=directions[fit].tolist(),delta=delta.tolist(),model=model,model_sha256=env.sha(model)))
        env.dump(record,dict(round=rnd,updates=updates,results_sha256=env.sha(root/f'train-r{rnd}-results.json')))
        print('TRAIN_ROUND_COMPLETE',rnd+1,flush=True)
    env.dump(root/'training-complete.json',dict(rounds=8,created_epoch=time.time(),runs=2304))

def validate(root,p,workers):
    assert (root/'training-complete.json').exists();jobs=[]
    for arm in ['real','shuffled']:
        for rnd in p['checkpoints']:
            for fit in range(3):
                model=root/'checkpoints'/f'{arm}_f{fit}_r{rnd}.txt'
                for seed in p['validation_seeds']:
                    for topo in ['symmetric','asymmetric']:
                        for c in p['validation_cases']:jobs.append(prepare_one(root,p,'val',c,seed,topo,'val_'+arm,fit,model,f'_r{rnd}'))
    assert len(jobs)==720;rows=execute(root,jobs,workers,'validation');scores=[]
    for arm in ['real','shuffled']:
        for rnd in p['checkpoints']:
            selected=[r for r in rows if r['arm']=='val_'+arm and r['name'].endswith(f'_r{rnd}')]
            scores.append(dict(arm=arm,round=rnd,loss=loss(selected)))
    chosen={arm:min([s for s in scores if s['arm']==arm],key=lambda s:(s['loss'],s['round'])) for arm in ['real','shuffled']}
    models={arm:[dict(fit=f,path=str(root/'checkpoints'/f"{arm}_f{f}_r{chosen[arm]['round']}.txt"),sha256=env.sha(root/'checkpoints'/f"{arm}_f{f}_r{chosen[arm]['round']}.txt")) for f in range(3)] for arm in chosen}
    env.dump(root/'selection.json',dict(created_epoch=time.time(),selected=chosen,scores=scores,models=models,protocol_sha256=env.sha(root/'protocol.json'),validation_results_sha256=env.sha(root/'validation-results.json')))
    print('SELECTED',chosen,flush=True)

def evaluate(root,p,workers,phase):
    if phase=='smoke':specs=[(a,0,None) for a in ['ecmp','conga','conweave','cable','deleted','fixed','mechanism','reserve_only','spread_only']];seeds=[98399];cases=scenarios('smoke')
    else:
        frozen=json.loads((root/'selection.json').read_text());seeds=p['test_seeds'];cases=p['test_cases']
        if (root/'protocol-amendment.json').exists():
            amendment=json.loads((root/'protocol-amendment.json').read_text());assert amendment['base_protocol_sha256']==env.sha(root/'protocol.json');cases=amendment['test_cases']
        specs=[(a,0,None) for a in p['baselines']]+[(a,f,None) for a in ['cable','deleted','mechanism','reserve_only','spread_only'] for f in range(3)]
        specs += [(a,f,root/'checkpoints'/f"{kind}_f{f}_r{frozen['selected'][kind]['round']}.txt") for a,kind in [('learned','real'),('shuffled','shuffled')] for f in range(3)]
    jobs=[prepare_one(root,p,phase,c,seed,t,a,f,m) for seed in seeds for t in ['symmetric','asymmetric'] for c in cases for a,f,m in specs]
    assert len(jobs)==p['smoke_native' if phase=='smoke' else 'test_native'],len(jobs)
    return execute(root,jobs,workers,phase)

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--stage',choices=['smoke','train','validate','test','all'],required=True);a.add_argument('--workers',type=int,default=16);args=a.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=args.root.resolve();p=init(root)
    for stage in ['smoke','train','validate','test']:
        if args.stage not in [stage,'all']:continue
        if stage=='train':train(root,p,args.workers)
        elif stage=='validate':
            if not (root/'selection.json').exists():validate(root,p,args.workers)
        else:evaluate(root,p,args.workers,stage)
if __name__=='__main__':main()
