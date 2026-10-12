#!/usr/bin/env python3
"""Causal assignment/debt ablations and offline conditional-policy search."""
import argparse,json,resource,shutil,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import offline_reservation_evaluation as prior
from offline_reservation_evaluation import env,np,coll,old,FAMILIES
PREVIOUS=Path('/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/offline-reservation-20260917-fhm73sty/experiment')
INITIAL=np.array([.25,.5,.05,.1,.016,16.,.2,1.])
LOWER=np.array([0,0,0,0,0,1,.02,.25]);UPPER=np.array([2,4,2,2,.25,256,2,4.])
SCALE=np.array([.3,.5,.1,.2,.03,20,.2,1.])
FLAGS={'accounting':(0,0,0),'rotor':(1,0,0),'work':(0,1,0),'combined':(1,1,0),'conditional':(1,1,1)}
BASELINES=['ecmp','wecmp','conga','conweave']

def export(theta,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    a=np.asarray(theta);assert a.shape==(8,) and np.isfinite(a).all() and np.all(a>=LOWER) and np.all(a<=UPPER)
    path.write_text('COORDINATED_PATH_V1\n'+' '.join(format(float(x),'.17g') for x in a)+'\n');return path

def cases(phase,rnd=0):
    result=[dict(family=f,ranks=0,size=0,load=.35) for f in FAMILIES[:4]]
    if phase=='smoke':return result[:2]+[dict(family='alltoall',ranks=48,size=131072),dict(family='allreduce',ranks=32,size=1048576)]
    if phase=='train':
        n=[32,48,64,48][rnd//2];b=[8192,131072][rnd//4]
        return result+[dict(family='alltoall',ranks=n,size=b),dict(family='allreduce',ranks=n,size=3145728 if n==48 else 1048576)]
    result += [dict(family='alltoall',ranks=n,size=b) for n in [32,64] for b in [8192,65536]]
    result += [dict(family='alltoall',ranks=48,size=131072)]
    result += [dict(family='allreduce',ranks=n,size=1048576) for n in [32,64]]+[dict(family='allreduce',ranks=48,size=3145728)]
    if phase=='test':
        result += [dict(family='alltoall',ranks=80,size=b,unseen=True) for b in [32768,262144]]+[dict(family='allreduce',ranks=80,size=5242880,unseen=True)]
    return result

def init(root):
    root.mkdir(parents=True,exist_ok=True)
    for d in ['models','topologies','traces','plans','source','runs','checkpoints']:(root/d).mkdir(exist_ok=True)
    binary=root/'binaries/scratch/network-load-balance'
    if not binary.exists():
        shutil.copytree(PREVIOUS/'binaries',root/'binaries',symlinks=True)
        shutil.copy2(env.NS/'build-cable/scratch/network-load-balance',binary)
        shutil.copy2(env.NS/'build-cable/libns3.19-point-to-point-optimized.so',root/'binaries/libns3.19-point-to-point-optimized.so')
    env.configure(root,binary)
    for f in (PREVIOUS/'models').glob('*.txt'):
        if f.name in ['initial.txt','fixed.txt','rl_9101.txt','rl_9102.txt','rl_9103.txt'] and not (root/'models'/f.name).exists():shutil.copy2(f,root/'models'/f.name)
    for f in (PREVIOUS/'topologies').glob('*.txt'):
        if not (root/'topologies'/f.name).exists():shutil.copy2(f,root/'topologies'/f.name)
    selected=json.loads((PREVIOUS/'selection.json').read_text())
    for item in selected['models']['real']:
        target=root/'models'/f"previous_learned_{item['fit']}.txt"
        if not target.exists():shutil.copy2(item['path'],target)
    if not (root/'models/initial.coord').exists():export(INITIAL,root/'models/initial.coord')
    if not (root/'protocol.json').exists():
        p=dict(created_epoch=time.time(),method='Frozen state-conditional initial-path scoring, source-offset weighted stride allocation, and observed-progress debt',
            mechanisms=FLAGS,theta_initial=INITIAL.tolist(),lower=LOWER.tolist(),upper=UPPER.tolist(),direction_scale=SCALE.tolist(),
            parameters=['remote_weight','destination_reservation_weight','port_reservation_weight','port_history_weight','eligible_window_ms','burst_half_count','feedback_half_age_ms','predicted_span_scale'],
            causal_state='Counts1ms; stride counts20ms; unsent predicted-byte debt20ms. All per-destination cells indexed by visible IP and aggregated ONLY with received route prefix. Idle2ms observed sequence spans update source-leaf EWMA(alpha.125), prior64KiB, clip1KiB..1MiB; no true flow size or completion input. Data high-water sequence span subtracts debt; missing/reordered bytes can make this an underestimate. First burst uses the prior.',
            conditional='burst=decayed new-flow count over200us; gate=burst/(burst+theta5); remote weight theta0/(1+age/theta6)*(1-.75gate); reservation multiplier .25+.75gate; eligibility window theta4*gate; no family/topology label input.',
            coordination='Weighted stride among eligible paths: minimum (20ms assigned count+1+source/prefix offset phase)/min(local,received remote capacity). Stable local port rank; source-specific hashed cyclic offset. No communication or guaranteed global matching.',
            train_seeds=list(range(193301,193309)),screen_seeds=[193401,193402],validation_seeds=[193501,193502],test_seeds=list(range(193601,193605)),route_seeds=[193201,193202,193203],optimizer_seeds=[193101,193102,193103],
            rounds=8,directions=4,top=2,fits=3,checkpoints=[0,4,8],sigma=[.6]*4+[.3]*4,step=[.25]*4+[.125]*4,normalized_update_cap=.6,
            objective='For each family loss=.2logmean+.4logP95+.4logP99 (streams), or .15logmean+.25logP95+.25logP99+.35logCCT (collectives). Subtract matched ConWeave loss, aggregate equally within family; total=mean6family relative losses +.25max family relative loss +.25mean positive log(P99/previous-reserve P99). No current-test inputs.',
            training='ARS8rounds,4paired directions,top2,3fits,real/shuffled matched budgets. Parameters of conditional BOTH-mechanisms actor only; all5fixed ablations reported even if they win. Each round6families, alternating topology, fresh traffic seed. Model selection on validation alone.',
            selection='One checkpoint per reward group shared across3fits using above objective over2topologies2validation seeds12cases; checkpoint0 allowed. Test generated only after selection. No selecting different policy per test family.',
            train_cases=[dict(round=r,seed=193301+r,topology=['symmetric','asymmetric'][r%2],cases=cases('train',r)) for r in range(8)],screen_cases=cases('screen'),validation_cases=cases('val'),test_cases=cases('test'),
            common={**old.COMMON,'SIMULATOR_EXTRA_TIME':'2.0'},conweave=old.CW,old_decay_ns=1000000,old_near_ms=.016,
            topology='Same128hosts8leaf8spine,host1Gbps;8x.75Gbps symmetric vs4x1+4x.5Gbps asymmetric;6Gbps/leaf;DCQCN/IRN/PFCoff.',
            families=FAMILIES,formal_arms=BASELINES+['cable','previous_learned','reserve_only']+list(FLAGS)+['learned','shuffled'],
            training_native=2400,validation_native=960,screen_native=384,smoke_native=80,test_native=4080,
            binary_sha256=env.sha(binary),library_sha256={f.name:env.sha(f) for f in (root/'binaries').glob('*.so')},previous_selection_sha256=env.sha(PREVIOUS/'selection.json'))
        env.dump(root/'protocol.json',p)
        for f in [Path(__file__),env.NS/'src/point-to-point/model/coordinated-path.h',env.NS/'src/point-to-point/model/coordinated-path.inc',env.NS/'src/point-to-point/model/switch-node.cc',env.NS/'src/point-to-point/model/route-learning.inc',env.NS/'scratch/network-load-balance.cc']:shutil.copy2(f,root/'source'/f.name)
    p=json.loads((root/'protocol.json').read_text());assert env.sha(binary)==p['binary_sha256']
    for name,h in p['library_sha256'].items():assert env.sha(root/'binaries'/name)==h
    return p

def prepare(root,p,phase,c,seed,topo,arm,fit=0,model=None,suffix='',packet_log=False):
    name=f"{phase}_{c['family']}_p{c['ranks']}_b{c['size']}_s{seed}_{topo}_{arm}_f{fit}{suffix}";folder=root/'runs'/name
    if folder.exists():return folder
    tr=prior.traffic(root,c,seed);coord=arm in FLAGS or arm in ['learned','shuffled','train_real','train_shuffled','val_real','val_shuffled'];native=coord or arm in ['cable','previous_learned','reserve_only']
    route_model=root/'models'/(f'rl_{9101+fit}.txt' if arm=='cable' else f'previous_learned_{fit}.txt' if arm=='previous_learned' else 'initial.txt') if native else None
    coord_model=Path(model) if model else root/'models/initial.coord'
    tf=root/'topologies'/('leaf_spine_128_750M_sym.txt' if topo=='symmetric' else 'leaf_spine_128_100G_asym_OS2_asymmetric.txt')
    ex={**p['common'],'TOPOLOGY_FILE':str(tf),'ONLINE_ENABLE':'0','FAC_ENABLE':'0','COORD_ENABLE':str(int(coord)),'OFFLINE_PATH_ENABLE':str(int(arm in ['previous_learned','reserve_only']))}
    if tr['plan']:ex.update(COLLECTIVE_PLAN_FILE=tr['plan'],COLLECTIVE_EVENT_FILE=str(folder/'collective-events.txt'))
    if native:ex.update(RL_SWITCH_MODE='8',RL_PATH_RESELECT='0',RL_GATE_MODE='0',RL_PRESSURE_ENABLE='0',RL_SWITCH_EPSILON='0',RL_SWITCH_MARGIN_NS='0',RL_SWITCH_MODEL=str(route_model),RL_SWITCH_OUTPUT=str(folder/'transitions.csv'))
    else:
        ex.update(RL_SWITCH_MODE='0',CONGA_ACK_FEEDBACK='0')
        if arm=='conweave':ex.update(p['conweave'])
    if arm in ['previous_learned','reserve_only']:ex.update(OFFLINE_PATH_RESERVATIONS='1',OFFLINE_PATH_SPREAD=str(int(arm=='previous_learned')),OFFLINE_PATH_DECAY_NS=str(p['old_decay_ns']),OFFLINE_PATH_NEAR_MS=str(p['old_near_ms']),OFFLINE_PATH_SEED=str(98701+fit))
    if coord:
        bal,work,cond=FLAGS.get(arm,(1,1,1));ex.update(COORD_BALANCE=str(bal),COORD_WORK=str(work),COORD_CONDITIONAL=str(cond),COORD_MODEL=str(coord_model),COORD_SEED=str(p['route_seeds'][fit]),COORD_PACKET_LOG=str(int(packet_log)))
    env.trace=lambda *a,_p=Path(tr['traffic']),**kw:_p
    f=env.make(name,scenario='uniform',seed=seed,method='oracle' if native else arm,remote=int(native),feedback=int(native),period=200,quantum=1024 if native else 0,gap=0 if native else 100,extra=ex)
    j=json.loads((f/'job.json').read_text());j.update(arm=arm,fit=fit,family=c['family'],ranks=c['ranks'],size=c['size'],seed=seed,topology_name=topo,phase=phase,unseen=c.get('unseen',False),topology=str(tf),topology_sha256=env.sha(tf),model=str(route_model) if route_model else None,model_sha256=env.sha(route_model) if route_model else None,coord_model=str(coord_model) if coord else None,coord_model_sha256=env.sha(coord_model) if coord else None,protocol_sha256=env.sha(root/'protocol.json'))
    if tr['plan']:j.update(plan=tr['plan'],plan_sha256=tr['plan_sha256'],plan_metadata=str(Path(tr['plan']).with_suffix('.json')))
    if phase=='test':j['selection_sha256']=env.sha(root/'selection.json');assert Path(tr['traffic']).stat().st_mtime>=json.loads((root/'selection.json').read_text())['created_epoch']
    env.dump(f/'job.json',j);return f

def run_one(folder):
    r=prior.run_one(folder);j=json.loads((folder/'job.json').read_text())
    if j['coord_model']:
        assert env.sha(j['coord_model'])==j['coord_model_sha256'];a=json.loads((folder/'transitions.csv.coordinated-audit.json').read_text());assert a['parameter_updates']==0 and sum(x['decisions'] for x in a['leaves'])==r['actor_audit']['terminals'];r['coordinated_audit']=a;env.dump(folder/'result.json',r)
    return r

def execute(root,jobs,workers,label):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run_one,x) for x in jobs]):rows.append(f.result());env.dump(root/f'{label}-progress.json',dict(done=len(rows),total=len(jobs)))
    rows.sort(key=lambda r:r['name']);env.dump(root/f'{label}-results.json',rows);return rows

def identity(r):return (r['seed'],r['topology_name'],r['family'],r['ranks'],r['size'])
def scalar(r):
    v=np.log(np.array([r['mean_us'],r['p95_us'],r['p99_us']])/1000)
    return float(v@[.2,.4,.4]) if r['makespan_us'] is None else float(v@[.15,.25,.25]+.35*np.log(r['makespan_us']/1000))
def objective(rows,baselines):
    refs={(r['arm'],identity(r)):r for r in baselines};groups=defaultdict(list);guard=defaultdict(list)
    for r in rows:
        assert r['all_complete'];k=identity(r);groups[r['family']].append(scalar(r)-scalar(refs['conweave',k]));guard[r['family']].append(max(0.,np.log(r['p99_us']/refs['reserve_only',k]['p99_us'])))
    v=[np.mean(groups[f]) for f in FAMILIES];return float(np.mean(v)+.25*max(v)+.25*np.mean([np.mean(guard[f]) for f in FAMILIES]))

def train(root,p,workers):
    theta={(a,f):INITIAL.copy() for a in ['real','shuffled'] for f in range(3)}
    for a,f in theta:export(theta[a,f],root/'checkpoints'/f'{a}_f{f}_r0.coord')
    for rnd in range(8):
        record=root/f'train-round{rnd}.json'
        if record.exists():
            for u in json.loads(record.read_text())['updates']:theta[u['arm'],u['fit']]=np.array(u['after'])
            continue
        case=p['train_cases'][rnd];refs=[prepare(root,p,'train',c,case['seed'],case['topology'],a) for c in case['cases'] for a in ['conweave','reserve_only']]
        baselines=execute(root,refs,workers,f'train-r{rnd}-references');jobs=[];owners={};directions={};permutations={}
        for fit in range(3):
            directions[fit]=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000).normal(size=(4,8))
            permutations[fit]=np.random.default_rng(p['optimizer_seeds'][fit]+rnd*10000+500000).permutation(8)
        for arm in ['real','shuffled']:
            for fit in range(3):
                for d in range(4):
                    for sign in [-1,1]:
                        proposal=np.clip(theta[arm,fit]+sign*p['sigma'][rnd]*directions[fit][d]*SCALE,LOWER,UPPER)
                        model=export(proposal,root/'models'/f'train_{arm}_f{fit}_r{rnd}_d{d}_z{sign}.coord')
                        for c in case['cases']:
                            f=prepare(root,p,'train',c,case['seed'],case['topology'],'train_'+arm,fit,model,f'_r{rnd}_d{d}_z{sign}',packet_log=(rnd==7 and fit==0 and d==0 and sign==1))
                            jobs.append(f);owners[f.name]=(arm,fit,d,sign)
        rows=execute(root,jobs,workers,f'train-r{rnd}');groups=defaultdict(list)
        for r in rows:groups[owners[r['name']]].append(r)
        updates=[]
        for arm in ['real','shuffled']:
            for fit in range(3):
                rewards=np.array([-objective(groups[arm,fit,d,s],baselines) for d in range(4) for s in [-1,1]])
                permutation=np.arange(8) if arm=='real' else permutations[fit];used=rewards[permutation].reshape(4,2)
                top=np.argsort(-used.max(axis=1),kind='stable')[:2];std=max(.001,float(used[top].std()))
                delta=p['step'][rnd]*np.mean((used[top,1]-used[top,0])[:,None]*directions[fit][top],axis=0)/std
                delta*=min(1.,p['normalized_update_cap']/max(1e-12,float(np.linalg.norm(delta))))
                before=theta[arm,fit].copy();theta[arm,fit]=np.clip(before+delta*SCALE,LOWER,UPPER)
                model=export(theta[arm,fit],root/'checkpoints'/f'{arm}_f{fit}_r{rnd+1}.coord')
                updates.append(dict(arm=arm,fit=fit,before=before.tolist(),after=theta[arm,fit].tolist(),raw_rewards=rewards.tolist(),permutation=permutation.tolist(),directions=directions[fit].tolist(),normalized_delta=delta.tolist(),model=str(model),model_sha256=env.sha(model)))
        env.dump(record,dict(round=rnd,updates=updates,results_sha256=env.sha(root/f'train-r{rnd}-results.json')));print('TRAIN_ROUND_COMPLETE',rnd+1,flush=True)
    env.dump(root/'training-complete.json',dict(rounds=8,created_epoch=time.time(),runs=2400))

def validate(root,p,workers):
    assert (root/'training-complete.json').exists();jobs=[]
    for seed in p['validation_seeds']:
        for topo in ['symmetric','asymmetric']:
            for c in p['validation_cases']:
                jobs.extend(prepare(root,p,'val',c,seed,topo,a) for a in ['conweave','reserve_only'])
                for arm in ['real','shuffled']:
                    for rnd in p['checkpoints']:
                        for fit in range(3):jobs.append(prepare(root,p,'val',c,seed,topo,'val_'+arm,fit,root/'checkpoints'/f'{arm}_f{fit}_r{rnd}.coord',f'_r{rnd}'))
    assert len(jobs)==p['validation_native'];rows=execute(root,jobs,workers,'validation');refs=[r for r in rows if r['arm'] in ['conweave','reserve_only']];scores=[]
    for arm in ['real','shuffled']:
        for rnd in p['checkpoints']:scores.append(dict(arm=arm,round=rnd,loss=objective([r for r in rows if r['arm']=='val_'+arm and r['name'].endswith(f'_r{rnd}')],refs)))
    chosen={arm:min([s for s in scores if s['arm']==arm],key=lambda s:(s['loss'],s['round'])) for arm in ['real','shuffled']}
    models={arm:[dict(fit=f,path=str(root/'checkpoints'/f"{arm}_f{f}_r{chosen[arm]['round']}.coord"),sha256=env.sha(root/'checkpoints'/f"{arm}_f{f}_r{chosen[arm]['round']}.coord")) for f in range(3)] for arm in chosen}
    env.dump(root/'selection.json',dict(created_epoch=time.time(),selected=chosen,scores=scores,models=models,protocol_sha256=env.sha(root/'protocol.json'),validation_results_sha256=env.sha(root/'validation-results.json')));print('SELECTED',chosen,flush=True)

def evaluate(root,p,workers,phase):
    if phase=='smoke':seeds=[193299];cs=cases('smoke');specs=[(a,0,None) for a in list(FLAGS)+['ecmp','conga','conweave','reserve_only','previous_learned']]
    elif phase=='screen':seeds=p['screen_seeds'];cs=p['screen_cases'];specs=[(a,0,None) for a in list(FLAGS)+['conweave','reserve_only','previous_learned']]
    else:
        sel=json.loads((root/'selection.json').read_text());seeds=p['test_seeds'];cs=p['test_cases']
        specs=[(a,0,None) for a in BASELINES]+[(a,f,None) for a in ['cable','previous_learned','reserve_only']+list(FLAGS) for f in range(3)]
        specs += [(a,f,Path(sel['models'][kind][f]['path'])) for a,kind in [('learned','real'),('shuffled','shuffled')] for f in range(3)]
    jobs=[prepare(root,p,phase,c,seed,t,a,f,m,packet_log=phase=='smoke' or (phase=='test' and seed==seeds[0] and f==0 and a in list(FLAGS)+['learned','shuffled'] and (c['ranks']==0 or (c['family'],c['ranks'],c['size']) in [('alltoall',48,131072),('allreduce',32,1048576)]))) for seed in seeds for t in ['symmetric','asymmetric'] for c in cs for a,f,m in specs]
    assert len(jobs)==p[phase+'_native'],len(jobs)
    return execute(root,jobs,workers,phase)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stage',choices=['smoke','screen','train','validate','test','all'],required=True);ap.add_argument('--workers',type=int,default=20);a=ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=a.root.resolve();p=init(root)
    for stage in ['smoke','screen','train','validate','test']:
        if a.stage not in [stage,'all']:continue
        if stage=='train':train(root,p,a.workers)
        elif stage=='validate':
            if not (root/'selection.json').exists():validate(root,p,a.workers)
        else:evaluate(root,p,a.workers,stage)
if __name__=='__main__':main()
