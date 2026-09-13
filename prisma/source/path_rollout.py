"""Native rollout runtime shared by the from-zero and continuation trainers.

Configure an isolated output directory before creating jobs. No historical
experiment modules or machine-specific source paths are imported.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NS = ROOT / "conweave-ns3-main"
TASK = None
BINARY = None
PATH_BINARY = None
BASE = None
METHODS = dict(ecmp=0, wecmp=1, drill=2, conga=3, letflow=6, conweave=9, oracle=12)
RULES = dict(dqn=6)


def configure(experiment_dir, binary=None):
    global TASK, BINARY, PATH_BINARY, BASE
    TASK = Path(experiment_dir).resolve()
    TASK.mkdir(parents=True, exist_ok=True)
    BINARY = Path(binary).resolve() if binary else TASK / "binaries/frozen/scratch/network-load-balance"
    PATH_BINARY = BINARY
    BASE = (NS / "config/path_learning_base.txt").read_text()


def export_weights(weights, path):
    # Preserve the float32 fixed-control export used by the original campaigns.
    params = [np.zeros((24, 32), dtype="f"), np.zeros(32, dtype="f"),
              np.zeros((32, 2), dtype="f"), np.zeros(2, dtype="f")]
    h = 0
    for i, weight in enumerate(weights):
        params[0][i, h] = 1
        params[2][h, 1] = weight
        h += 1
        if i in {5, 6, 11, 15}:
            params[0][i, h] = -1
            params[2][h, 1] = -weight
            h += 1
    assert h <= 32
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **{f"p{i}": value for i, value in enumerate(params)},
             mask=False, gamma=1., step=0, transport=True)
    path.write_text("ROUTE_DQN_V1 24 32\n" + "\n".join(
        " ".join(format(float(value), ".12g") for value in param.ravel())
        for param in params) + "\n")
    return str(path)



def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace(scenario, seed, hosts=128, duration=.05):
    path = TASK/'traces'/f'{scenario}_n{hosts}_s{seed}_t{duration}.txt'
    if path.exists(): return path
    if scenario=='storage':
        path.parent.mkdir(parents=True,exist_ok=True)
        cdf=NS/'traffic_gen/AliStorage2019.txt'
        command=['/usr/bin/python3',str(NS/'traffic_gen/traffic_gen.py'),'-c',str(cdf),'-n',str(hosts),
                 '-l','0.35','-b','1G','-t',str(duration),'-o',str(path),'--seed',str(seed)]
        subprocess.run(command,cwd=NS,check=True,stdout=subprocess.DEVNULL)
        lines=path.read_text().splitlines()
        dump(path.with_suffix('.json'),dict(seed=seed,scenario=scenario,hosts=hosts,duration=duration,
            flows=int(lines[0]),foreground_count=int(lines[0]),sha256=sha(path),generator_command=command,
            generator_sha256=sha(NS/'traffic_gen/traffic_gen.py'),cdf_sha256=sha(cdf)))
        return path
    rng = random.Random(seed); flows = []
    sizes, weights = [2000,8000,64000,256000], [.5,.35,.1,.05]
    # Free routing for every flow; random hotspot dwell and destination changes.
    if scenario in ['mixed','mixed_fast']:
        sizes,weights=[2000,8000,64000,1048576],[.5,.35,.13,.02]
        mean=sum(s*w for s,w in zip(sizes,weights)); rate=hosts*.30e9/(mean*8)
        phases=[];at=0.
        while at<duration:
            phases.append((at,rng.randrange(hosts//16),rng.choice([0.,.25,.5])))
            at+=rng.uniform(.0002,.001) if scenario=='mixed_fast' else rng.uniform(.001,.005)
        at=0.;phase=0
        while True:
            at+=rng.expovariate(rate)
            if at>=duration:break
            while phase+1<len(phases) and phases[phase+1][0]<=at:phase+=1
            src=rng.randrange(hosts); dst=rng.randrange(hosts-1)
            if dst>=src:dst+=1
            if rng.random()<phases[phase][2]:dst=((src//128)*8+phases[phase][1]%8)*16+rng.randrange(16)
            if dst==src:dst=(dst+1)%hosts
            flows.append((round((2+at)*1e9),src,dst,rng.choices(sizes,weights)[0]))
    elif scenario == 'uniform':
        rate = hosts*.35e9/(23000*8); at = 0.
        while True:
            at += rng.expovariate(rate)
            if at >= duration: break
            src = rng.randrange(hosts); dst = rng.randrange(hosts-1)
            if dst >= src: dst += 1
            flows.append((round((2+at)*1e9), src, dst, rng.choices(sizes,weights)[0]))
    else:
        for group in range(hosts//128):
            offset = group*128; at = 0.
            while True:
                at += rng.expovariate(8000)
                if at >= duration: break
                flows.append((round((2+at)*1e9),offset+rng.randrange(16),offset+16+rng.randrange(16),rng.choices(sizes,weights)[0]))
            # Each background flow is statically pinned by source identity.
            # Two unrelated source racks converge onto one destination-spine link.
            period = .0005
            for phase in range(int(duration/period)):
                first = 32 if scenario in ['persistent','sweep'] or phase%2 == 0 else 64
                for index in range(32):
                    at = phase*period+rng.uniform(0, .000025)
                    flows.append((round((2+at)*1e9),offset+first+index,offset+16+rng.randrange(16),4096))
    flows.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(len(flows))+'\n'+''.join(f'{s} {d} 3 {b} {ns/1e9:.9f}\n' for ns,s,d,b in flows))
    if scenario=='sweep':
        schedule=path.with_suffix('.paths.txt')
        schedule.write_text(''.join(f'{i} {((ns-2000000000)//500000)%8}\n' for i,(ns,s,d,b) in enumerate(flows) if 32<=s%128<96))
    dump(path.with_suffix('.json'), dict(seed=seed,scenario=scenario,hosts=hosts,duration=duration,
        flows=len(flows),bytes=sum(f[3] for f in flows),sha256=sha(path),sizes=sizes,weights=weights,
        foreground_count=sum(s%128<32 or s%128>=96 or scenario in ['uniform','mixed','mixed_fast'] for _,s,_,_ in flows)))
    return path


def make(name, scenario='uniform', seed=9601, hosts=128, method='conga', remote=0,
         delay=0, gap=100, cnp=1, enabled=1, duration=.05, binary=None,
         feedback=0, period=50, guard=0, margin=0, extra=None, quantum=0, age_guard=0,padded=0):
    folder = TASK/'runs'/name
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    traffic = trace(scenario,seed,hosts,duration)
    topo = NS/'config'/('leaf_spine_128_100G_asym_OS2.txt' if hosts==128 else 'leaf_spine_1024_1G_asym_OS2.txt')
    overrides = dict(TOPOLOGY_FILE=str(topo), FLOW_FILE=str(traffic), FLOWGEN_STOP_TIME=str(2+duration),
        QLEN_MON_START='2.0',QLEN_MON_END='2.0', LB_MODE=str(METHODS[method]),
        CC_MODE='1' if method=='conweave' else '8', RANDOM_SEED='100',
        HAS_WIN='0' if method=='conweave' else '1',VAR_WIN='0' if method=='conweave' else '1',
        CONWEAVE_REPLY_TIMEOUT_EXTRA='200',DIAG_ENABLE=str(enabled),DIAG_REMOTE=str(remote),
        DIAG_DELAY_NS=str(delay*1000), DIAG_GAP_NS=str(gap*1000), DIAG_OOO_CNP=str(cnp),
        DIAG_SAMPLE_NS='10000',DIAG_BACKGROUND_MODE=str(dict(uniform=0,storage=0,mixed=0,mixed_fast=0,persistent=1,rotating=2,sweep=3)[scenario]),
        DIAG_OUTPUT=str(folder/'diagnostic.json'))
    overrides.update(DIAG_FEEDBACK=str(feedback),DIAG_FEEDBACK_PERIOD_NS=str(period*1000),
        DIAG_GUARD=str(guard),DIAG_GUARD_MARGIN_NS=str(margin*1000),
        DIAG_REPORT_QUANTUM_BYTES=str(quantum),DIAG_AGE_GUARD=str(age_guard),DIAG_PADDED_REPORTS=str(padded))
    overrides.update(extra or {})
    if scenario=='sweep':overrides['DIAG_BACKGROUND_PATHS_FILE']=str(traffic.with_suffix('.paths.txt'))
    for line in BASE.splitlines():
        parts = line.split()
        if parts and (parts[0].endswith('_OUTPUT_FILE') or parts[0].endswith('_MON_FILE') or parts[0]=='FLOW_INPUT_FILE' or parts[0]=='VOQ_MON_DETAIL_FILE'):
            overrides[parts[0]] = str(folder/(parts[0].lower()+'.txt'))
    out=[]
    for line in BASE.splitlines():
        parts=line.split()
        if parts and parts[0] in overrides: out.append(parts[0]+' '+overrides.pop(parts[0]))
        else: out.append(line)
    out += [k+' '+v for k,v in overrides.items()]
    config=folder/'config.txt'; config.write_text('\n'.join(out)+'\n')
    b=Path(binary or BINARY)
    job=dict(name=name,scenario=scenario,traffic_seed=seed,hosts=hosts,method=method,remote=remote,
        delay_us=delay,gap_us=gap,cnp=cnp,enabled=enabled,duration=duration,config=str(config),binary=str(b),
        traffic=str(traffic),traffic_sha256=sha(traffic),topology=str(topo),topology_sha256=sha(topo),
        binary_sha256=sha(b),config_sha256=sha(config),feedback=feedback,period_us=period,
        guard=guard,margin_us=margin,extra=extra)
    job.update(quantum=quantum,age_guard=age_guard,padded=padded)
    if scenario=='sweep':
        job['background_paths_file']=str(traffic.with_suffix('.paths.txt'))
        job['background_paths_sha256']=sha(job['background_paths_file'])
    dump(folder/'job.json',job)
    return folder


def execute(folder):
    folder=Path(folder); job=json.loads((folder/'job.json').read_text())
    b=Path(job['binary']); lib=b.parent.parent
    libraries = [str(lib), os.environ.get('LD_LIBRARY_PATH', '')]
    env={**os.environ,'LD_LIBRARY_PATH':':'.join(p for p in libraries if p),'OMP_NUM_THREADS':'1'}
    state=dict(state='running',started_epoch=time.time(),name=job['name'])
    dump(folder/'status.json',state)
    with (folder/'log.txt').open('wb') as log:
        p=subprocess.Popen([str(b),job['config']],cwd=NS,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=p.pid;dump(folder/'status.json',state)
        try: rc=p.wait(timeout=1800)
        except subprocess.TimeoutExpired: p.kill();p.wait();rc=-999
    state.update(state='complete' if rc==0 else 'failed',returncode=rc,elapsed_seconds=time.time()-state['started_epoch'])
    dump(folder/'status.json',state)
    if rc: raise RuntimeError(str(folder))
    print('COMPLETE',job['name'],round(state['elapsed_seconds'],2),flush=True)
    return folder


def load(folder):
    job=json.loads((folder/'job.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    assert status['state']=='complete', folder
    assert job['traffic_sha256']==sha(job['traffic'])
    assert job['config_sha256']==sha(job['config'])
    offered={}; sports=defaultdict(lambda:10000);dports=defaultdict(lambda:100)
    lines=open(job['traffic']).read().splitlines();assert int(lines[0])==len(lines)-1
    for line in lines[1:]:
        s,d,pg,b,t=line.split();s,d,b=int(s),int(d),int(b)
        key=(s,d,sports[s],dports[d]);sports[s]+=1;dports[d]+=1
        assert key not in offered
        group=int(job['scenario'] in ['persistent','rotating','sweep'] and 32<=s%128<96)
        offered[key]=dict(size=b,group=group,start_ns=round(float(t)*1e9))
    flows={}
    for line in (folder/'fct_output_file.txt').read_text().splitlines():
        x=list(map(int,line.split()));assert len(x)==8
        key=tuple(x[:4]);assert key in offered and key not in flows
        assert x[4]==offered[key]['size'] and abs(x[5]-offered[key]['start_ns'])<=1
        assert x[6]>0 and x[7]>0
        flows[key]=dict(**offered[key],fct_us=x[6]/1000,slowdown=x[6]/x[7])
    diag=json.loads((folder/'diagnostic.json').read_text()) if job['enabled'] else None
    summary=dict(**job,completed=len(flows),offered=len(offered),elapsed_seconds=status['elapsed_seconds'],groups=[])
    for group in [0,1]:
        selected=[v for v in flows.values() if v['group']==group]
        g=dict(group=group,offered=sum(v['group']==group for v in offered.values()),completed=len(selected))
        if selected:
            for attr in ['fct_us','slowdown']:
                values=np.array([v[attr] for v in selected]);g['mean_'+attr]=float(values.mean());g['p99_'+attr]=float(np.percentile(values,99))
        if diag:
            g['diagnostic']=diag['groups'][group]
            d=g['diagnostic'];g['queue_regret_us']=d['queue_regret_ns_sum']/max(1,d['packets'])/1000
            g['ooo_fraction']=d['out_of_order']/max(1,d['rx_packets'])
        summary['groups'].append(g)
    return summary,flows


def job(name, family, seed, policy, hosts=128, model=None, epsilon=0, train_seed=301, mask=False, extra=None, duration=.05, binary=None):
    name='r128_'+name
    ex=dict(SIMULATOR_EXTRA_TIME='1.0',IRN_RTO_LOW_US='10000',IRN_RTO_HIGH_US='32000')
    opts=dict(method='oracle',remote=1,gap=0,feedback=1,period=200,quantum=1024)
    if policy=='conga_ack':opts=dict(method='conga');ex['CONGA_ACK_FEEDBACK']='1'
    elif policy=='original_adaptive':opts.update(guard=1,margin=50,age_guard=1)
    elif policy=='original_flow':opts.update(gap=1000000)
    else:
        ex.update(RL_SWITCH_MODE=str(RULES[policy]),RL_SWITCH_OUTPUT=str(TASK/'runs'/name/'transitions.csv'),
            RL_SWITCH_SEED=str(train_seed),RL_SWITCH_EPSILON=str(epsilon),RL_SWITCH_MASK_HISTORY=str(int(mask)))
        if policy.startswith('ack'):ex['RL_SWITCH_MARGIN_NS']=str(int(policy[3:])*1000)
        if policy.startswith('fixed'):ex['RL_SWITCH_MARGIN_NS']=str(int(policy[5:])*1000)
        if model:ex['RL_SWITCH_MODEL']=str(model)
    ex.update(extra or {})
    f=make(name,scenario=family,seed=seed,hosts=hosts,duration=duration,binary=binary,extra=ex,**opts)
    j=json.loads((f/'job.json').read_text());j.update(policy=policy,training_seed=train_seed,mask=mask,epsilon=epsilon,
        model=str(model) if model else None,model_sha256=sha(model) if model else None)
    dump(f/'job.json',j);return f


def prepare(name,family,traffic,model=None,mode=9,reselect=1,margin=0,interval=50,method=None,optimizer=0,binary=None,execution_seed=100,config_extra=None):
    name='v2_'+name
    extra={'RL_SWITCH_MODE':str(mode),'RL_PATH_RESELECT':str(reselect),'RL_SWITCH_MARGIN_NS':str(round(margin*1000)),
           'RL_SWITCH_INTERVAL_NS':str(round(interval*1000)),'RANDOM_SEED':str(execution_seed)}
    extra.update(config_extra or {})
    if method=='conga_ack':
        f=job(name,family,traffic,'conga_ack',binary=binary or PATH_BINARY,extra={'RANDOM_SEED':str(execution_seed),**(config_extra or {})})
    else:
        f=job(name,family,traffic,'dqn',model=model,binary=binary or PATH_BINARY,extra=extra)
    j=json.loads((f/'job.json').read_text());j.update(optimizer_seed=optimizer,analysis_label=name,mode=0 if method else mode,
        reselect=reselect,margin_us=margin,interval_us=interval,execution_seed=execution_seed)
    dump(f/'job.json',j);return f


def evaluate(f):
    f=Path(f)
    try:
        j=json.loads((f/'job.json').read_text());execute(f);s,flows=load(f)
        assert s['completed']==s['offered'],f
        v=np.array([x['fct_us'] for x in flows.values()]);n=max(1,int(np.ceil(len(v)*.01)))
        finish=max(x['start_ns']+x['fct_us']*1000 for x in flows.values())
        s.update(valid=True,mode=j['mode'],reselect=j['reselect'],margin_us=j['margin_us'],interval_us=j['interval_us'],
            analysis_label=j['analysis_label'],optimizer_seed=j['optimizer_seed'],all_mean_us=float(v.mean()),
            all_p99_us=float(np.percentile(v,99)),all_p999_us=float(np.percentile(v,99.9)),all_max_us=float(v.max()),
            all_cvar99_us=float(np.sort(v)[-n:].mean()),makespan_us=(finish-2e9)*.001,
            all_goodput_gbps=sum(x['size'] for x in flows.values())*8/(finish-2e9),
            route_changes=sum(g['diagnostic']['route_changes'] for g in s['groups']))
        if (f/'transitions.csv.audit.json').exists():
            a=json.loads((f/'transitions.csv.audit.json').read_text());assert a['unfinished_episodes']==0
            assert sum(a['path_assignments'])==a['transitions'];s['rl_audit']=a
            if j['mode']==8:assert a['transitions']==a['terminals'] and s['route_changes']==0
            else:
                assert a['transitions']-a['path_reselections']==a['terminals']
                assert a['path_changes']==s['route_changes']
                b=np.loadtxt(f/'transitions.csv.boundaries.csv',delimiter=',',skiprows=1,ndmin=2)
                assert len(b)==a['transitions'];re=b[:,6]>0
                assert int(re.sum())==a['path_reselections']
                assert np.all(b[re,4]>=b[re,2])
                if j['reselect']==1:assert np.all(b[re,3]>=b[re,2]),'Undrained ACK boundary'
                if j['reselect']==0:assert not re.any()
                s['boundaries_checked']=len(b)
        s['fct_sha256']=sha(f/'fct_output_file.txt');dump(f/'result.json',s);return s
    except Exception as exc:
        j=json.loads((f/'job.json').read_text());r={**j,'valid':False,'error':repr(exc)};dump(f/'error.json',r);return r


def make_run(name,fam,traffic,actor=None,gate_model=None,gate_mode=1,margin=50,threshold=0,execution=100,override=None,mode=9,reselect=3,binary=None,conga=False,admission_ppm=1000000,admission_seed=901,tail_threshold=0,pressure=True,pressure_mask=False):
    path=TASK/'runs'/('r128_v2_'+name)
    if path.exists():
        j=json.loads((path/'job.json').read_text())
        if 'gate_mode' not in j:
            j.update(gate_mode=0 if conga else gate_mode,gate_model=str(gate_model) if gate_model else None,
                gate_model_sha256=sha(gate_model) if gate_model else None,gate_threshold_ms=threshold,override=override)
            dump(path/'job.json',j)
        return path
    if actor is None and not conga:
        raise ValueError('A native path model is required')
    extra={'RL_PRESSURE_ENABLE':str(int(pressure)),'RL_PRESSURE_MASK':str(int(pressure_mask)),'RL_GATE_MODE':str(gate_mode),'RL_GATE_THRESHOLD_MS':str(threshold),
        'RL_GATE_TAIL_THRESHOLD_MS':str(tail_threshold),'RL_MIGRATION_ADMISSION_PPM':str(admission_ppm),'RL_MIGRATION_ADMISSION_SEED':str(admission_seed)}
    if gate_model:extra['RL_GATE_MODEL']=str(gate_model)
    if override is not None:
        extra.update(RL_SWITCH_OVERRIDE_FLOW=str(override['flow']),RL_SWITCH_OVERRIDE_STEP=str(override['step']),RL_SWITCH_OVERRIDE_ACTION=str(override['action']))
    if conga:extra={'CONGA_ACK_FEEDBACK':'0'};pressure=False;pressure_mask=False
    f=prepare(name,fam,traffic,model=actor,mode=mode,reselect=reselect,margin=margin,binary=binary or BINARY,
        execution_seed=execution,method='conga_ack' if conga else None,config_extra=extra)
    j=json.loads((f/'job.json').read_text());j.update(gate_mode=0 if conga else gate_mode,gate_model=str(gate_model) if gate_model else None,
        gate_model_sha256=sha(gate_model) if gate_model else None,gate_threshold_ms=threshold,override=override,
        admission_ppm=admission_ppm,admission_seed=admission_seed,gate_tail_threshold_ms=tail_threshold,pressure=pressure,pressure_mask=pressure_mask)
    dump(f/'job.json',j);return f


def run_one(f):
    f=Path(f)
    if (f/'result.json').exists():return json.loads((f/'result.json').read_text())
    return evaluate(f)


def read(name):
    return json.loads((TASK / name).read_text())


def write(name, value):
    dump(TASK / name, value)


def make_trial(name, family, traffic, label, phase, model=None, fit=0, mode=8, conga=False):
    f = make_run(name, family, traffic, actor=model, mode=mode, reselect=0,
                     gate_mode=0, pressure=False, pressure_mask=False, margin=0,
                     binary=BINARY, conga=conga)
    j = json.loads((f / "job.json").read_text())
    j.update(label=label, phase=phase, fit_seed=fit)
    dump(f / "job.json", j)
    return f


def collect_run(f):
    if (f / "result.json").exists():
        return json.loads((f / "result.json").read_text())
    if not (f / "status.json").exists():
        r = run_one(f)
        if r["valid"]:
            return r
    status = json.loads((f / "status.json").read_text())
    if status["state"] != "complete" or status["returncode"] != 0:
        raise RuntimeError((str(f), "native simulator did not exit cleanly"))
    summary, flows = load(f)
    if summary["completed"] == summary["offered"]:
        raise RuntimeError((str(f), "non-censoring audit failure"))
    stop = json.loads((f / "diagnostic.json.unfinished.jsonl").read_text().splitlines()[0])
    assert stop["offered"] == summary["offered"] and stop["completed"] == summary["completed"]
    assert stop["stop_ns"] == 3049999999
    summary.update(valid=False, training_failure="unfinished_at_horizon", stop_ns=stop["stop_ns"],
                   fct_sha256=sha(f / "fct_output_file.txt"))
    dump(f / "result.json", summary)
    return summary


def execute_jobs(jobs):
    rows = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(collect_run, f): f for f in jobs}
        for done in as_completed(futures):
            r = done.result()
            if not r["valid"] and r.get("training_failure") != "unfinished_at_horizon":
                raise RuntimeError((str(futures[done]), r.get("error")))
            rows.append(r)
    return sorted(rows, key=lambda x: x["name"])
