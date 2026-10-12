#!/usr/bin/env python3
"""Exploratory, paired interventions on frozen CABLE; never trains online."""
import argparse, csv, json, os, resource, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import collective_evaluation as coll
from collective_evaluation import env, np

ARCHIVE=Path('/home/gengjiayao/ict/work/prisma_conweave_data/new_runs/cable-collectives-20260917-h3a4o58p')
VARIANTS={'no_remote':[1,5], 'no_history':[7,8], 'no_port':[8],
          'no_destination':[7], 'no_serialization':[2,3],
          'local_only':[1,2,3,5,6,7,8], 'feedback20':[], 'reselect':[]}

def write_model(source, target, remove, preserve_remote_serial=False):
    vals=source.read_text().split(); assert vals[:3]==['ROUTE_DQN_V1','24','32']
    data=np.array(list(map(float,vals[3:]))); assert len(data)==866
    w=data[:768].reshape(24,32)
    if preserve_remote_serial:w[3,:]+=w[1,:]
    for feature in remove:w[feature,:]=0
    target.write_text('ROUTE_DQN_V1 24 32\n'+' '.join(format(x,'.17g') for x in data)+'\n')

def prepare(root, decomposition=False):
    root.mkdir(parents=True,exist_ok=True)
    for d in ['models','source','runs']:(root/d).mkdir(exist_ok=True)
    if not (root/'binaries').exists():shutil.copytree(ARCHIVE/'formal/binaries',root/'binaries',symlinks=True)
    env.configure(root,root/'binaries/scratch/network-load-balance')
    rows=[r for r in json.loads((ARCHIVE/'analysis/collective-runs.json').read_text()) if r['family']=='alltoall']
    protocol=dict(created_epoch=time.time(),purpose='Diagnosis, not held-out claims for a newly tuned policy',
                  original_archive=str(ARCHIVE),seeds=[89001,89002],ranks=[32,64],sizes=[1024,8192,65536],
                  fits=[9101,9102,9103],topologies=['symmetric','asymmetric'],variants=VARIANTS,
                  preserved='Original frozen simulator/libraries, packet feedback, offered traffic, all other model coefficients and transport settings',
                  reselect='Frozen same actor; ACK-drained decisions at least 50us apart; no learning',
                  feedback20='Real packet period 20us instead of 200us; includes extra control cost',
                  hash='Zero scoring model, same flow-hash tie order and same feedback overhead',
                  decomposition=decomposition,
                  remote_queue_removal='For queue-removal arms transfer x1 input weights to x3 before zeroing x1; retain the remote serialization contribution exactly.',
                  expected_runs=288 if decomposition else 612)
    if not (root/'protocol.json').exists():env.dump(root/'protocol.json',protocol)
    jobs=[]
    specs=[(r,v) for r in rows if r['arm']=='rl' and r['seed'] in protocol['seeds'] for v in VARIANTS]
    if not decomposition:
        specs += [(r,'hash') for r in rows if r['arm']=='rl' and r['fit']==9101 and r['seed'] in protocol['seeds']]
        specs += [(r,'replay') for r in rows if r['arm']=='rl' and r['seed']==89001 and r['size']==65536]
    for r,variant in specs:
        src=ARCHIVE/'formal/runs'/r['name'];j=json.loads((src/'job.json').read_text())
        name=r['name'].replace('main_','diagnosis_').replace('_rl_',f'_{variant}_');folder=root/'runs'/name
        if folder.exists():jobs.append(folder);continue
        model=root/'models'/f"{variant}_{r['fit']}.txt"
        if not model.exists():write_model(Path(j['model']),model,list(range(24)) if variant=='hash' else VARIANTS.get(variant,[]),preserve_remote_serial='queue' in variant)
        env.trace=lambda *a,_p=Path(j['traffic']),**kw:_p
        ex={**j['extra'],'RL_SWITCH_MODEL':str(model),'RL_SWITCH_OUTPUT':str(folder/'transitions.csv'),
            'COLLECTIVE_EVENT_FILE':str(folder/'collective-events.txt')}
        if variant=='reselect':ex.update(RL_SWITCH_MODE='9',RL_PATH_RESELECT='1',RL_SWITCH_INTERVAL_NS='50000')
        f=env.make(name,scenario='uniform',seed=r['seed'],method='oracle',remote=1,feedback=1,
                   period=20 if variant=='feedback20' else 200,quantum=1024,gap=0,extra=ex)
        new=json.loads((f/'job.json').read_text())
        for key in ['fit','family','ranks','size','seed','topology_name','plan','plan_sha256','plan_metadata','topology','topology_sha256']:new[key]=j[key]
        new.update(arm=variant,phase='diagnosis',model=str(model),model_sha256=env.sha(model),
                   reference_folder=str(src),reference_metrics={k:r[k] for k in ['p95_us','p99_us','makespan_us','mean_us']})
        env.dump(f/'job.json',new);jobs.append(f)
    assert len(jobs)==protocol['expected_runs'],len(jobs)
    shutil.copy2(__file__,root/'source'/Path(__file__).name)
    env.dump(root/'jobs.json',list(map(str,jobs)));return jobs

def audit_reselection(folder):
    j=json.loads((folder/'job.json').read_text());p=json.loads(Path(j['plan_metadata']).read_text())
    for key in ['binary','config','traffic','plan','topology','model']:assert env.sha(j[key])==j[key+'_sha256']
    starts={};ends={};keys={}
    for line in (folder/'collective-events.txt').read_text().splitlines():
        x=line.split();idx=int(x[1]);t=int(x[2])
        if x[0]=='start':
            s,d,sp,dp,b=map(int,x[3:]);f=p['flows'][idx]
            assert [s,d,b]==[f['src'],f['dst'],f['bytes']] and t==f['release_ns'] and idx not in starts
            starts[idx]=t;keys[s,d,sp,dp]=idx
        else:assert idx in starts and idx not in ends;ends[idx]=t
    values={}
    for line in (folder/'fct_output_file.txt').read_text().splitlines():
        x=list(map(int,line.split()));idx=keys[tuple(x[:4])]
        assert idx not in values and x[6]==ends[idx]-starts[idx] and x[4]==p['flows'][idx]['bytes']
        values[idx]=x[6]/1000
    assert len(values)==len(p['flows'])
    a=json.loads((folder/'transitions.csv.audit.json').read_text());assert a['unfinished_episodes']==0
    with (folder/'transitions.csv.boundaries.csv').open() as f:
        for b in csv.DictReader(f):
            if int(b['previous_port']):assert int(b['cumulative_ack'])>=int(b['sent_end']) and int(b['sequence'])>=int(b['sent_end'])
    row={k:j[k] for k in ['name','arm','fit','family','ranks','size','seed','topology_name','phase']}
    row.update(native_ok=True,analysis_ok=True,all_complete=True,offered=len(values),completed=len(values),
               p95_us=float(np.percentile(list(values.values()),95)),p99_us=float(np.percentile(list(values.values()),99)),
               mean_us=float(np.mean(list(values.values()))),makespan_us=(max(ends.values())-coll.START_NS)/1000,
               actor_audit=a,folder=str(folder));return row

def run(folder):
    out=folder/'diagnosis-result.json'
    if out.exists():return json.loads(out.read_text())
    j=json.loads((folder/'job.json').read_text());env.execute(folder)
    r=audit_reselection(folder) if j['arm']=='reselect' else coll.audit(folder)
    assert r['all_complete'] and r['analysis_ok']
    if j['arm']=='replay':
        for fn in ['fct_output_file.txt','transitions.csv.candidates.csv','diagnostic.json']:
            assert env.sha(folder/fn)==env.sha(Path(j['reference_folder'])/fn),fn
    r['reference_metrics']=j['reference_metrics'];r['reference_folder']=j['reference_folder']
    env.dump(out,r);return r

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--workers',type=int,default=16);a.add_argument('--remote-decomposition',action='store_true');args=a.parse_args()
    if args.remote_decomposition:
        global VARIANTS
        VARIANTS={'no_trend':[5], 'no_queue':[1], 'no_queue_trend':[1,5], 'no_queue_trend_destination':[1,5,7]}
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));jobs=prepare(args.root,args.remote_decomposition);rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(f.result());env.dump(args.root/'progress.json',dict(done=len(rows),total=len(jobs)))
    env.dump(args.root/'results.json',sorted(rows,key=lambda r:r['name']))
if __name__=='__main__':main()
