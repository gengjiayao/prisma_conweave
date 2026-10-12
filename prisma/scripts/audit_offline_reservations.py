#!/usr/bin/env python3
"""Independent reconstruction of native reservation state and sampled actions."""
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np

def uniform(key,node,seed):
    mask=(1<<64)-1;x=(key^(node<<32)^seed)+0x9e3779b97f4a7c15;x&=mask
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask;x=((x^(x>>27))*0x94d049bb133111eb)&mask
    return ((x^(x>>31))>>11)/9007199254740992.

def audit(folder):
    j=json.loads((folder/'job.json').read_text());a=json.loads((folder/'transitions.csv.offline-audit.json').read_text())
    states=np.loadtxt(folder/'transitions.csv.candidates.csv',delimiter=',',skiprows=1).reshape(-1,8,29)
    x=states[:,:,5:];v=np.array(list(map(float,Path(j['model']).read_text().split()[3:])))
    scores=np.maximum(0,x@v[:768].reshape(24,32)+v[768:800])@(v[800:864].reshape(32,2)[:,1]-v[800:864].reshape(32,2)[:,0])+v[865]-v[864]
    score_error=float(abs(scores-states[:,:,4]).max());assert score_error<1e-8
    ports={};dest={};seen=set();decisions=defaultdict(int);dispersed=defaultdict(int);largest=0.
    def read(table,key,t):
        if key not in table:return 0.
        previous,value=table[key];assert previous<=t
        return value*math.exp(-(t-previous)/a['decay_ns'])
    with (folder/'transitions.csv.offline.csv').open() as f:
        it=csv.DictReader(f)
        for q,group in enumerate(states):
            log=[next(it) for _ in range(8)];node=int(log[0]['node']);key=int(log[0]['key']);flow=int(log[0]['flow']);t=int(log[0]['ns']);assert (node,key) not in seen;seen.add((node,key))
            assert flow==int(group[0,0]) and t==int(group[0,1]);u=uniform(key,node,a['seed'])
            native_scores=np.array([float(r['score']) for r in log]);greedy=0
            for i in range(1,8):
                if native_scores[i]<native_scores[greedy]-1e-12:greedy=i
            prob=np.where(native_scores<=native_scores[greedy]+a['near_ms']+1e-12,1/x[q,:,2],0.) if a['spread'] else np.eye(8)[greedy]
            prob/=prob.sum();selected=int(np.searchsorted(np.cumsum(prob),u,side='right'));assert selected<8
            for i,r in enumerate(log):
                assert int(r['candidate'])==i and int(r['greedy'])==greedy and int(r['selected'])==int(i==selected)
                assert int(group[i,3])==int(i==selected) and abs(float(r['uniform'])-u)<1e-15 and abs(float(r['probability'])-prob[i])<1e-10
                p=int(r['port']);prefix=int(r['prefix']);width=prefix&255;network=prefix>>8;ip=0x0b000001+((key&65535)<<8)
                mask=((0xffffffff<<(32-width))&0xffffffff);assert ip&mask==network
                if not int(x[q,i,14]):assert width==32
                pc=read(ports,(node,p),t) if a['reservations'] else 0.;dc=read(dest,(node,p,prefix),t) if a['reservations'] else 0.
                for idx,value,col in [(22,pc*x[q,i,2],'port_reservation_ms'),(23,dc*x[q,i,3],'destination_reservation_ms')]:
                    err=abs(value-x[q,i,idx]);largest=max(largest,err);assert err<1e-8 and abs(value-float(r[col]))<1e-8
            chosen=log[selected];p=int(chosen['port']);prefix=int(chosen['prefix'])
            if a['reservations']:
                ports[node,p]=(t,read(ports,(node,p),t)+1);dest[node,p,prefix]=(t,read(dest,(node,p,prefix),t)+1)
            decisions[node]+=1;dispersed[node]+=selected!=greedy
        assert next(it,None) is None
    assert sum(decisions.values())==len(states)
    for leaf in a['leaves']:assert decisions[leaf['node']]==leaf['decisions'] and dispersed[leaf['node']]==leaf['dispersed']
    return dict(name=j['name'],decisions=len(states),candidate_rows=len(states)*8,max_reservation_error_ms=largest,max_score_error=score_error,passed=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--phase',default='smoke');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    jobs=[]
    for folder in (args.experiment/'runs').glob(args.phase+'_*'):
        if (folder/'transitions.csv.offline.csv').exists():jobs.append(folder)
    results=[]
    for folder in sorted(jobs):results.append(audit(folder));print('VERIFIED',folder.name,flush=True)
    args.output.write_text(json.dumps(dict(runs=len(results),decisions=sum(r['decisions'] for r in results),results=results),indent=2)+'\n')
if __name__=='__main__':main()
