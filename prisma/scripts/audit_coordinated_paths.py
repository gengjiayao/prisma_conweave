#!/usr/bin/env python3
"""Independently replay causal packet spans, reservations and conditional routing."""
import argparse,csv,json,math,heapq
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict
import numpy as np
MASK=(1<<64)-1

def hash64(x):
    x=(x+0x9e3779b97f4a7c15)&MASK;x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK;x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def audit(folder):
    j=json.loads((folder/'job.json').read_text());a=json.loads((folder/'transitions.csv.coordinated-audit.json').read_text());theta=np.array(a['theta'])
    assert a['packet_log']==1
    loaded=list(map(float,Path(j['coord_model']).read_text().split()[1:]));assert np.array_equal(theta,loaded)
    native=np.loadtxt(folder/'transitions.csv.candidates.csv',delimiter=',',skiprows=1).reshape(-1,8,29)
    cells={};ports={};flows={};arrivals={};byport=defaultdict(set);idle=defaultdict(list);estimates=defaultdict(lambda:65536.);nsamples=defaultdict(int);packets=defaultdict(int);bytes_=defaultdict(int);decisions=defaultdict(int);nongreedy=defaultdict(int)
    maxerror=0.;index=0
    def read(table,key,t):
        previous,v=table.get(key,(t,np.zeros(3)));return v*np.exp(-(t-previous)/np.array([1e6,2e7,2e7]))
    def add(table,key,t,v):table[key]=(t,read(table,key,t)+v)
    def destination(node,port,prefix,t):
        width=prefix&255;first=prefix>>8;last=first|((1<<(32-width))-1)
        return sum((read(cells,(node,port,d),t) for d in byport[node,port] if first<=d<=last),np.zeros(3))
    def prune(node):
        h=idle[node]
        while h and (flows[node,h[0][1]]['sampled'] or flows[node,h[0][1]]['last']+2000000!=h[0][0]):heapq.heappop(h)
    def close(x,y,tol=1e-8):
        nonlocal maxerror
        error=abs(float(x)-float(y));maxerror=max(maxerror,error);assert error<=tol,(x,y,error)
    with (folder/'transitions.csv.coordinated.csv').open() as lf,(folder/'transitions.csv.coordinated-events.csv').open() as ef:
        logs=csv.DictReader(lf)
        for e in csv.DictReader(ef):
            kind=e['kind'];t=int(e['ns']);node=int(e['node']);key=int(e['key']);port=int(e['port']);dip=int(e['destination']);end=int(e['end']);delta=int(e['delta']);fk=(node,key)
            if kind=='I':
                prune(node);assert idle[node] and idle[node][0][1]==key and idle[node][0][0]<=t;heapq.heappop(idle[node])
                f=flows[fk];assert not f['sampled'] and t-f['last']>=2000000 and f['end']==end and (f['port'],f['dip'])==(port,dip)
                estimates[node]=.875*estimates[node]+.125*max(1024.,min(1048576.,end));f['sampled']=True;nsamples[node]+=1;close(estimates[node],e['estimate'])
            elif kind=='D':
                f=flows[fk];assert (f['port'],f['dip'])==(port,dip);assert delta==max(0,end-f['end']);f['end']=max(f['end'],end)
                debt=f['debt']*math.exp(-(t-f['debt_ns'])/2e7);consumed=min(debt,delta);f['debt']=debt-consumed;f['debt_ns']=t
                add(cells,(node,port,dip),t,np.array([0,0,-consumed]));add(ports,(node,port),t,np.array([0,0,-consumed]));f['last']=t
                if not f['sampled']:heapq.heappush(idle[node],(t+2000000,key))
                packets[node]+=1;bytes_[node]+=delta;close(estimates[node],e['estimate'])
            elif kind=='A':
                assert fk not in flows
                prune(node);assert not idle[node] or idle[node][0][0]>t
                log=[next(logs) for _ in range(8)];group=native[index];index+=1
                assert int(group[0,0])==int(log[0]['flow']) and int(group[0,1])==t
                x=group[:,5:];old_t,old_a=arrivals.get(node,(t,0.));burst=old_a*math.exp(-(t-old_t)/200000.);gate=burst/(burst+theta[5]) if a['conditional'] else 1.
                prediction=max(1024.,min(4194304.,estimates[node]*theta[7]));close(prediction,e['prediction']);close(estimates[node],e['estimate'])
                trust=1/(1+x[:,4]/theta[6])*(1-.75*gate) if a['conditional'] else np.ones(8)
                reserve=.25+.75*gate if a['conditional'] else 1.
                scores=x[:,0]+theta[0]*trust*x[:,1]+theta[1]*reserve*x[:,23]+theta[2]*reserve*x[:,22]+theta[3]*x[:,8]
                greedy=0
                for i in range(1,8):
                    if scores[i]<scores[greedy]-1e-12:greedy=i
                stable=sorted(int(r['port']) for r in log);fairs=[];eligible=scores<=scores[greedy]+theta[4]*gate+1e-12
                for i,r in enumerate(log):
                    assert (int(r['node']),int(r['key']),int(r['destination']),int(r['ns']),int(r['candidate']))==(node,key,dip,t,i)
                    p=int(r['port']);prefix=int(r['prefix']);width=prefix&255;mask=(0xffffffff<<(32-width))&0xffffffff
                    assert dip&mask==prefix>>8
                    if not int(x[i,14]):assert width==32
                    v=destination(node,p,prefix,t);pv=read(ports,(node,p),t);field=2 if a['work'] else 0;scale=1/65536 if a['work'] else 1
                    close(x[i,22],pv[field]*scale*x[i,2]);close(x[i,23],v[field]*scale*x[i,3])
                    close(r['port_reservation_ms'],x[i,22]);close(r['destination_reservation_ms'],x[i,23]);close(scores[i],r['score']);close(scores[i],group[i,4])
                    phase=((stable.index(p)+hash64((node<<32)^prefix^a['seed'])%8)%8)/8;capacity=1/max(x[i,2],x[i,3]);fair=(v[1]+1+phase)/capacity;fairs.append(fair)
                    close(r['phase'],phase);close(r['capacity'],capacity,1e-6);close(r['slots'],v[1]);close(r['fair'],fair);close(r['burst'],burst);close(r['gate'],gate);close(r['estimate'],estimates[node]);close(r['prediction'],prediction)
                    assert int(r['greedy'])==greedy and int(r['eligible'])==int(eligible[i])
                selected=greedy
                if a['balance']:
                    first=True
                    for i in range(8):
                        if eligible[i] and (first or fairs[i]<fairs[selected]-1e-12 or (abs(fairs[i]-fairs[selected])<=1e-12 and int(log[i]['port'])<int(log[selected]['port']))):selected=i;first=False
                assert int(log[selected]['port'])==port
                for i,r in enumerate(log):assert int(r['selected'])==int(group[i,3])==int(i==selected)
                byport[node,port].add(dip)
                add(cells,(node,port,dip),t,np.array([1,1,prediction]));add(ports,(node,port),t,np.array([1,1,prediction]));arrivals[node]=(t,burst+1)
                flows[fk]=dict(port=port,dip=dip,end=0,last=-1,sampled=False,debt=prediction,debt_ns=t)
                decisions[node]+=1;nongreedy[node]+=selected!=greedy
            else:raise AssertionError(kind)
        assert next(logs,None) is None and index==len(native)
    for leaf in a['leaves']:
        n=leaf['node'];assert leaf['decisions']==decisions[n] and leaf['non_greedy']==nongreedy[n] and leaf['idle_samples']==nsamples[n] and leaf['observed_packets']==packets[n] and leaf['observed_unique_span_bytes']==bytes_[n];close(leaf['estimated_bytes'],estimates[n])
    return dict(name=j['name'],decisions=index,packets=sum(packets.values()),idle_samples=sum(nsamples.values()),max_numeric_error=maxerror,passed=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',default='smoke');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();results=[]
    folders=[f for f in sorted((a.root/'runs').glob(a.phase+'_*')) if (f/'transitions.csv.coordinated-events.csv').exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(audit,folders):results.append(result);print('AUDITED',result['name'],flush=True)
    a.output.write_text(json.dumps(dict(runs=len(results),decisions=sum(r['decisions'] for r in results),packets=sum(r['packets'] for r in results),results=results),indent=2)+'\n')
if __name__=='__main__':main()
