#!/usr/bin/env python3
"""Reconstruct the queue/reservation prior and isolate learned policy corrections.

Reads only already-written native logs. This is an inference audit, never a
counterfactual performance estimate or a replacement for learning-off runs.
"""
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np

def audit(folder):
 j=json.loads((folder/'job.json').read_text());assert j['arm'] in ['ac_on','ac_off','hold_on','hold_off','ac_ack'];tau=float(j['extra']['FAC_TEMPERATURE']);reservations={};previous={};tvs=[];residuals=[];greedy=[];versions=[];max_frozen=0
 with (folder/'transitions.csv.fac-decisions.csv').open() as df,(folder/'transitions.csv.candidates.csv').open() as cf:
  decisions=csv.DictReader(df);candidates=csv.DictReader(cf)
  for d in decisions:
   group=[next(candidates) for _ in range(8)];now=int(d['ns']);node=int(d['node']);key=int(d['key']);port=int(d['port']);chosen=next(i for i,v in enumerate(group) if int(v['selected']));assert [int(v['candidate']) for v in group]==list(range(8));assert all(int(v['start_ns'])==now for v in group)
   assert 17<=port<=24;offset=(port-17-chosen)%8;prior=[];actual=[]
   for i,v in enumerate(group):
    candidate_port=17+(offset+i)%8;at,count=reservations.get((node,candidate_port),(0,0.));recent=count*math.exp(-(now-at)/200000.) if at else 0.;base=.008+float(v['x0'])+.25*float(v['x1'])+.1*float(v['x8'])+recent*float(v['x2']);prior.append(-tau*math.log(base/.02));actual.append(-float(v['score']))
   prior,actual=np.array(prior),np.array(actual);difference=actual-prior;centered=difference-difference.mean();residuals.append(float(np.max(abs(centered))));versions.append(int(d['version']))
   def mix(x):
    p=np.exp(x-x.max());return .99*p/p.sum()+.01/len(x)
   observed=mix(actual);assert abs(observed[chosen]-float(d['probability']))<1e-8
   tvs.append(float(abs(observed-mix(prior)).sum()/2));greedy.append(bool(prior[np.argmax(actual)]<prior.max()-1e-7))
   if int(d['version'])==0:max_frozen=max(max_frozen,float(np.max(abs(difference))))
   if previous.get((node,key))!=port:
    at,count=reservations.get((node,port),(0,0.));reservations[node,port]=(now,(count*math.exp(-(now-at)/200000.) if at else 0.)+1)
   previous[node,key]=port
  assert next(candidates,None) is None
 assert max_frozen<1e-7
 if j['arm'] in ['ac_off','hold_off']:assert max(tvs)<1e-8 and not any(greedy)
 def summarize(mask):
  inds=np.flatnonzero(mask)
  if not len(inds):return None
  return dict(decisions=len(inds),greedy_changes=int(np.asarray(greedy)[inds].sum()),greedy_change_fraction=float(np.asarray(greedy)[inds].mean()),mean_policy_total_variation=float(np.asarray(tvs)[inds].mean()),policy_total_variation_percentiles=dict(zip(['p50','p95','p99'],np.percentile(np.asarray(tvs)[inds],[50,95,99]).tolist())),centered_logit_correction_percentiles=dict(zip(['p50','p95','p99'],np.percentile(np.asarray(residuals)[inds],[50,95,99]).tolist())))
 return dict(folder=str(folder),arm=j['arm'],family=j['family'],topology=j['topology_name'],seed=j['seed'],fit=j['fit'],all=summarize(np.ones(len(versions),dtype=bool)),updated_version=summarize(np.array(versions)>0),max_zero_version_logit_error=max_frozen)

def main():
 p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args();results=[audit(f) for f in a.folders];a.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
