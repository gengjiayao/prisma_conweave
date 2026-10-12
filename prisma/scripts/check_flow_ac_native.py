#!/usr/bin/env python3
"""Compile native MLP finite-difference and >65535 packet-tag regression checks."""
import argparse,subprocess,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
ns=Path(__file__).resolve().parents[2]/'conweave-ns3-main';s=(ns/'src/point-to-point/model/flow-actor-critic.inc').read_text();net=s[s.index('const unsigned FAC_D'):s.index('struct FacHost')]
code='''#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cassert>
#define NS_FATAL_ERROR(x) throw std::runtime_error(x)
'''+net+r'''
int main(){
 std::mt19937 rng(1591);std::normal_distribution<double> normal(0.,.25);FacNet net;net.Init(rng);
 for(auto &w:net.w)w=normal(rng);std::array<FacX,8> xs;for(auto &x:xs)for(auto &v:x)v=normal(rng);
 double maxerr=0,maxmix=0;const double eps=1e-6;FacW analytic;net.Eval(xs[0],&analytic);
 for(unsigned k=0;k<FAC_P;++k){double orig=net.w[k];net.w[k]=orig+eps;double plus=net.Eval(xs[0]);net.w[k]=orig-eps;double minus=net.Eval(xs[0]);net.w[k]=orig;maxerr=std::max(maxerr,std::abs(analytic[k]-(plus-minus)/(2*eps)));}
 auto prob=[&](){std::array<double,8> p;double sum=0;for(unsigned i=0;i<8;++i){p[i]=std::exp(net.Eval(xs[i])+.1*i);sum+=p[i];}for(auto &v:p)v/=sum;return p;};
 auto pr=prob();std::array<FacW,8> grads;for(unsigned i=0;i<8;++i)net.Eval(xs[i],&grads[i]);unsigned chosen=3;double mixture=.99*pr[chosen]+.01/8;
 for(unsigned k=0;k<FAC_P;++k){double expected=0;for(unsigned i=0;i<8;++i)expected+=pr[i]*grads[i][k];double an=.99*pr[chosen]/mixture*(grads[chosen][k]-expected);double orig=net.w[k];net.w[k]=orig+eps;double plus=std::log(.99*prob()[chosen]+.01/8);net.w[k]=orig-eps;double minus=std::log(.99*prob()[chosen]+.01/8);net.w[k]=orig;maxmix=std::max(maxmix,std::abs(an-(plus-minus)/(2*eps)));}
 assert(maxerr<1e-7 && maxmix<1e-7);FacW gradient;gradient.fill(100);double delta=net.Apply(gradient,1,.1);assert(delta<=.100000001);
 std::cout<<"{\"mlp_gradient_max_error\":"<<maxerr<<",\"mixture_gradient_max_error\":"<<maxmix<<",\"adam_cap\":"<<delta<<"}\n";
}
'''
src=a.output/'gradient.cc';src.write_text(code);binary=a.output/'gradient';subprocess.run(['g++','-std=c++11','-O2',str(src),'-o',str(binary)],check=True);gradient=json.loads(subprocess.check_output([str(binary)]))
code='''#include "ns3/packet.h"
#include "ns3/flow-id-num-tag.h"
#include <cassert>
#include <iostream>
using namespace ns3;
int main(){for(int id: {0,65535,65536,73048,1000000}){Ptr<Packet> p=Create<Packet>(1000);FlowIDNUMTag t;t.SetId(id);t.SetFlowSize(123456);p->AddPacketTag(t);FlowIDNUMTag r;assert(p->PeekPacketTag(r));assert(r.GetId()==id && r.GetFlowSize()==123456);assert(p->GetSize()==1000);}std::cout<<"packet-tag round trips pass; packet wire size unchanged\\n";}
'''
src=a.output/'tag.cc';src.write_text(code);binary=a.output/'tag';lib=ns/'build-cable';subprocess.run(['g++','-std=c++11','-O2','-I'+str(lib),str(src),'-L'+str(lib),'-lns3.19-network-optimized','-lns3.19-core-optimized','-Wl,-rpath,'+str(lib),'-o',str(binary)],check=True);tag=subprocess.check_output([str(binary)]).decode();(a.output/'results.json').write_text(json.dumps(dict(gradient=gradient,tag=tag),indent=2)+'\n');print(gradient,tag)
