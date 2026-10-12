// Included once in switch-node.cc, inside namespace ns3.
bool FlowActorCritic::enabled=false, FlowActorCritic::learn=true;
unsigned FlowActorCritic::seed=97101, FlowActorCritic::rolloutWindows=8, FlowActorCritic::rewardMode=0;
uint64_t FlowActorCritic::startNs=2000000000, FlowActorCritic::periodNs=1000000,
    FlowActorCritic::computeNs=200000, FlowActorCritic::tau95Ns=20000000, FlowActorCritic::tau99Ns=50000000;
double FlowActorCritic::actorStep=.003, FlowActorCritic::criticStep=.01,
    FlowActorCritic::temperature=16., FlowActorCritic::lambda95=2., FlowActorCritic::lambda99=4.;
namespace {
const unsigned FAC_D=20, FAC_H=16, FAC_P=FAC_D*FAC_H+2*FAC_H;
using FacX=std::array<double,FAC_D>;
using FacW=std::array<double,FAC_P>;
using FacStats=std::array<uint64_t,10>; // integrated ns, excess95/99, active95/99, arrivals/completions, ACK ns/count
struct FacNet {
    FacW w{},m{},v{};uint64_t steps=0;
    void Init(std::mt19937 &rng) {
        std::normal_distribution<double> normal(0.,.15);
        for(unsigned i=0;i<FAC_D*FAC_H;++i)w[i]=normal(rng);
    }
    double Eval(const FacX &x, FacW *gradient=nullptr) const {
        double out=0;if(gradient)gradient->fill(0);
        for(unsigned h=0;h<FAC_H;++h) {
            double u=w[FAC_D*FAC_H+h];
            for(unsigned i=0;i<FAC_D;++i)u+=w[i*FAC_H+h]*x[i];
            double z=std::tanh(u),o=w[FAC_D*FAC_H+FAC_H+h];out+=o*z;
            if(gradient) {
                (*gradient)[FAC_D*FAC_H+FAC_H+h]=z;
                double d=o*(1.-z*z);(*gradient)[FAC_D*FAC_H+h]=d;
                for(unsigned i=0;i<FAC_D;++i)(*gradient)[i*FAC_H+h]=d*x[i];
            }
        }
        return out;
    }
    double Apply(FacW g,double step,double maxDelta) {
        double norm=0;for(double x:g)norm+=x*x;norm=std::sqrt(norm);
        if(norm>10)for(double &x:g)x*=10/norm;
        ++steps;FacW delta{};double dn=0;
        for(unsigned i=0;i<FAC_P;++i) {
            m[i]=.9*m[i]+.1*g[i];v[i]=.999*v[i]+.001*g[i]*g[i];
            delta[i]=step*(m[i]/(1-std::pow(.9,steps)))/(std::sqrt(v[i]/(1-std::pow(.999,steps)))+1e-8);
            dn+=delta[i]*delta[i];
        }
        dn=std::sqrt(dn);double scale=dn>maxDelta ? maxDelta/dn : 1.;
        for(unsigned i=0;i<FAC_P;++i) {w[i]+=scale*delta[i];if(!std::isfinite(w[i]))NS_FATAL_ERROR("Nonfinite FAC parameter");}
        return dn*scale;
    }
};
struct FacHost {
    std::function<void(const std::vector<uint8_t>&)> send;
    std::map<uint32_t,int64_t> active;
    int64_t last=0;uint64_t seq=0;FacStats pending{},total{};
};
struct FacFrame {
    uint64_t seq=0,version=0,decisions=0;FacX state{};FacW gradient{};
    double value=0;
};
struct FacFlow {int64_t first=-1,lastProbe=-1;uint64_t maxEnd=0,end=0;int64_t sent=-1;uint32_t port=0;bool pending=false;};
struct FacLeaf {
    std::mt19937 rng;FacNet actor,critic;
    std::map<uint64_t,FacFlow> flows;
    std::map<uint32_t,std::pair<int64_t,double>> reservations;
    std::map<uint64_t,std::map<uint32_t,FacStats>> hostReports,leafReports;
    std::map<uint64_t,FacStats> rewards,ackStats;
    std::map<uint64_t,int64_t> receiptNs;
    FacStats lastGlobal{};FacX lastContext{},endState{};
    std::vector<FacFrame> frames;uint64_t blockStart=0,version=0,updates=0,decisions=0,trainedDecisions=0,blocks=0;
    bool collecting=false,busy=false,ready=false;int64_t firstUpdate=-1,lastUpdate=-1,lastDecision=-1;
};
std::map<uint32_t,FacHost> facHosts;
std::map<uint32_t,FacLeaf> facLeaves;
std::map<uint32_t,uint32_t> facHostLeaf;
std::map<uint32_t,unsigned> facExpectedHosts;
std::ofstream facEvents,facHostLog,facReceiveLog,facDecisions,facUpdates,facFrames;
uint64_t facWirePackets=0,facWireBytes=0;
double FacScale=6400000000.; // 128 hosts x 50 ms, fixed by the registered fabric size
uint64_t FacSeq(int64_t now) {return uint64_t(now-int64_t(FlowActorCritic::startNs))/FlowActorCritic::periodNs;}
uint64_t FacExcess(int64_t age,uint64_t threshold) {return age>int64_t(threshold) ? age-threshold : 0;}
void FacAccumulate(FacHost &h,int64_t start,int64_t now) {
    int64_t from=std::max(start,h.last),a=now-start,b=from-start;
    h.pending[0]+=now-from;
    h.pending[1]+=FacExcess(a,FlowActorCritic::tau95Ns)-FacExcess(b,FlowActorCritic::tau95Ns);
    h.pending[2]+=FacExcess(a,FlowActorCritic::tau99Ns)-FacExcess(b,FlowActorCritic::tau99Ns);
}
std::vector<uint8_t> FacEncode(uint64_t kind,uint32_t origin,uint64_t seq,const FacStats &s) {
    std::vector<uint8_t> b;auto add=[&b](uint64_t x){for(int i=7;i>=0;--i)b.push_back((x>>(8*i))&255);};
    add(0x46414331);add(kind);add(origin);add(seq);add(FlowActorCritic::startNs+(seq+1)*FlowActorCritic::periodNs);
    for(auto x:s)add(x);return b;
}
void FacHostTick(uint32_t node) {
    auto &h=facHosts.at(node);const int64_t now=Simulator::Now().GetNanoSeconds();
    for(const auto &f:h.active)FacAccumulate(h,f.second,now);
    h.pending[3]=h.active.size();h.pending[4]=h.pending[5]=0;
    for(const auto &f:h.active) {h.pending[4]+=now-f.second>int64_t(FlowActorCritic::tau95Ns);h.pending[5]+=now-f.second>int64_t(FlowActorCritic::tau99Ns);}
    facHostLog<<node<<','<<h.seq<<','<<now;for(auto x:h.pending)facHostLog<<','<<x;facHostLog<<'\n';
    for(unsigned i:{0u,1u,2u,6u,7u})h.total[i]+=h.pending[i];
    h.send(FacEncode(1,node,h.seq,h.pending));h.pending.fill(0);h.last=now;++h.seq;
    Simulator::Schedule(NanoSeconds(FlowActorCritic::periodNs),&FacHostTick,node);
}
FacX FacState(const FacLeaf &l) {
    FacX x{};x[0]=1;
    const double scales[8]={6400000000.,6400000000.,6400000000.,2048.,512.,128.,256.,256.};
    for(unsigned i=0;i<8;++i)x[i+1]=std::tanh(double(l.lastGlobal[i])/scales[i]);
    for(unsigned i=0;i<11;++i)x[i+9]=l.lastContext[i];
    return x;
}
double FacValue(const FacLeaf &l,const FacX &x) {return l.critic.Eval(x);}
void FacPublish(uint32_t node) {
    auto &l=facLeaves.at(node);if(!l.busy || l.collecting)NS_FATAL_ERROR("FAC publication state");
    double gamma=std::exp(-double(FlowActorCritic::periodNs)/200000000.);
    double ret=FacValue(l,l.endState);std::vector<double> returns(l.frames.size()),advantages(l.frames.size());
    int64_t latestReceipt=-1;uint64_t n=0;
    for(int i=int(l.frames.size())-1;i>=0;--i) {
        const auto &f=l.frames[i];if(f.version!=l.version)NS_FATAL_ERROR("FAC stale-policy gradient");
        const auto &s=l.rewards.at(f.seq);
        double cost=(s[0]+FlowActorCritic::lambda95*s[1]+FlowActorCritic::lambda99*s[2])/FacScale;
        if(FlowActorCritic::rewardMode==1)cost=s[9] ? double(s[8])/s[9]/1000000. : 0.;
        ret=-cost+gamma*ret;returns[i]=ret;advantages[i]=ret-f.value;n+=f.decisions;
        latestReceipt=std::max(latestReceipt,l.receiptNs.at(f.seq));
        facFrames<<node<<','<<l.blocks<<','<<f.seq<<','<<f.version<<','<<f.decisions<<','<<cost<<','<<f.value<<','<<ret<<','<<advantages[i]<<'\n';
    }
    double mean=0,variance=0;for(double a:advantages)mean+=a/advantages.size();
    for(double a:advantages)variance+=(a-mean)*(a-mean)/advantages.size();
    double sd=std::max(.01,std::sqrt(variance));FacW ag{},cg{};
    for(unsigned i=0;i<l.frames.size();++i) {
        const auto &f=l.frames[i];double adv=(advantages[i]-mean)/sd;
        for(unsigned k=0;k<FAC_P;++k)ag[k]+=adv*f.gradient[k]/std::max(uint64_t(1),n);
        FacW grad{};l.critic.Eval(f.state,&grad);
        // Huber critic gradient bounds individual sample influence, not predictions.
        double error=std::max(-10.,std::min(10.,returns[i]-f.value));
        for(unsigned k=0;k<FAC_P;++k)cg[k]+=error*grad[k]/l.frames.size();
    }
    double da=0,dc=0;
    if(FlowActorCritic::learn) {
        if(n)da=l.actor.Apply(ag,FlowActorCritic::actorStep,.10);
        dc=l.critic.Apply(cg,FlowActorCritic::criticStep,.50);++l.updates;++l.version;
        const int64_t now=Simulator::Now().GetNanoSeconds();if(l.firstUpdate<0)l.firstUpdate=now;l.lastUpdate=now;
    }
    const int64_t now=Simulator::Now().GetNanoSeconds();
    facUpdates<<node<<','<<l.blocks<<','<<l.blockStart<<','<<l.frames.back().seq<<','<<latestReceipt<<','<<now<<','<<l.version<<','<<n<<','<<da<<','<<dc;
    for(double x:l.actor.w)facUpdates<<','<<x;for(double x:l.critic.w)facUpdates<<','<<x;facUpdates<<'\n';
    for(const auto &f:l.frames) {l.rewards.erase(f.seq);l.receiptNs.erase(f.seq);}
    ++l.blocks;l.trainedDecisions+=n;l.frames.clear();l.busy=false;l.ready=true;
}
void FacMaybeTrain(uint32_t node) {
    auto &l=facLeaves.at(node);
    if(l.collecting || l.busy || l.frames.size()!=FlowActorCritic::rolloutWindows)return;
    for(const auto &f:l.frames)if(!l.rewards.count(f.seq))return;
    l.busy=true;Simulator::Schedule(NanoSeconds(FlowActorCritic::computeNs),&FacPublish,node);
}
void FacTick() {
    const int64_t now=Simulator::Now().GetNanoSeconds();uint64_t seq=FacSeq(now);
    for(auto &entry:facLeaves) {
        auto &l=entry.second;
        if(l.collecting && seq-l.blockStart==FlowActorCritic::rolloutWindows) {
            l.collecting=false;l.endState=FacState(l);FacMaybeTrain(entry.first);
        }
        if(l.ready && !l.busy && !l.collecting) {l.ready=false;l.collecting=true;l.blockStart=seq;}
        if(l.collecting) {
            FacFrame f;f.seq=seq;f.version=l.version;f.state=FacState(l);f.value=FacValue(l,f.state);l.frames.push_back(f);
        }
        // Reward-only gaps do not enter an on-policy rollout; raw reports remain in logs.
        while(!l.rewards.empty() && l.rewards.begin()->first<l.blockStart) {l.receiptNs.erase(l.rewards.begin()->first);l.rewards.erase(l.rewards.begin());}
    }
    Simulator::Schedule(NanoSeconds(FlowActorCritic::periodNs),&FacTick);
}
}
void FlowActorCritic::RegisterHost(uint32_t host,std::function<void(const std::vector<uint8_t>&)> send) {
    if(!enabled)return;facHosts[host].send=send;
}
void FlowActorCritic::Initialize() {
    if(!enabled)return;
    if(OnlineRouting::enabled || (RouteLearning::mode!=8 && RouteLearning::mode!=9) || RouteLearning::gateMode || RouteLearning::epsilon ||
       !periodNs || !computeNs || rolloutWindows<2 || actorStep<=0 || criticStep<=0 || temperature<=0 || tau95Ns>=tau99Ns || rewardMode>1)
        NS_FATAL_ERROR("Invalid FAC settings");
    auto open=[](std::ofstream &f,const char *suffix){f.open((RouteLearning::output+suffix).c_str());if(!f)NS_FATAL_ERROR("FAC audit open failure");f<<std::setprecision(17);};
    open(facEvents,".fac-events.csv");open(facHostLog,".fac-host.csv");open(facReceiveLog,".fac-receive.csv");open(facDecisions,".fac-decisions.csv");open(facUpdates,".fac-updates.csv");open(facFrames,".fac-frames.csv");
    facEvents<<"kind,host,flow,ns\n";facHostLog<<"host,seq,end_ns,area_ns,excess95_ns,excess99_ns,active,old95,old99,starts,finishes,ack_ns,acks\n";
    facReceiveLog<<"leaf,kind,origin,seq,end_ns,receive_ns,area_ns,excess95_ns,excess99_ns,active,old95,old99,starts,finishes,ack_ns,acks\n";
    facDecisions<<"node,key,ns,version,port,probability,logit,collecting,window\n";
    facFrames<<"node,block,window,version,decisions,cost,value,return,advantage\n";
    facUpdates<<"node,block,first_window,last_window,last_report_ns,publish_ns,version,decisions,actor_delta,critic_delta";
    for(unsigned i=0;i<FAC_P;++i)facUpdates<<",a"<<i;for(unsigned i=0;i<FAC_P;++i)facUpdates<<",c"<<i;facUpdates<<'\n';
    for(const auto &e:Settings::hostId2IpMap) {
        uint32_t leaf=Settings::hostIp2SwitchId.at(e.second);facHostLeaf[e.first]=leaf;++facExpectedHosts[leaf];
        if(!facHosts.count(e.first) || !facHosts.at(e.first).send)NS_FATAL_ERROR("Missing FAC host transport");
        facHosts[e.first].last=startNs;Simulator::Schedule(NanoSeconds(startNs+periodNs),&FacHostTick,e.first);
    }
    FacScale=facHosts.size()*50000000.;
    for(const auto &e:facExpectedHosts) {
        auto &l=facLeaves[e.first];l.rng.seed(seed+e.first*1009);l.actor.Init(l.rng);l.critic.Init(l.rng);l.ready=true;
    }
    Simulator::Schedule(NanoSeconds(startNs),&FacTick);
}
void FlowActorCritic::HostStart(uint32_t host,uint32_t flow,int64_t now) {
    if(!enabled)return;auto &h=facHosts.at(host);if(!h.active.emplace(flow,now).second)NS_FATAL_ERROR("FAC duplicate flow");
    ++h.pending[6];facEvents<<"start,"<<host<<','<<flow<<','<<now<<'\n';
}
void FlowActorCritic::HostComplete(uint32_t host,uint32_t flow,int64_t now) {
    if(!enabled)return;auto &h=facHosts.at(host);auto it=h.active.find(flow);if(it==h.active.end())NS_FATAL_ERROR("FAC missing flow");
    FacAccumulate(h,it->second,now);h.active.erase(it);++h.pending[7];facEvents<<"finish,"<<host<<','<<flow<<','<<now<<'\n';
}
std::vector<uint8_t> FlowActorCritic::Receive(uint32_t node,const std::vector<uint8_t>& b,int64_t now) {
    if(!enabled || b.size()!=120 || !facLeaves.count(node))NS_FATAL_ERROR("Invalid FAC report");
    unsigned offset=0;auto get=[&](){uint64_t x=0;for(unsigned i=0;i<8;++i)x=(x<<8)|b[offset++];return x;};
    if(get()!=0x46414331)NS_FATAL_ERROR("FAC wire magic");
    uint64_t kind=get();uint32_t origin=get();uint64_t seq=get(),end=get();FacStats s{};for(auto &x:s)x=get();
    if(end!=startNs+(seq+1)*periodNs || now<int64_t(end))NS_FATAL_ERROR("FAC future report");
    facReceiveLog<<node<<','<<kind<<','<<origin<<','<<seq<<','<<end<<','<<now;for(auto x:s)facReceiveLog<<','<<x;facReceiveLog<<'\n';
    auto &l=facLeaves.at(node);std::vector<uint8_t> broadcast;
    if(kind==1) {
        if(facHostLeaf.at(origin)!=node)NS_FATAL_ERROR("FAC wrong host destination");
        auto &pending=l.hostReports[seq];if(!pending.emplace(origin,s).second)NS_FATAL_ERROR("FAC duplicate host report");
        if(pending.size()!=facExpectedHosts.at(node))return {};
        FacStats sum{};for(const auto &h:pending)for(unsigned i=0;i<10;++i)sum[i]+=h.second[i];l.hostReports.erase(seq);
        auto a=l.ackStats.find(seq);if(a!=l.ackStats.end()){sum[8]=a->second[8];sum[9]=a->second[9];l.ackStats.erase(a);}
        broadcast=FacEncode(2,node,seq,sum);s=sum;origin=node;
    } else if(kind!=2)NS_FATAL_ERROR("FAC unknown report kind");
    auto &pending=l.leafReports[seq];if(!facExpectedHosts.count(origin) || !pending.emplace(origin,s).second)NS_FATAL_ERROR("FAC duplicate/unknown leaf report");
    if(pending.size()==facLeaves.size()) {
        FacStats sum{};for(const auto &r:pending)for(unsigned i=0;i<10;++i)sum[i]+=r.second[i];
        l.rewards[seq]=sum;l.receiptNs[seq]=now;l.lastGlobal=sum;l.leafReports.erase(seq);FacMaybeTrain(node);
    }
    return broadcast;
}
unsigned FlowActorCritic::Select(uint32_t node,uint64_t key,int64_t now,const std::vector<RouteLearning::State>& states,
    const std::vector<uint32_t>& ports,RouteLearning::GateContext context,int previous,std::vector<double>& scores) {
    auto &l=facLeaves.at(node);auto &flow=l.flows[key];if(flow.first<0)flow.first=now;
    std::vector<FacW> gradients;std::vector<FacX> features;std::vector<double> logits,probs;double maximum=-1e100;
    for(unsigned i=0;i<states.size();++i) {
        const auto &s=states[i];auto &res=l.reservations[ports[i]];
        double recent=res.first ? res.second*std::exp(-(now-res.first)/200000.) : 0.;double reserved=recent*s[2];
        FacX x{{1.,std::tanh(s[0]/.05),std::tanh(s[1]/.05),std::tanh(s[8]/.05),std::tanh(s[7]/.05),
            std::min(2.,s[4]/.2),std::tanh(s[5]/.05),std::min(2.,s[2]/.016),std::min(2.,s[3]/.016),
            std::tanh(reserved/.05),std::tanh(s[18]/.05),s[14],std::tanh((now-flow.first)/50000000.),
            std::min(2.,std::log1p(context[0]/64000.)/4.),std::min(2.,std::log1p(context[1]/8000.)/4.),context[3],
            std::tanh(context[2]/1000000.),double(int(i)==previous),std::tanh(context[5]/100000.),std::tanh(l.lastGlobal[3]/2048.)}};
        double base=.008+s[0]+.25*s[1]+.1*s[8]+reserved;FacW gradient{};
        double logit=-temperature*std::log(base/.02)+l.actor.Eval(x,&gradient);
        if(!std::isfinite(logit))NS_FATAL_ERROR("FAC nonfinite logit");
        gradients.push_back(gradient);features.push_back(x);logits.push_back(logit);maximum=std::max(maximum,logit);
    }
    double sum=0;for(double v:logits){probs.push_back(std::exp(v-maximum));sum+=probs.back();}
    for(double &p:probs)p/=sum;
    const double epsilon=.01;std::vector<double> mixed=probs;for(double &p:mixed)p=(1-epsilon)*p+epsilon/probs.size();
    double draw=std::generate_canonical<double,53>(l.rng),cdf=0;unsigned chosen=probs.size()-1;
    for(unsigned i=0;i<probs.size();++i){cdf+=mixed[i];if(draw<cdf){chosen=i;break;}}
    if(l.collecting) {
        if(l.frames.empty() || l.frames.back().seq!=FacSeq(now) || l.frames.back().version!=l.version)NS_FATAL_ERROR("FAC rollout version mismatch");
        auto &f=l.frames.back();double factor=(1-epsilon)*probs[chosen]/mixed[chosen];
        for(unsigned k=0;k<FAC_P;++k) {double expected=0;for(unsigned i=0;i<probs.size();++i)expected+=probs[i]*gradients[i][k];f.gradient[k]+=factor*(gradients[chosen][k]-expected);}
        ++f.decisions;
    }
    for(unsigned k=0;k<FAC_D;++k)l.lastContext[k]=.9*l.lastContext[k]+.1*features[chosen][k];
    if(!flow.port || flow.port!=ports[chosen]) {
        auto &res=l.reservations[ports[chosen]];res.second=(res.first ? res.second*std::exp(-(now-res.first)/200000.) : 0.)+1;res.first=now;
    }
    flow.port=ports[chosen];scores.clear();for(double v:logits)scores.push_back(-v);
    ++l.decisions;l.lastDecision=now;
    facDecisions<<node<<','<<key<<','<<now<<','<<l.version<<','<<ports[chosen]<<','<<mixed[chosen]<<','<<logits[chosen]<<','<<l.collecting<<','<<FacSeq(now)<<'\n';
    return chosen;
}
void FlowActorCritic::Data(uint32_t node,uint64_t key,uint32_t port,uint64_t seq,uint64_t end,int64_t now) {
    if(!enabled)return;auto &l=facLeaves.at(node);auto it=l.flows.find(key);if(it==l.flows.end())return;
    auto &f=it->second;if(seq<f.maxEnd)return;f.maxEnd=std::max(f.maxEnd,end);
    if(f.pending || (f.lastProbe>=0 && now-f.lastProbe<50000))return;
    f.pending=true;f.lastProbe=now;f.sent=now;f.end=end;
}
void FlowActorCritic::Ack(uint32_t node,uint64_t key,uint64_t seq,int64_t now) {
    if(!enabled)return;auto &l=facLeaves.at(node);auto it=l.flows.find(key);if(it==l.flows.end())return;
    auto &f=it->second;if(!f.pending || seq<f.end)return;f.pending=false;
    auto &s=l.ackStats[FacSeq(now)];s[8]+=now-f.sent;++s[9];
}
void FlowActorCritic::CountWire(uint64_t bytes) {++facWirePackets;facWireBytes+=bytes;}
void FlowActorCritic::Finish() {
    if(!enabled)return;
    facEvents.close();facHostLog.close();facReceiveLog.close();facDecisions.close();facUpdates.close();facFrames.close();
    std::ofstream out((RouteLearning::output+".fac-audit.json").c_str());out<<std::setprecision(17);
    FacStats total{};uint64_t active=0;for(const auto &h:facHosts) {for(unsigned i:{0u,1u,2u,6u,7u})total[i]+=h.second.total[i]+h.second.pending[i];active+=h.second.active.size();}
    out<<"{\"learn\":"<<(learn ? "true" : "false")<<",\"wire_packets\":"<<facWirePackets<<",\"wire_bytes\":"<<facWireBytes<<",\"active\":"<<active
       <<",\"area_ns\":"<<total[0]<<",\"excess95_ns\":"<<total[1]<<",\"excess99_ns\":"<<total[2]<<",\"starts\":"<<total[6]<<",\"finishes\":"<<total[7]<<",\"leaves\":[";
    bool first=true;for(const auto &e:facLeaves) {const auto &l=e.second;double norm=0;for(double x:l.actor.w)norm+=x*x;
        out<<(first ? "" : ",")<<"{\"node\":"<<e.first<<",\"updates\":"<<l.updates<<",\"blocks\":"<<l.blocks<<",\"decisions\":"<<l.decisions<<",\"trained_decisions\":"<<l.trainedDecisions
           <<",\"first_update_ns\":"<<l.firstUpdate<<",\"last_update_ns\":"<<l.lastUpdate<<",\"last_decision_ns\":"<<l.lastDecision<<",\"actor_norm\":"<<std::sqrt(norm)<<"}";first=false;}
    out<<"]}\n";
}
