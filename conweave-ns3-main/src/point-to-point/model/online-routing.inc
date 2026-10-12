// Included once inside ns3, after route-learning.inc.
bool OnlineRouting::enabled=false, OnlineRouting::learn=true;
double OnlineRouting::step=.02, OnlineRouting::exploration=.02, OnlineRouting::reservationWeight=1.;
unsigned OnlineRouting::seed=9401;
uint64_t OnlineRouting::updateDelayNs=20000, OnlineRouting::sampleIntervalNs=50000;
namespace {
const unsigned ONLINE_D=12;
using OnlineX=std::array<double,ONLINE_D>;
struct OnlineFlow {
    OnlineX x{};
    double base=.008;
    uint32_t port=0;
    uint64_t maxEnd=0, probeEnd=0;
    int64_t sent=-1,lastProbe=-1;
    bool pending=false,decided=false;
};
struct OnlineLeaf {
    double w[3][ONLINE_D]{};
    std::map<uint64_t,OnlineFlow> flows;
    std::map<uint32_t,std::pair<int64_t,double>> reservations;
    std::mt19937 rng;
    bool seeded=false;
    uint64_t updates=0,decisions=0, samples=0, retransmissions=0,skipped=0,explorations=0;
    int64_t lastQueued=-1,firstUpdate=-1,lastUpdate=-1,lastDecision=-1;
};
std::map<uint32_t,OnlineLeaf> onlineLeaves;
std::ofstream onlineUpdates,onlineDecisions;
double OnlineDot(const double *w,const OnlineX& x) {
    double out=0;for(unsigned i=0;i<ONLINE_D;++i)out+=w[i]*x[i];return out;
}
double OnlineClip(double x,double bound){return std::max(-bound,std::min(bound,x));}
struct OnlineUpdateBatch {uint32_t node;uint64_t key;uint32_t port;int64_t ackNs;OnlineX x;double target,rttMs;};
void OnlineUpdate(OnlineUpdateBatch b) {
    const auto node=b.node;const auto key=b.key;const auto port=b.port;const auto ackNs=b.ackNs;
    const auto x=b.x;const double target=b.target,rttMs=b.rttMs;
    auto &leaf=onlineLeaves[node];const double tau[3]={.5,.95,.99};
    double norm=1,delta=0;for(double v:x)norm+=v*v;
    for(unsigned q=0;q<3;++q) {
        const double prediction=OnlineDot(leaf.w[q],x),error=target-prediction;
        // Smooth pinball loss; fixed forgetting keeps adaptation active.
        const double gradient=(error>=0 ? tau[q] : tau[q]-1.)*std::min(1.,std::abs(error)/.5);
        for(unsigned i=0;i<ONLINE_D;++i) {
            const double old=leaf.w[q][i];
            leaf.w[q][i]=OnlineClip(old*(1.-OnlineRouting::step*.001)+OnlineRouting::step*gradient*x[i]/norm,2.);
            delta+=(leaf.w[q][i]-old)*(leaf.w[q][i]-old);
        }
    }
    const int64_t now=Simulator::Now().GetNanoSeconds();++leaf.updates;
    if(leaf.firstUpdate<0)leaf.firstUpdate=now;leaf.lastUpdate=now;
    onlineUpdates<<node<<','<<key<<','<<port<<','<<ackNs<<','<<now<<','<<leaf.updates<<','
        <<rttMs<<','<<target<<','<<std::sqrt(delta);
    for(const auto &head:leaf.w)for(double w:head)onlineUpdates<<','<<w;
    onlineUpdates<<'\n';
}
}
void OnlineRouting::Initialize() {
    if(!enabled)return;
    if(RouteLearning::mode!=9 || step<=0 || step>1 || exploration<0 || exploration>1 ||
       !updateDelayNs || !sampleIntervalNs || reservationWeight<0)
        NS_FATAL_ERROR("Invalid online-routing settings");
    onlineUpdates.open((RouteLearning::output+".online-updates.csv").c_str());
    onlineDecisions.open((RouteLearning::output+".online-decisions.csv").c_str());
    if(!onlineUpdates || !onlineDecisions)NS_FATAL_ERROR("Cannot open online audit logs");
    onlineUpdates<<std::setprecision(12)<<"node,key,port,ack_ns,update_ns,updates,rtt_ms,target,delta_norm";
    for(unsigned q=0;q<3;++q)for(unsigned i=0;i<ONLINE_D;++i)onlineUpdates<<",w"<<q<<'_'<<i;
    onlineUpdates<<'\n';
    onlineDecisions<<std::setprecision(12)<<"node,key,ns,updates,port,score,explored\n";
}
unsigned OnlineRouting::Select(uint32_t node,uint64_t key,int64_t now,
    const std::vector<RouteLearning::State>& states,const std::vector<uint32_t>& ports,std::vector<double>& actualScores) {
    auto &leaf=onlineLeaves[node];
    if(!leaf.seeded){leaf.rng.seed(seed+node*1009);leaf.seeded=true;}
    if(states.empty() || states.size()!=ports.size())NS_FATAL_ERROR("Bad online candidate set");
    std::vector<OnlineX> features;std::vector<double> bases,scores;
    unsigned best=0;
    for(unsigned i=0;i<states.size();++i) {
        const auto &s=states[i];auto &reservation=leaf.reservations[ports[i]];
        const double recent=reservation.first ? reservation.second*std::exp(-(now-reservation.first)/200000.) : 0.;
        const double reserved=recent*s[2];
        OnlineX x{{1.,std::tanh(s[0]/.05),std::tanh(s[1]/.05),std::tanh(s[8]/.05),
            std::tanh(s[7]/.05),std::min(2.,s[4]/.2),std::tanh(s[5]/.05),
            std::min(2.,s[2]/.016),std::min(2.,s[3]/.016),std::tanh(reserved/.05),
            std::tanh(s[18]/.05),s[14]}};
        // Explicit fixed initialization, shared by the learning-off control.
        const double base=.008+s[0]+.25*s[1]+.1*s[8]+reservationWeight*reserved;
        double q[3];for(unsigned h=0;h<3;++h)q[h]=OnlineClip(OnlineDot(leaf.w[h],x),1.5);
        q[1]=std::max(q[0],q[1]);q[2]=std::max(q[1],q[2]);
        double score=base*std::exp(.2*q[0]+.4*q[1]+.4*q[2]);
        if(!std::isfinite(score) || score<=0)NS_FATAL_ERROR("Nonfinite online score");
        features.push_back(x);bases.push_back(base);scores.push_back(score);
        if(score<scores[best]-1e-12)best=i;
    }
    bool explore=false;
    if(exploration && std::generate_canonical<double,53>(leaf.rng)<exploration) {
        std::vector<unsigned> near;for(unsigned i=0;i<scores.size();++i)if(scores[i]<=scores[best]*1.25)near.push_back(i);
        best=near[std::uniform_int_distribution<unsigned>(0,near.size()-1)(leaf.rng)];explore=true;++leaf.explorations;
    }
    actualScores=scores;
    auto &flow=leaf.flows[key];
    if(!flow.decided || flow.port!=ports[best]) {
        auto &v=leaf.reservations[ports[best]];
        v.second=(v.first ? v.second*std::exp(-(now-v.first)/200000.) : 0.)+1.;v.first=now;
    }
    // Pending probes retain their own immutable decision context in the map below.
    flow.x=features[best];flow.base=bases[best];flow.port=ports[best];flow.decided=true;
    ++leaf.decisions;leaf.lastDecision=now;
    onlineDecisions<<node<<','<<key<<','<<now<<','<<leaf.updates<<','<<ports[best]<<','<<scores[best]<<','<<explore<<'\n';
    return best;
}
namespace {
struct OnlineProbe { OnlineX x;double base;uint32_t port;int64_t sent;uint64_t end; };
std::map<std::pair<uint32_t,uint64_t>,OnlineProbe> onlineProbes;
}
void OnlineRouting::Data(uint32_t node,uint64_t key,uint32_t port,uint64_t sequence,uint64_t end,int64_t now) {
    if(!enabled)return;auto &leaf=onlineLeaves[node];auto it=leaf.flows.find(key);
    if(it==leaf.flows.end())return;auto &f=it->second;
    if(sequence<f.maxEnd){++leaf.retransmissions;return;}f.maxEnd=std::max(f.maxEnd,end);
    if(f.pending || (f.lastProbe>=0 && now-f.lastProbe<int64_t(sampleIntervalNs)))return;
    f.pending=true;f.lastProbe=now;
    onlineProbes[{node,key}]=OnlineProbe{f.x,f.base,port,now,end};
}
void OnlineRouting::Ack(uint32_t node,uint64_t key,uint64_t sequence,int64_t now) {
    if(!enabled)return;auto p=onlineProbes.find({node,key});
    if(p==onlineProbes.end() || sequence<p->second.end)return;
    const auto probe=p->second;onlineProbes.erase(p);auto &leaf=onlineLeaves[node];leaf.flows[key].pending=false;++leaf.samples;
    if(now<=probe.sent)NS_FATAL_ERROR("Noncausal online ACK sample");
    const double rtt=(now-probe.sent)*1e-6;
    const double target=OnlineClip(std::log(std::max(.001,rtt)/probe.base),4.);
    if(!learn)return;
    if(leaf.lastQueued>=0 && now-leaf.lastQueued<int64_t(updateDelayNs)){++leaf.skipped;return;}
    leaf.lastQueued=now;
    Simulator::Schedule(NanoSeconds(updateDelayNs),&OnlineUpdate,OnlineUpdateBatch{node,key,probe.port,now,probe.x,target,rtt});
}
void OnlineRouting::Finish() {
    if(!enabled)return;onlineUpdates.close();onlineDecisions.close();
    std::ofstream out((RouteLearning::output+".online-audit.json").c_str());
    out<<"{\"learn\":"<<(learn ? "true" : "false")<<",\"leaves\":[";bool first=true;
    for(const auto &entry:onlineLeaves){const auto &l=entry.second;double norm=0;
        for(const auto &h:l.w)for(double w:h)norm+=w*w;
        out<<(first ? "" : ",")<<"{\"node\":"<<entry.first<<",\"updates\":"<<l.updates
           <<",\"decisions\":"<<l.decisions<<",\"samples\":"<<l.samples<<",\"skipped\":"<<l.skipped
           <<",\"explorations\":"<<l.explorations<<",\"first_update_ns\":"<<l.firstUpdate
           <<",\"last_update_ns\":"<<l.lastUpdate<<",\"last_decision_ns\":"<<l.lastDecision
           <<",\"weight_norm\":"<<std::sqrt(norm)<<"}";first=false;}
    out<<"]}\n";
}
