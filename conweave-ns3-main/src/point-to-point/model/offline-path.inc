// Included once inside ns3. All state belongs to the observing source leaf.
bool OfflinePath::enabled=false, OfflinePath::reservations=true, OfflinePath::spread=true;
uint64_t OfflinePath::decayNs=1000000;
double OfflinePath::nearMs=.016;
unsigned OfflinePath::seed=98101;
namespace {
struct OfflineCount {int64_t ns=-1;double value=0;};
struct OfflineLeaf {
    std::map<std::pair<uint32_t,uint64_t>,OfflineCount> destination;
    std::map<uint32_t,OfflineCount> port;
    uint64_t decisions=0,dispersed=0;
};
std::map<uint32_t,OfflineLeaf> offlineLeaves;
std::ofstream offlineLog;
double OfflineRead(const OfflineCount &c,int64_t now) {
    if(c.ns<0)return 0;
    if(now<c.ns)NS_FATAL_ERROR("Noncausal offline reservation read");
    return c.value*std::exp(-double(now-c.ns)/OfflinePath::decayNs);
}
void OfflineAdd(OfflineCount &c,int64_t now) {c.value=OfflineRead(c,now)+1.;c.ns=now;}
uint64_t OfflineHash(uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
}
void OfflinePath::Initialize() {
    if(!enabled)return;
    if(RouteLearning::mode!=8 || RouteLearning::epsilon || OnlineRouting::enabled || FlowActorCritic::enabled ||
       !decayNs || !std::isfinite(nearMs) || nearMs<0)
        NS_FATAL_ERROR("Offline path accounting requires a frozen initial-path actor");
    offlineLog.open((RouteLearning::output+".offline.csv").c_str());
    if(!offlineLog)NS_FATAL_ERROR("Cannot open offline path audit");
    offlineLog<<std::setprecision(17)<<"node,key,flow,ns,candidate,port,prefix,greedy,selected,eligible,probability,uniform,score,port_reservation_ms,destination_reservation_ms\n";
}
void OfflinePath::Features(uint32_t node,uint32_t port,uint64_t prefix,int64_t now,RouteLearning::State &s) {
    if(!enabled)return;
    auto &leaf=offlineLeaves[node];
    s[22]=reservations ? OfflineRead(leaf.port[port],now)*s[2] : 0.;
    s[23]=reservations ? OfflineRead(leaf.destination[{port,prefix}],now)*s[3] : 0.;
}
unsigned OfflinePath::Select(uint32_t node,uint64_t key,uint32_t flow,int64_t now,
    const std::vector<RouteLearning::State>& states,const std::vector<uint32_t>& ports,
    const std::vector<uint64_t>& prefixes,const std::vector<double>& scores,unsigned greedy) {
    if(states.empty() || states.size()!=ports.size() || states.size()!=prefixes.size() || greedy>=states.size())
        NS_FATAL_ERROR("Invalid offline candidate set");
    auto &leaf=offlineLeaves[node];std::vector<double> probability(states.size(),0.);double sum=0;
    const double uniform=double(OfflineHash(key^(uint64_t(node)<<32)^uint64_t(seed))>>11)/9007199254740992.;
    if(spread) {
        for(unsigned i=0;i<states.size();++i)if(scores[i]<=scores[greedy]+nearMs+1e-12) {
            if(states[i][2]<=0)NS_FATAL_ERROR("Missing local link service time");
            probability[i]=1./states[i][2];sum+=probability[i];
        }
        for(double &v:probability)v/=sum;
    } else probability[greedy]=1.;
    unsigned chosen=greedy;double cumulative=0;
    for(unsigned i=0;i<states.size();++i) {cumulative+=probability[i];if(uniform<cumulative){chosen=i;break;}}
    if(probability[chosen]<=0)NS_FATAL_ERROR("Invalid offline sampled action");
    for(unsigned i=0;i<states.size();++i)
        offlineLog<<node<<','<<key<<','<<flow<<','<<now<<','<<i<<','<<ports[i]<<','<<prefixes[i]<<','<<greedy<<','<<(i==chosen)<<','<<(probability[i]>0)<<','<<probability[i]<<','<<uniform<<','<<scores[i]<<','<<states[i][22]<<','<<states[i][23]<<'\n';
    // Update only after every candidate's observation and probability is frozen.
    if(reservations) {
        OfflineAdd(leaf.port[ports[chosen]],now);
        OfflineAdd(leaf.destination[{ports[chosen],prefixes[chosen]}],now);
    }
    ++leaf.decisions;if(chosen!=greedy)++leaf.dispersed;
    return chosen;
}
void OfflinePath::Finish() {
    if(!enabled)return;offlineLog.close();
    std::ofstream out((RouteLearning::output+".offline-audit.json").c_str());
    out<<"{\"enabled\":true,\"parameter_updates\":0,\"reservations\":"<<reservations<<",\"spread\":"<<spread
       <<",\"decay_ns\":"<<decayNs<<",\"near_ms\":"<<nearMs<<",\"seed\":"<<seed<<",\"leaves\":[";
    bool first=true;for(const auto &p:offlineLeaves) {
        if(!first)out<<',';first=false;
        out<<"{\"node\":"<<p.first<<",\"decisions\":"<<p.second.decisions<<",\"dispersed\":"<<p.second.dispersed
           <<",\"destination_entries\":"<<p.second.destination.size()<<",\"port_entries\":"<<p.second.port.size()<<'}';
    }
    out<<"]}\n";
}
