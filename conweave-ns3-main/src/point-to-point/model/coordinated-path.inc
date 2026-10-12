// Frozen conditional inference. Every mutable counter is local, causal observation.
bool CoordinatedPath::enabled=false,CoordinatedPath::balance=true,CoordinatedPath::work=true;
bool CoordinatedPath::conditional=false,CoordinatedPath::packetLog=false;
unsigned CoordinatedPath::seed=99101;
std::string CoordinatedPath::model;
namespace {
const double CoordDebtNs=20000000.,CoordCountNs=1000000.;
const int64_t CoordIdleNs=2000000;
double coordTheta[8]={.25,.5,.05,.1,.016,16.,.2,1.};
struct CoordCell {
    int64_t ns=-1;double count=0,slots=0,debt=0;
    void Age(int64_t now) {
        if(ns>=0){if(now<ns)NS_FATAL_ERROR("Noncausal coordinated counter");
            count*=std::exp(-double(now-ns)/CoordCountNs);
            double d=std::exp(-double(now-ns)/CoordDebtNs);slots*=d;debt*=d;}
        ns=now;
    }
};
struct CoordFlow {
    uint32_t port=0,destination=0;uint64_t end=0;int64_t last=-1,debtNs=-1;
    double debt=0;bool sampled=false;
};
struct CoordLeaf {
    std::map<std::pair<uint32_t,uint32_t>,CoordCell> destinations;
    std::map<uint32_t,CoordCell> ports;
    std::map<uint64_t,CoordFlow> flows;
    std::set<std::pair<int64_t,uint64_t> > idle;
    uint32_t destination=0;int64_t now=-1,arrivalNs=-1;
    double arrivals=0,estimate=65536.;
    uint64_t decisions=0,samples=0,packets=0,bytes=0,departures=0;
};
std::map<uint32_t,CoordLeaf> coordLeaves;
std::ofstream coordLog,coordPackets;
uint64_t CoordHash(uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
double CoordUnit(uint64_t x){return double(CoordHash(x)>>11)/9007199254740992.;}
double CoordRead(CoordCell &c,int64_t now,unsigned field) {
    c.Age(now);return field==0?c.count:field==1?c.slots:c.debt;
}
double CoordDestination(CoordLeaf &leaf,uint32_t port,uint64_t prefix,int64_t now,unsigned field) {
    unsigned width=prefix&255;uint32_t first=uint32_t(prefix>>8);
    if(width>32)NS_FATAL_ERROR("Invalid received prefix");
    uint32_t mask=width ? uint32_t(0xffffffffu<<(32-width)):0;
    uint32_t last=first|~mask;double value=0;
    auto it=leaf.destinations.lower_bound({port,first});
    for(;it!=leaf.destinations.end() && it->first.first==port && it->first.second<=last;++it)
        value+=CoordRead(it->second,now,field);
    return value;
}
}
void CoordinatedPath::Initialize() {
    if(!enabled)return;
    if(RouteLearning::mode!=8 || OfflinePath::enabled || OnlineRouting::enabled || FlowActorCritic::enabled || RouteLearning::epsilon)
        NS_FATAL_ERROR("Coordinated paths require frozen initial-flow inference");
    if(!model.empty()){
        std::ifstream in(model.c_str());std::string magic;in>>magic;
        if(magic!="COORDINATED_PATH_V1")NS_FATAL_ERROR("Invalid coordinated model");
        for(double &x:coordTheta)if(!(in>>x)||!std::isfinite(x))NS_FATAL_ERROR("Invalid coordinated coefficient");
        std::string extra;if(in>>extra)NS_FATAL_ERROR("Extra coordinated coefficients");
    }
    for(double x:coordTheta)if(x<0)NS_FATAL_ERROR("Negative coordinated coefficient");
    if(coordTheta[5]<=0||coordTheta[6]<=0||coordTheta[7]<=0)NS_FATAL_ERROR("Invalid coordinated normalization");
    coordLog.open((RouteLearning::output+".coordinated.csv").c_str());
    if(!coordLog)NS_FATAL_ERROR("Cannot open coordinated audit");
    coordLog<<std::setprecision(17)<<"node,key,flow,ns,destination,candidate,port,prefix,selected,greedy,eligible,score,fair,phase,slots,capacity,burst,gate,estimate,prediction,port_reservation_ms,destination_reservation_ms\n";
    if(packetLog){coordPackets.open((RouteLearning::output+".coordinated-events.csv").c_str());
        if(!coordPackets)NS_FATAL_ERROR("Cannot open causal packet log");
        coordPackets<<std::setprecision(17)<<"kind,ns,node,key,port,destination,end,delta,estimate,prediction\n";}
}
void CoordinatedPath::Prepare(uint32_t node,uint32_t destination,int64_t now) {
    if(!enabled)return;auto &leaf=coordLeaves[node];
    while(!leaf.idle.empty() && leaf.idle.begin()->first<=now){
        uint64_t key=leaf.idle.begin()->second;leaf.idle.erase(leaf.idle.begin());auto &f=leaf.flows.at(key);
        if(f.sampled||now-f.last<CoordIdleNs)NS_FATAL_ERROR("Invalid causal idle sample");
        // A quiescent observed span is an estimate, not a true completion label.
        leaf.estimate=.875*leaf.estimate+.125*std::max(1024.,std::min(1048576.,double(f.end)));
        f.sampled=true;++leaf.samples;
        if(packetLog)coordPackets<<"I,"<<now<<','<<node<<','<<key<<','<<f.port<<','<<f.destination<<','<<f.end<<",0,"<<leaf.estimate<<",0\n";
    }
    leaf.destination=destination;leaf.now=now;
    if(leaf.arrivalNs>=0)leaf.arrivals*=std::exp(-double(now-leaf.arrivalNs)/200000.);
    leaf.arrivalNs=now;
}
void CoordinatedPath::Features(uint32_t node,uint32_t port,uint64_t prefix,int64_t now,RouteLearning::State &s) {
    if(!enabled)return;auto &leaf=coordLeaves[node];
    if(leaf.now!=now)NS_FATAL_ERROR("Missing coordinated context");
    unsigned field=work?2:0;double scale=work?1./65536.:1.;
    s[22]=CoordRead(leaf.ports[port],now,field)*scale*s[2];
    s[23]=CoordDestination(leaf,port,prefix,now,field)*scale*s[3];
}
unsigned CoordinatedPath::Select(uint32_t node,uint64_t key,uint32_t flow,int64_t now,
    const std::vector<RouteLearning::State>& states,const std::vector<uint32_t>& ports,
    const std::vector<uint64_t>& prefixes,std::vector<double>& scores) {
    auto &leaf=coordLeaves[node];if(leaf.now!=now||leaf.flows.count(key))NS_FATAL_ERROR("Repeated initial coordinated action");
    const double burst=leaf.arrivals,gate=conditional?burst/(burst+coordTheta[5]):1.;
    const double prediction=std::max(1024.,std::min(4194304.,leaf.estimate*coordTheta[7]));
    std::vector<uint32_t> stable=ports;std::sort(stable.begin(),stable.end());
    std::vector<double> fair(states.size()),slots(states.size()),phase(states.size()),capacity(states.size());
    unsigned greedy=0;
    for(unsigned i=0;i<states.size();++i){
        const auto &s=states[i];double trust=conditional?1./(1.+s[4]/coordTheta[6])*(1.-.75*gate):1.;
        double reserve=conditional?.25+.75*gate:1.;
        scores[i]=s[0]+coordTheta[0]*trust*s[1]+coordTheta[1]*reserve*s[23]+coordTheta[2]*reserve*s[22]+coordTheta[3]*s[8];
        if(scores[i]<scores[greedy]-1e-12)greedy=i;
        capacity[i]=1./std::max(s[2],s[3]);
        if(!std::isfinite(capacity[i])||capacity[i]<=0)NS_FATAL_ERROR("Missing coordinated capacity");
        slots[i]=CoordDestination(leaf,ports[i],prefixes[i],now,1);
        unsigned rank=std::lower_bound(stable.begin(),stable.end(),ports[i])-stable.begin();
        unsigned offset=CoordHash((uint64_t(node)<<32)^prefixes[i]^uint64_t(seed))%stable.size();
        phase[i]=double((rank+offset)%stable.size())/stable.size();
        fair[i]=(slots[i]+1.+phase[i])/capacity[i];
    }
    unsigned chosen=greedy;
    if(balance){bool first=true;
        for(unsigned i=0;i<states.size();++i)if(scores[i]<=scores[greedy]+coordTheta[4]*gate+1e-12){
            if(first||fair[i]<fair[chosen]-1e-12||(std::abs(fair[i]-fair[chosen])<=1e-12&&ports[i]<ports[chosen])){chosen=i;first=false;}
        }
    }
    for(unsigned i=0;i<states.size();++i)
        coordLog<<node<<','<<key<<','<<flow<<','<<now<<','<<leaf.destination<<','<<i<<','<<ports[i]<<','<<prefixes[i]<<','<<(i==chosen)<<','<<greedy<<','<<(scores[i]<=scores[greedy]+coordTheta[4]*gate+1e-12)<<','<<scores[i]<<','<<fair[i]<<','<<phase[i]<<','<<slots[i]<<','<<capacity[i]<<','<<burst<<','<<gate<<','<<leaf.estimate<<','<<prediction<<','<<states[i][22]<<','<<states[i][23]<<'\n';
    CoordFlow f;f.port=ports[chosen];f.destination=leaf.destination;f.debt=prediction;f.debtNs=now;
    leaf.flows[key]=f;
    for(CoordCell *c:{&leaf.ports[f.port],&leaf.destinations[{f.port,f.destination}]}){c->Age(now);c->count+=1.;c->slots+=1.;c->debt+=prediction;}
    leaf.arrivals+=1.;++leaf.decisions;leaf.departures+=chosen!=greedy;
    if(packetLog)coordPackets<<"A,"<<now<<','<<node<<','<<key<<','<<f.port<<','<<f.destination<<",0,0,"<<leaf.estimate<<','<<prediction<<'\n';
    return chosen;
}
void CoordinatedPath::Data(uint32_t node,uint64_t key,uint32_t port,uint64_t end,int64_t now) {
    if(!enabled)return;auto li=coordLeaves.find(node);if(li==coordLeaves.end())NS_FATAL_ERROR("Data without source context");
    auto &leaf=li->second;auto fi=leaf.flows.find(key);if(fi==leaf.flows.end())NS_FATAL_ERROR("Data without initial assignment");
    auto &f=fi->second;if(f.port!=port)NS_FATAL_ERROR("Unexpected coordinated migration");
    uint64_t delta=end>f.end?end-f.end:0;f.end=std::max(f.end,end);
    f.debt*=std::exp(-double(now-f.debtNs)/CoordDebtNs);f.debtNs=now;
    double consumed=std::min(f.debt,double(delta));f.debt-=consumed;
    for(CoordCell *c:{&leaf.ports[port],&leaf.destinations[{port,f.destination}]}){c->Age(now);c->debt-=consumed;if(c->debt< -1e-6)NS_FATAL_ERROR("Negative causal debt");c->debt=std::max(0.,c->debt);}
    if(!f.sampled){if(f.last>=0)leaf.idle.erase({f.last+CoordIdleNs,key});leaf.idle.insert({now+CoordIdleNs,key});}
    f.last=now;++leaf.packets;leaf.bytes+=delta;
    if(packetLog)coordPackets<<"D,"<<now<<','<<node<<','<<key<<','<<port<<','<<f.destination<<','<<end<<','<<delta<<','<<leaf.estimate<<",0\n";
}
void CoordinatedPath::Finish() {
    if(!enabled)return;coordLog.close();if(packetLog)coordPackets.close();
    std::ofstream out((RouteLearning::output+".coordinated-audit.json").c_str());
    out<<std::setprecision(17)<<"{\"parameter_updates\":0,\"balance\":"<<balance<<",\"work\":"<<work<<",\"conditional\":"<<conditional<<",\"packet_log\":"<<packetLog<<",\"seed\":"<<seed<<",\"theta\":[";
    for(unsigned i=0;i<8;++i)out<<(i?",":"")<<coordTheta[i];out<<"],\"leaves\":[";bool first=true;
    for(const auto &v:coordLeaves){if(!first)out<<',';first=false;const auto &l=v.second;
        out<<"{\"node\":"<<v.first<<",\"decisions\":"<<l.decisions<<",\"non_greedy\":"<<l.departures<<",\"idle_samples\":"<<l.samples<<",\"observed_packets\":"<<l.packets<<",\"observed_unique_span_bytes\":"<<l.bytes<<",\"estimated_bytes\":"<<l.estimate<<",\"flow_entries\":"<<l.flows.size()<<",\"destination_entries\":"<<l.destinations.size()<<'}';}
    out<<"]}\n";
}
