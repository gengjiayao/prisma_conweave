// Included once inside namespace ns3 by switch-node.cc.
bool RouteLearning::pressureEnabled = false, RouteLearning::pressureMask = false;
unsigned RouteLearning::mode = 0;
unsigned RouteLearning::pathReselect = 1;
unsigned RouteLearning::gateMode = 0;
std::string RouteLearning::gateModel;
double RouteLearning::gateThresholdMs = 0;
double RouteLearning::gateTailThresholdMs = 0;
unsigned RouteLearning::migrationAdmissionPpm = 1000000, RouteLearning::migrationAdmissionSeed = 901;
uint64_t RouteLearning::intervalNs = 50000, RouteLearning::marginNs = 50000;
unsigned RouteLearning::seed = 301;
double RouteLearning::epsilon = 0;
bool RouteLearning::maskHistory = false;
bool RouteLearning::maskTransport = false;
int64_t RouteLearning::overrideFlow = -1;
unsigned RouteLearning::overrideStep = 0, RouteLearning::overrideAction = 0;
std::string RouteLearning::model, RouteLearning::output;
namespace {
// Stable nested flow cohorts for controlled intervention; never consumes routeRng.
uint32_t MigrationCohort(uint64_t flow) {
    uint32_t x = uint32_t(flow) ^ uint32_t(flow >> 32) ^ uint32_t(RouteLearning::migrationAdmissionSeed);
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
struct RouteTransition {
    RouteLearning::State state;
    int64_t now;
    unsigned action, step;
};
std::unordered_map<uint32_t, RouteTransition> routePending;
std::unordered_map<uint32_t, unsigned> routeSteps;
std::unordered_set<uint32_t> routeCompleted;
std::mt19937 routeRng;
std::ofstream routeLog;
std::ofstream routeCandidateLog;
std::ofstream routeBoundaryLog;
std::ofstream routeGateLog, routePressureLog;
unsigned gateInputs=24,gateHidden=32;
double gateW1[32][48],gateB1[48],gateW2[48][2],gateB2[2];
double routeW1[RouteLearning::INPUTS][RouteLearning::HIDDEN];
double routeB1[RouteLearning::HIDDEN];
double routeW2[RouteLearning::HIDDEN][2], routeB2[2];
uint64_t routeActions[2] = {0,0}, routeTerminals = 0, routeTransitions = 0, routeOverrides = 0;
uint64_t routeLateActions = 0;
uint64_t routePathAssignments[8] = {};
uint64_t routePathReselections = 0, routePathChanges = 0;
void WriteRouteTransition(uint32_t id, int64_t end, bool done, const RouteLearning::State &next) {
    auto it = routePending.find(id);
    if (it == routePending.end()) return;
    const auto &t = it->second;
    if (end < t.now) NS_FATAL_ERROR("Negative routing transition duration");
    ++routeTransitions;
    if (done) ++routeTerminals;
    if (routeLog.is_open()) {
        routeLog << id << ',' << t.step << ',' << t.now << ',' << end << ','
                 << t.action << ',' << done << ',' << (end-t.now)*1e-6;
        for (double x : t.state) routeLog << ',' << x;
        for (double x : next) routeLog << ',' << x;
        routeLog << '\n';
    }
}
}
void RouteLearning::RecordPressure(char kind, std::initializer_list<uint64_t> fields) {
    if (!routePressureLog.is_open()) NS_FATAL_ERROR("Pressure event without initialized output");
    routePressureLog << kind;
    for (uint64_t value:fields) routePressureLog << ',' << value;
    routePressureLog << '\n';
}
void RouteLearning::Initialize() {
    if (!mode) return;
    if (mode > 9 || !intervalNs || epsilon < 0 || epsilon > 1 || output.empty() || pathReselect>3)
        NS_FATAL_ERROR("Invalid route learning configuration");
    if (gateMode>3 || (gateMode && mode!=9) || !std::isfinite(gateThresholdMs) ||
        !std::isfinite(gateTailThresholdMs) || migrationAdmissionPpm>1000000)
        NS_FATAL_ERROR("Invalid route gate configuration");
    if (pressureEnabled) {
        if (mode<8 || !RoutingDiagnostics::feedback) NS_FATAL_ERROR("Pressure observations require report-based path learning");
        routePressureLog.open((output+".pressure.csv").c_str());
        if (!routePressureLog) NS_FATAL_ERROR("Cannot open pressure observations");
        routePressureLog << "kind,ns,node,port,key,values...\n";
    }
    routeRng.seed(seed);
    routeLog.open(output.c_str());
    if (!routeLog) NS_FATAL_ERROR("Cannot open routing transitions");
    routeLog << std::setprecision(12);
    routeLog << "flow,step,start_ns,end_ns,action,done,cost_ms";
    for (unsigned i=0;i<INPUTS;++i) routeLog << ",x" << i;
    for (unsigned i=0;i<INPUTS;++i) routeLog << ",next" << i;
    routeLog << '\n';
    if (mode >= 8) {
        if (maskHistory || maskTransport) NS_FATAL_ERROR("Switch feature masks do not apply to path inputs");
        routeCandidateLog.open((output+".candidates.csv").c_str());
        if (!routeCandidateLog) NS_FATAL_ERROR("Cannot open initial-path candidates");
        routeCandidateLog << std::setprecision(12) << "flow,start_ns,candidate,selected,score";
        for (unsigned i=0;i<INPUTS;++i) routeCandidateLog << ",x" << i;
        routeCandidateLog << '\n';
    }
    if (mode == 9) {
        routeBoundaryLog.open((output+".boundaries.csv").c_str());
        if (!routeBoundaryLog) NS_FATAL_ERROR("Cannot open path boundaries");
        routeBoundaryLog << "flow,start_ns,sent_end,cumulative_ack,sequence,gap_ns,previous_port,selected_port\n";
    }
    if (gateMode) {
        routeGateLog.open((output+".gate.csv").c_str());
        if (!routeGateLog) NS_FATAL_ERROR("Cannot open route gate observations");
        routeGateLog << std::setprecision(12) << "flow,step,start_ns,previous,candidate,selected";
        routeGateLog << (gateMode==3 ? ",pred_total_advantage_ms,pred_p99_advantage_ms" : ",q_hold,q_switch");
        for (unsigned i=0;i<(pressureEnabled ? 32u : INPUTS);++i) routeGateLog << ",g" << i;
        for (unsigned i=0;i<7;++i) routeGateLog << ",context" << i;
        routeGateLog << '\n';
    }
    if (gateMode>=2) {
        std::ifstream in(gateModel.c_str()); std::string magic; unsigned inputs=0, hidden=0;
        in >> magic >> inputs >> hidden;
        if (!in || !((magic=="ROUTE_GATE_V1" && inputs==24 && hidden==32) ||
            (magic=="ROUTE_GATE_V2" && inputs==32 && hidden==48 && pressureEnabled)))
            NS_FATAL_ERROR("Invalid route gate model header");
        gateInputs=inputs;gateHidden=hidden;
        for(unsigned i=0;i<inputs;++i) for(unsigned h=0;h<hidden;++h) in >> gateW1[i][h];
        for(unsigned h=0;h<hidden;++h) in >> gateB1[h];
        for(unsigned h=0;h<hidden;++h) for(unsigned a=0;a<2;++a) in >> gateW2[h][a];
        for(double &x:gateB2) in >> x;
        if (!in) NS_FATAL_ERROR("Truncated route gate model");
        for(auto &row:gateW1) for(double x:row) if(!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite gate weight");
        for(double x:gateB1) if(!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite gate bias");
        for(auto &row:gateW2) for(double x:row) if(!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite gate weight");
        for(double x:gateB2) if(!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite gate bias");
    }
    if (mode == 6 || mode >= 8) {
        std::ifstream in(model.c_str()); std::string magic; unsigned inputs=0, hidden=0;
        in >> magic >> inputs >> hidden;
        if (!in || magic != "ROUTE_DQN_V1" || inputs != INPUTS || hidden != HIDDEN)
            NS_FATAL_ERROR("Invalid route DQN header");
        for (auto &row : routeW1) for (double &x : row) in >> x;
        for (double &x : routeB1) in >> x;
        for (auto &row : routeW2) for (double &x : row) in >> x;
        for (double &x : routeB2) in >> x;
        if (!in) NS_FATAL_ERROR("Truncated route DQN model");
        for (auto &row : routeW1) for (double x : row) if (!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite DQN weight");
        for (double x : routeB1) if (!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite DQN bias");
        for (auto &row : routeW2) for (double x : row) if (!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite DQN weight");
        for (double x : routeB2) if (!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite DQN bias");
    }
}
unsigned RouteLearning::ChoosePath(uint32_t flowId, int64_t now, const std::vector<State> &states,
                                   int previousCandidate, GateContext context, const std::vector<PressureState> &pressure, uint64_t flowKey) {
    if (mode<8 || states.empty() || states.size()>8 || (mode==8 && routePending.count(flowId)) || routeCompleted.count(flowId))
        NS_FATAL_ERROR("Invalid initial-path decision");
    std::vector<double> scores;
    unsigned chosen=0; double best=std::numeric_limits<double>::infinity();
    for (unsigned candidate=0;candidate<states.size();++candidate) {
        const State &state=states[candidate];
        for (double x:state) if (!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite path input");
        double score=routeB2[1]-routeB2[0];
        for(unsigned h=0;h<HIDDEN;++h) {
            double v=routeB1[h];
            for(unsigned i=0;i<INPUTS;++i) v+=state[i]*routeW1[i][h];
            score+=std::max(0.,v)*(routeW2[h][1]-routeW2[h][0]);
        }
        if (!std::isfinite(score)) NS_FATAL_ERROR("Nonfinite path score");
        scores.push_back(score);
        // Fixed hash order resolves numerical ties in equivalent delay sums.
        if (score<best-1e-12) {chosen=candidate;best=score;}
    }
    const unsigned proposal=chosen;
    std::array<double,32> gateState{}; double gateQ[2]={0,0};
    if (previousCandidate>=0) {
        if (unsigned(previousCandidate)>=states.size()) NS_FATAL_ERROR("Invalid previous path");
        if (scores[previousCandidate]-scores[chosen] <= marginNs*1e-6) chosen=previousCandidate;
        if (gateMode) {
            const auto &old=states[previousCandidate]; const auto &next=states[proposal];
            gateState=std::array<double,32>{{(scores[previousCandidate]-scores[proposal])/.2,
                old[0]/.2,next[0]/.2,old[1]/.2,next[1]/.2,old[4]/.2,next[4]/.2,
                old[5]/.2,next[5]/.2,old[6]/.2,next[6]/.2,old[7]/.2,next[7]/.2,
                old[8]/.2,next[8]/.2,(old[2]+old[3])/.032,(next[2]+next[3])/.032,
                std::log1p(context[0]/64000.),context[1]/8000.,context[2]/200000.,context[3],
                std::log1p(context[4]/200000.),context[5]/50000.,context[6]}};
            if (pressureEnabled && !pressureMask) {
                if (pressure.size()!=states.size()) NS_FATAL_ERROR("Missing per-path pressure observations");
                for(unsigned i=0;i<4;++i) {
                    gateState[24+2*i]=pressure[previousCandidate][i];
                    gateState[25+2*i]=pressure[proposal][i];
                }
            }
            for(double &x:gateState) {
                if(!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite gate observation");
                x=std::max(-10.,std::min(10.,x));
            }
            gateQ[0]=scores[previousCandidate]-scores[proposal];gateQ[1]=marginNs*1e-6;
            if (gateMode>=2) {
                gateQ[0]=gateB2[0];gateQ[1]=gateB2[1];
                for(unsigned h=0;h<gateHidden;++h) {
                    double v=gateB1[h];
                    for(unsigned i=0;i<gateInputs;++i) v+=gateState[i]*gateW1[i][h];
                    for(unsigned a=0;a<2;++a) gateQ[a]+=std::max(0.,v)*gateW2[h][a];
                }
                if(!std::isfinite(gateQ[0]) || !std::isfinite(gateQ[1])) NS_FATAL_ERROR("Nonfinite gate value");
                const bool allow = gateMode==3 ?
                    (gateQ[0]>gateThresholdMs && gateQ[1]>gateTailThresholdMs) :
                    gateQ[0]-gateQ[1]>gateThresholdMs;
                chosen=allow ? proposal : unsigned(previousCandidate);
            }
        }
    }
    if (epsilon && std::generate_canonical<double,53>(routeRng)<epsilon)
        chosen=std::uniform_int_distribution<unsigned>(0,states.size()-1)(routeRng);
    unsigned step=mode==9 ? routeSteps[flowId]++ : 0;
    if (mode==9 && previousCandidate>=0 &&
        uint64_t(MigrationCohort(flowKey))*1000000 >= (uint64_t(migrationAdmissionPpm)<<32))
        chosen=unsigned(previousCandidate);
    if (int64_t(flowId)==overrideFlow && overrideStep==step) {
        if (overrideAction>=states.size()) NS_FATAL_ERROR("Invalid path override");
        chosen=overrideAction;++routeOverrides;
    }
    if (gateMode && previousCandidate>=0) {
        routeGateLog << flowId << ',' << step << ',' << now << ',' << previousCandidate << ',' << proposal << ',' << chosen << ',' << gateQ[0] << ',' << gateQ[1];
        for(unsigned i=0;i<(pressureEnabled ? 32u : INPUTS);++i) routeGateLog << ',' << gateState[i];
        for(double x:context) routeGateLog << ',' << x;
        routeGateLog << '\n';
    }
    for (unsigned candidate=0;candidate<states.size();++candidate) {
        routeCandidateLog << flowId << ',' << now << ',' << candidate << ',' << (candidate==chosen) << ',' << scores[candidate];
        for (double x:states[candidate]) routeCandidateLog << ',' << x;
        routeCandidateLog << '\n';
    }
    if (mode==9) {
        WriteRouteTransition(flowId,now,false,states[chosen]);
        if (previousCandidate>=0) {++routePathReselections;if(chosen!=unsigned(previousCandidate)) ++routePathChanges;}
    }
    routePending[flowId]=RouteTransition{states[chosen],now,chosen,step};++routePathAssignments[chosen];
    return chosen;
}
void RouteLearning::RecordPathBoundary(uint32_t flowId, int64_t now, uint64_t sentEnd,
                                       uint64_t cumulativeAck, uint64_t sequence, uint64_t gapNs,
                                       uint32_t previousPort, uint32_t selectedPort) {
    if (mode!=9) return;
    routeBoundaryLog << flowId << ',' << now << ',' << sentEnd << ',' << cumulativeAck << ','
                     << sequence << ',' << gapNs << ',' << previousPort << ',' << selectedPort << '\n';
}
unsigned RouteLearning::Decide(uint32_t flowId, int64_t now, State state,
                               double slack, uint64_t age, double trend) {
    if (maskHistory) for (unsigned i : {8u,9u,13u,14u,15u}) state[i]=0;
    if (maskTransport) for(unsigned i=16;i<INPUTS;++i) state[i]=0;
    for (double x : state) if (!std::isfinite(x)) NS_FATAL_ERROR("Nonfinite route observation");
    WriteRouteTransition(flowId,now,false,state);
    unsigned action=0;
    if (mode==2) action=1;
    if (mode==3) action=slack >= double(std::max(marginNs,age));
    if (mode==4) action=slack >= double(marginNs);
    // An observable hand rule: rising old-path delay permits an aggressive switch.
    if (mode==5) action=(trend>0.15 && state[4]>0.15) || slack>=double(std::max(marginNs,age));
    if (mode==7) action=slack>=double(std::max(uint64_t(50000),age)) ||
        (state[23]>0 && state[18]<std::log1p(double(marginNs)/16000.)/std::log(1001.) && state[4]>.15 && state[21]<.1);
    if (mode==6) {
        double q[2]={routeB2[0],routeB2[1]};
        for(unsigned h=0;h<HIDDEN;++h) {
            double v=routeB1[h];
            for(unsigned i=0;i<INPUTS;++i) v+=state[i]*routeW1[i][h];
            v=std::max(0.0,v);
            for(unsigned a=0;a<2;++a) q[a]+=v*routeW2[h][a];
        }
        if (!std::isfinite(q[0]) || !std::isfinite(q[1])) NS_FATAL_ERROR("Nonfinite route Q value");
        action=q[1]<q[0]; // Q estimates completion cost, so minimize.
    }
    if (epsilon && std::generate_canonical<double,53>(routeRng)<epsilon)
        action=std::uniform_int_distribution<unsigned>(0,1)(routeRng);
    unsigned step=routeSteps[flowId]++;
    if (int64_t(flowId)==overrideFlow && step==overrideStep) {action=overrideAction; ++routeOverrides;}
    if (action>1) NS_FATAL_ERROR("Invalid route action");
    // Late duplicate packets still follow the policy, but a completed application
    // must never create a second learning episode. This flag is not an input.
    if (!routeCompleted.count(flowId)) {
        routePending[flowId]=RouteTransition{state,now,action,step}; ++routeActions[action];
    } else ++routeLateActions;
    return action;
}
void RouteLearning::Complete(uint32_t flowId, int64_t now) {
    if (!mode) return;
    State terminal{};
    WriteRouteTransition(flowId,now,true,terminal);
    routePending.erase(flowId); routeSteps.erase(flowId);
    routeCompleted.insert(flowId);
}
void RouteLearning::Finish(int64_t now) {
    if (!mode) return;
    routeLog.close();
    routeCandidateLog.close();
    routeBoundaryLog.close();
    routeGateLog.close();
    routePressureLog.close();
    std::ofstream out((output+".audit.json").c_str());
    out << "{\"hold\":" << routeActions[0] << ",\"switch\":" << routeActions[1]
        << ",\"transitions\":" << routeTransitions << ",\"terminals\":" << routeTerminals
        << ",\"unfinished_episodes\":" << routePending.size() << ",\"overrides\":" << routeOverrides
        << ",\"late_packet_actions\":" << routeLateActions
        << ",\"stop_ns\":" << now;
    if (mode>=8) {
        out << ",\"path_assignments\":[";
        for(unsigned i=0;i<8;++i) out << (i ? "," : "") << routePathAssignments[i];
        out << ']';
    }
    if (mode==9) out << ",\"path_reselections\":" << routePathReselections << ",\"path_changes\":" << routePathChanges;
    out << "}\n";
}
