// Completion-triggered communication DAGs. Included after StartFlowInput.
// A node is a network transfer; dependencies wait for sender-confirmed QP completion.
#include <set>

struct CollectiveTransfer {
    FlowInput flow;
    uint64_t releaseNs, delayNs;
    uint32_t remaining;
    std::vector<uint32_t> children;
    bool launched = false, finished = false;
};
static std::vector<CollectiveTransfer> collective_transfers;

static void LaunchCollective(uint32_t id) {
    auto &x = collective_transfers.at(id);
    NS_ABORT_MSG_IF(x.launched || x.remaining, "Invalid collective launch");
    x.launched = true;
    x.flow.start_time = Simulator::Now().GetSeconds();
    StartFlowInput(x.flow, nullptr);
}

static void ScheduleCollective(uint32_t id) {
    auto &x = collective_transfers.at(id);
    uint64_t now = Simulator::Now().GetNanoSeconds();
    uint64_t when = std::max(x.releaseNs, now + x.delayNs);
    Simulator::Schedule(NanoSeconds(when - now), &LaunchCollective, id);
}

static void CollectiveComplete(uint32_t id) {
    auto &x = collective_transfers.at(id);
    NS_ABORT_MSG_IF(!x.launched || x.finished, "Invalid collective completion");
    x.finished = true;
    collective_events << "finish " << id << " " << Simulator::Now().GetNanoSeconds() << "\n";
    for (uint32_t child : x.children) {
        auto &y = collective_transfers.at(child);
        NS_ABORT_MSG_IF(!y.remaining, "Duplicate collective dependency");
        if (--y.remaining == 0) ScheduleCollective(child);
    }
}

static void InitializeCollective() {
    std::ifstream plan(collective_plan_file.c_str());
    uint32_t count = 0;
    plan >> count;
    NS_ABORT_MSG_IF(!plan || count != flow_num || !count, "Invalid collective plan size");
    collective_transfers.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id = 0, ndep = 0;
        auto &x = collective_transfers[i];
        plan >> id >> x.flow.src >> x.flow.dst >> x.flow.pg >> x.flow.maxPacketCount
             >> x.releaseNs >> x.delayNs >> ndep;
        NS_ABORT_MSG_IF(!plan || id != i || !x.flow.maxPacketCount || x.flow.src == x.flow.dst,
                        "Invalid collective transfer");
        NS_ABORT_MSG_IF(x.flow.src >= Settings::host_num || x.flow.dst >= Settings::host_num,
                        "Invalid collective host");
        x.flow.idx = id;
        x.remaining = ndep;
        std::set<uint32_t> parents;
        for (uint32_t j = 0; j < ndep; ++j) {
            uint32_t parent = 0;
            plan >> parent;
            NS_ABORT_MSG_IF(!plan || parent >= id || !parents.insert(parent).second,
                            "Collective plan must be a topologically sorted DAG");
            collective_transfers[parent].children.push_back(id);
        }
    }
    std::string extra;
    NS_ABORT_MSG_IF(plan >> extra, "Trailing data in collective plan");
    NS_ABORT_MSG_IF(collective_event_file.empty(), "Collective audit path is required");
    collective_events.open(collective_event_file.c_str());
    NS_ABORT_MSG_IF(!collective_events, "Cannot write collective audit");
    flowf.close();
    for (uint32_t i = 0; i < count; ++i)
        if (!collective_transfers[i].remaining) ScheduleCollective(i);
}
