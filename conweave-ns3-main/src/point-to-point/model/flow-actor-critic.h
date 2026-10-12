#ifndef FLOW_ACTOR_CRITIC_H
#define FLOW_ACTOR_CRITIC_H
#include "route-learning.h"
#include <functional>
#include <map>
namespace ns3 {
// Cooperative online A2C. Learners only consume reports delivered as packets.
struct FlowActorCritic {
    static const uint8_t PROTOCOL = 0xf8;
    static bool enabled, learn;
    static unsigned seed, rolloutWindows, rewardMode;
    static uint64_t startNs, periodNs, computeNs, tau95Ns, tau99Ns;
    static double actorStep, criticStep, temperature, lambda95, lambda99;
    static void Initialize();
    static void RegisterHost(uint32_t host, std::function<void(const std::vector<uint8_t>&)> send);
    static void HostStart(uint32_t host, uint32_t flow, int64_t now);
    static void HostComplete(uint32_t host, uint32_t flow, int64_t now);
    static std::vector<uint8_t> Receive(uint32_t leaf, const std::vector<uint8_t>& bytes, int64_t now);
    static unsigned Select(uint32_t node, uint64_t key, int64_t now,
        const std::vector<RouteLearning::State>& states, const std::vector<uint32_t>& ports,
        RouteLearning::GateContext context, int previous, std::vector<double>& scores);
    static void Data(uint32_t node,uint64_t key,uint32_t port,uint64_t seq,uint64_t end,int64_t now);
    static void Ack(uint32_t node,uint64_t key,uint64_t seq,int64_t now);
    static void CountWire(uint64_t bytes);
    static void Finish();
};
}
#endif
