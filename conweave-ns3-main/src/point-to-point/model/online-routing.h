#ifndef ONLINE_ROUTING_H
#define ONLINE_ROUTING_H
#include "route-learning.h"
namespace ns3 {
// Local, ACK-trained risk-sensitive contextual bandit. No simulator FCT labels.
struct OnlineRouting {
    static bool enabled, learn;
    static double step, exploration, reservationWeight;
    static unsigned seed;
    static uint64_t updateDelayNs, sampleIntervalNs;
    static void Initialize();
    static unsigned Select(uint32_t node, uint64_t key, int64_t now,
        const std::vector<RouteLearning::State>& states, const std::vector<uint32_t>& ports,std::vector<double>& actualScores);
    static void Data(uint32_t node, uint64_t key, uint32_t port, uint64_t sequence,
                     uint64_t endSequence, int64_t now);
    static void Ack(uint32_t node, uint64_t key, uint64_t sequence, int64_t now);
    static void Finish();
};
}
#endif
