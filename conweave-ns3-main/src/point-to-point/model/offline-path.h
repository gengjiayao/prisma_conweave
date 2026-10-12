#ifndef OFFLINE_PATH_H
#define OFFLINE_PATH_H
#include "route-learning.h"
namespace ns3 {
// Frozen inference with local assignment accounting. No completion labels/updates.
struct OfflinePath {
    static bool enabled, reservations, spread;
    static uint64_t decayNs;
    static double nearMs;
    static unsigned seed;
    static void Initialize();
    static void Features(uint32_t node,uint32_t port,uint64_t prefix,int64_t now,RouteLearning::State &state);
    static unsigned Select(uint32_t node,uint64_t key,uint32_t flow,int64_t now,
        const std::vector<RouteLearning::State>& states,const std::vector<uint32_t>& ports,
        const std::vector<uint64_t>& prefixes,const std::vector<double>& scores,unsigned greedy);
    static void Finish();
};
}
#endif
