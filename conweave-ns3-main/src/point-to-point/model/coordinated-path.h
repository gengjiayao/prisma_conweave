#ifndef COORDINATED_PATH_H
#define COORDINATED_PATH_H
#include "route-learning.h"
namespace ns3 {
struct CoordinatedPath {
    static bool enabled, balance, work, conditional, packetLog;
    static unsigned seed;
    static std::string model;
    static void Initialize();
    static void Prepare(uint32_t node,uint32_t destination,int64_t now);
    static void Features(uint32_t node,uint32_t port,uint64_t prefix,int64_t now,RouteLearning::State &state);
    static unsigned Select(uint32_t node,uint64_t key,uint32_t flow,int64_t now,
        const std::vector<RouteLearning::State>& states,const std::vector<uint32_t>& ports,
        const std::vector<uint64_t>& prefixes,std::vector<double>& scores);
    static void Data(uint32_t node,uint64_t key,uint32_t port,uint64_t end,int64_t now);
    static void Finish();
};
}
#endif
