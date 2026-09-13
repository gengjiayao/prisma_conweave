#ifndef ROUTE_LEARNING_H
#define ROUTE_LEARNING_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <initializer_list>

namespace ns3 {
// Opt-in research controller. Training records are never actor observations.
struct RouteLearning {
    static const unsigned INPUTS = 24, HIDDEN = 32;
    using State = std::array<double, INPUTS>;
    using GateContext = std::array<double, 7>;
    using PressureState = std::array<double, 4>;
    static bool pressureEnabled, pressureMask;
    static void RecordPressure(char kind, std::initializer_list<uint64_t> fields);
    static unsigned mode; // 0 off, 1-7 switch, 8 initial path, 9 boundary-based path RL
    static unsigned pathReselect; // mode 9: 0 hold, 1 drained, 2 gap, 3 one bounded-outstanding migration
    static unsigned gateMode; // 0 legacy, 1 log fixed, 2 scalar value, 3 separate total-FCT/P99 values
    static std::string gateModel;
    static double gateThresholdMs;
    static double gateTailThresholdMs;
    static unsigned migrationAdmissionPpm, migrationAdmissionSeed;
    static uint64_t intervalNs, marginNs;
    static unsigned seed;
    static double epsilon;
    static bool maskHistory, maskTransport;
    static int64_t overrideFlow;
    static unsigned overrideStep, overrideAction;
    static std::string model, output;
    static void Initialize();
    static unsigned Decide(uint32_t flowId, int64_t now, State state,
                           double arrivalSlackNs, uint64_t ageNs, double trend);
    static void Complete(uint32_t flowId, int64_t now);
    static unsigned ChoosePath(uint32_t flowId, int64_t now, const std::vector<State> &states,
                               int previousCandidate = -1, GateContext context = {},
                               const std::vector<PressureState> &pressure = {}, uint64_t flowKey = 0);
    static void RecordPathBoundary(uint32_t flowId, int64_t now, uint64_t sentEnd,
                                   uint64_t cumulativeAck, uint64_t sequence, uint64_t gapNs,
                                   uint32_t previousPort, uint32_t selectedPort);
    static void Finish(int64_t now);
};
}
#endif
