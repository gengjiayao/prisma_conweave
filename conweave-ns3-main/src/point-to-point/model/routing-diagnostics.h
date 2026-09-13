#ifndef ROUTING_DIAGNOSTICS_H
#define ROUTING_DIAGNOSTICS_H

#include <cstdint>
#include <string>

namespace ns3 {
// Experimental interventions; defaults preserve the original simulator.
// feedback=true uses decoded wire reports. Explicit oracle interventions are
// restricted to non-RL diagnostic runs.
struct RoutingDiagnostics {
    static bool enabled;
    static bool remote;
    static bool oooCnp;
    static uint64_t delayNs;
    static uint64_t gapNs;
    static uint64_t sampleNs;
    static uint64_t backgroundPeriodNs;
    static uint32_t backgroundMode; // 0=none, 1=persistent, 2=rotating per flow
    static bool feedback;
    static bool guard;
    static uint64_t feedbackPeriodNs;
    static uint64_t guardMarginNs;
    static uint32_t reportQuantumBytes; // zero=full 32-bit queue/leaf pairs
    static bool ageGuard;
    static bool paddedReports;
    static bool congaAckFeedback;
    static uint64_t congaAckUpdates;
    static uint64_t congaBackgroundBytes;
    static double startSeconds;
    static std::string output;
    static std::string backgroundPathsFile;
    static void LoadBackgroundPaths();
    static unsigned Group(uint32_t sourceIp);
    static void Receive(uint32_t sourceIp, uint32_t seq, uint32_t expected,
                        uint32_t size, bool ecn, bool oooCnp);
    static void Write();
};
}
#endif
