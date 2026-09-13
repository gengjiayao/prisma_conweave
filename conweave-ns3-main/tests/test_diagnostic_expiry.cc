#include "../src/opengym/model/diagnostic-expiry.h"
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

struct State { double ts = 0; uint32_t count = 0; };

int main() {
    std::unordered_map<uint64_t, State> reference, optimized;
    double oldest = std::numeric_limits<double>::infinity();
    uint64_t referenceScans = 0, optimizedScans = 0;
    auto event = [&](uint64_t key, double now) {
        if (reference.size() > 10000) {
            ++referenceScans;
            for (auto it = reference.begin(); it != reference.end();) {
                if (now - it->second.ts > .1) it = reference.erase(it);
                else ++it;
            }
        }
        optimizedScans += ns3::PruneDiagnosticFlowSequences(optimized, now, .1, oldest);
        auto& a = reference[key]; auto& b = optimized[key];
        assert(a.ts == b.ts && a.count == b.count);
        a.ts = b.ts = now; ++a.count; ++b.count;
        oldest = std::min(oldest, now);
        assert(reference.size() == optimized.size());
    };
    // Exceed the 10,000-flow threshold without any expired records, then
    // repeatedly refresh entries, including the entry that set the bound.
    for (uint64_t k = 0; k < 12050; ++k) event(k, 0.0);
    for (uint64_t k = 0; k < 5000; ++k) event((k * 37) % 12050, .05);
    assert(optimizedScans == 0);
    event(0, .1);  // Strict comparison: a record at zero is not expired yet.
    assert(optimizedScans == 0);
    event(1, std::nextafter(.1, 1.0));
    // Cross the size threshold again; expire and reinsert old flow identities.
    for (uint64_t k = 20000; k < 32050; ++k) event(k, .12);
    event(0, .16);
    event(20000, .23);
    for (const auto& kv : reference) {
        const auto& b = optimized.at(kv.first);
        assert(kv.second.ts == b.ts && kv.second.count == b.count);
    }
    assert(optimizedScans < 10 && referenceScans > 10000);
    std::cout << "Exact diagnostic state preserved; scans " << referenceScans
              << " -> " << optimizedScans << "\n";
}
