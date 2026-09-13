#ifndef CAPACITY_WEIGHTED_ECMP_H
#define CAPACITY_WEIGHTED_ECMP_H
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace ns3 {
// Stable per-flow mapping. Use the entire 32-bit hash range even when the
// sum of link rates exceeds 2^32; hash % total_bps would omit slower ports.
inline std::size_t CapacityWeightedIndex(uint32_t hash,
                                         const std::vector<uint64_t>& rates) {
    long double total = 0;
    for (uint64_t rate : rates) total += rate;
    if (total <= 0) throw std::invalid_argument("No positive ECMP link capacity");
    const long double ticket = (static_cast<long double>(hash) / 4294967296.0L) * total;
    long double cumulative = 0;
    for (std::size_t i = 0; i < rates.size(); ++i) {
        cumulative += rates[i];
        if (ticket < cumulative) return i;
    }
    throw std::logic_error("Weighted ECMP ticket outside capacity range");
}
}
#endif
