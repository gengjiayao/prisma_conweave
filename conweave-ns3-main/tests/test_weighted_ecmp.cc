#include "../src/point-to-point/model/weighted-ecmp.h"
#include <cassert>
#include <limits>

int main() {
    using ns3::CapacityWeightedIndex;
    std::vector<uint64_t> rates{1000000000,1000000000,500000000,500000000};
    assert(CapacityWeightedIndex(0, rates) == 0);
    assert(CapacityWeightedIndex(std::numeric_limits<uint32_t>::max(), rates) == 3);
    std::vector<uint64_t> count(4,0);
    for (uint64_t i=0; i<120000; ++i)
        ++count[CapacityWeightedIndex(static_cast<uint32_t>(i*4294967296ULL/120000),rates)];
    for (int i=0; i<4; ++i) {
        const uint64_t expected = i<2 ? 40000 : 20000;
        assert(count[i]+1 >= expected && count[i] <= expected+1);
    }
    assert(CapacityWeightedIndex(0,{0,100000000000ULL,0})==1);
    assert(CapacityWeightedIndex(UINT32_MAX,{0,100000000000ULL,0})==1);
    bool rejected=false;
    try { CapacityWeightedIndex(42,{0,0}); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
}
