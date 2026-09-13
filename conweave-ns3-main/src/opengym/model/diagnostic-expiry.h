#ifndef NS3_DIAGNOSTIC_EXPIRY_H
#define NS3_DIAGNOSTIC_EXPIRY_H

#include <algorithm>
#include <limits>

namespace ns3 {

// oldestTimestamp is a lower bound on every stored timestamp. Refreshing an
// entry can leave that bound conservatively old, but never invalid. A scan is
// needed only if the original strict TTL predicate could remove an entry.
template <typename Map>
bool PruneDiagnosticFlowSequences(Map& flows, double now, double ttl,
                                 double& oldestTimestamp) {
    if (flows.size() <= 10000 || !(now - oldestTimestamp > ttl)) {
        return false;
    }
    oldestTimestamp = std::numeric_limits<double>::infinity();
    for (auto it = flows.begin(); it != flows.end();) {
        if (now - it->second.ts > ttl) {
            it = flows.erase(it);
        } else {
            oldestTimestamp = std::min(oldestTimestamp, it->second.ts);
            ++it;
        }
    }
    return true;
}

}  // namespace ns3

#endif
