#ifndef ROUTE_FEEDBACK_H
#define ROUTE_FEEDBACK_H

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

namespace ns3 {

// Wire payload of IPv4 protocol 0xf9. No packet tags or simulator objects.
// All integers use network byte order. The sender clock is used only for
// differences between samples; freshness uses the receiver's own clock.
struct RouteFeedback {
    static const uint8_t PROTOCOL = 0xf9;
    static const unsigned MAX_ENTRIES = 64;
    struct Entry {
        uint32_t network = 0;
        uint8_t prefix = 32;
        uint32_t queueBytes = 0;
        uint64_t rateBps = 0;
        uint32_t prefixBytes = 0, prefixFlows = 0;
        uint32_t Mask() const { return prefix == 0 ? 0 : uint32_t(0xffffffffu << (32-prefix)); }
        bool Contains(uint32_t ip) const { return (ip & Mask()) == network; }
    };
    uint64_t sequence = 0, sampleNs = 0;
    bool pressure = false;
    std::vector<Entry> entries;

    bool Valid() const {
        if (!sequence || entries.empty() || entries.size() > MAX_ENTRIES) return false;
        for (size_t i=0; i<entries.size(); ++i) {
            const auto &e=entries[i];
            if (!e.prefix || e.prefix>32 || !e.rateBps || (e.network & e.Mask())!=e.network) return false;
            for (size_t j=0; j<i; ++j)
                if (entries[j].Contains(e.network) || e.Contains(entries[j].network)) return false;
        }
        return true;
    }
    std::vector<uint8_t> Encode() const {
        if (!Valid()) throw std::invalid_argument("Invalid route feedback");
        std::vector<uint8_t> bytes;
        auto put=[&bytes](uint64_t value, unsigned width) {
            for (unsigned i=width; i; --i) bytes.push_back(uint8_t(value >> (8*(i-1))));
        };
        put(0x52514642,4); put(1,1); put(pressure ? 1 : 0,1); put(entries.size(),2);
        put(sequence,8); put(sampleNs,8);
        for (const auto &e:entries) {
            put(e.network,4); put(e.prefix,1); put(e.queueBytes,4); put(e.rateBps,8);
            if (pressure) { put(e.prefixBytes,4); put(e.prefixFlows,4); }
        }
        return bytes;
    }
    static bool Decode(const std::vector<uint8_t> &bytes, RouteFeedback &out) {
        if (bytes.size()<24) return false;
        size_t pos=0;
        auto get=[&bytes,&pos](unsigned width) {
            uint64_t value=0;
            for (unsigned i=0; i<width; ++i) value=(value<<8)|bytes[pos++];
            return value;
        };
        if (get(4)!=0x52514642 || get(1)!=1) return false;
        const unsigned flags=get(1), count=get(2);
        if (flags>1 || !count || count>MAX_ENTRIES || bytes.size()!=24+count*(flags ? 25u : 17u)) return false;
        RouteFeedback decoded;
        decoded.pressure=flags; decoded.sequence=get(8); decoded.sampleNs=get(8);
        for (unsigned i=0; i<count; ++i) {
            Entry e; e.network=get(4); e.prefix=get(1); e.queueBytes=get(4); e.rateBps=get(8);
            if (flags) { e.prefixBytes=get(4); e.prefixFlows=get(4); }
            decoded.entries.push_back(e);
        }
        if (!decoded.Valid()) return false;
        out=decoded;
        return true;
    }
};

// Each instance belongs to one receiving switch. Only Accept, called after
// reception and wire decoding, can change its remote measurements.
class RouteFeedbackCache {
public:
    struct Received {
        RouteFeedback report;
        int64_t receivedNs = 0;
        std::vector<double> queueDelta; // change in bytes over 200 us
    };
    bool Accept(uint32_t port, const RouteFeedback &report, int64_t now) {
        if (!report.Valid() || now<0) return false;
        auto old=m_received.find(port);
        if (old!=m_received.end() && (report.sequence<=old->second.report.sequence || now<old->second.receivedNs)) return false;
        Received r; r.report=report; r.receivedNs=now; r.queueDelta.resize(report.entries.size(),0.);
        if (old!=m_received.end() && report.sampleNs>old->second.report.sampleNs) {
            for (size_t i=0; i<report.entries.size(); ++i) {
                const auto &entry=report.entries[i];
                for (const auto &prior:old->second.report.entries)
                    if (entry.network==prior.network && entry.prefix==prior.prefix)
                        r.queueDelta[i]=(double(entry.queueBytes)-prior.queueBytes)*200000./
                            (report.sampleNs-old->second.report.sampleNs);
            }
        }
        m_received[port]=r;
        return true;
    }
    const Received *Get(uint32_t port, int64_t now, uint64_t ttl) const {
        auto it=m_received.find(port);
        if (it==m_received.end() || now<it->second.receivedNs || uint64_t(now-it->second.receivedNs)>ttl) return nullptr;
        return &it->second;
    }
    const RouteFeedback::Entry *Find(uint32_t port, uint32_t destination, int64_t now, uint64_t ttl,
                                     double *delta=nullptr, uint64_t *age=nullptr) const {
        const auto *r=Get(port,now,ttl);
        if (!r) return nullptr;
        for (size_t i=0; i<r->report.entries.size(); ++i) {
            const auto &e=r->report.entries[i];
            if (e.Contains(destination)) {
                if (delta) *delta=r->queueDelta[i];
                if (age) *age=uint64_t(now-r->receivedNs);
                return &e;
            }
        }
        return nullptr;
    }
private:
    std::map<uint32_t,Received> m_received;
};

struct RouteFeedbackPrefix { uint32_t network; uint8_t prefix; uint32_t port; };

// Advertise a compact prefix only if it does not cover a locally routed
// destination on another output. Otherwise advertise explicit /32 entries.
inline std::vector<RouteFeedbackPrefix> RouteFeedbackPrefixes(const std::map<uint32_t,uint32_t> &routes) {
    std::map<uint32_t,std::vector<uint32_t>> byPort;
    for (auto r:routes) byPort[r.second].push_back(r.first);
    std::vector<RouteFeedbackPrefix> prefixes;
    for (const auto &group:byPort) {
        const auto &ips=group.second;
        const uint32_t different=ips.front()^ips.back();
        uint8_t bits=32;
        for (uint32_t d=different; d; d>>=1) --bits;
        uint32_t mask=bits ? uint32_t(0xffffffffu << (32-bits)) : 0;
        uint32_t network=ips.front() & mask;
        bool disjoint=bits>0;
        for (auto r:routes) if (r.second!=group.first && (r.first & mask)==network) disjoint=false;
        if (disjoint) prefixes.push_back({network,bits,group.first});
        else for (auto ip:ips) prefixes.push_back({ip,32,group.first});
    }
    return prefixes;
}

} // namespace ns3
#endif
