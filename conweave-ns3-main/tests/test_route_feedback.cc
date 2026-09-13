#include "../src/point-to-point/model/route-feedback.h"
#include <cassert>
#include <iostream>
#include <random>
using namespace ns3;

static RouteFeedback MakeReport(bool pressure=false) {
    RouteFeedback r; r.sequence=1; r.sampleNs=9000000000ULL; r.pressure=pressure;
    RouteFeedback::Entry e; e.network=0x0b001000; e.prefix=20;
    e.queueBytes=32768; e.rateBps=500000000; e.prefixBytes=8192; e.prefixFlows=3;
    r.entries.push_back(e); return r;
}
int main() {
    // Known network-byte-order fields; a byte-only receiver reconstructs all
    // remote measurements, without a sender object or packet identifier.
    auto original=MakeReport(); auto wire=original.Encode();
    assert(wire.size()==41 && wire[0]==0x52 && wire[1]==0x51 && wire[2]==0x46 && wire[3]==0x42);
    assert(wire[4]==1 && wire[5]==0 && wire[6]==0 && wire[7]==1 && wire[15]==1);
    assert(wire[24]==11 && wire[25]==0 && wire[26]==16 && wire[27]==0 && wire[28]==20);
    assert(wire[29]==0 && wire[30]==0 && wire[31]==128 && wire[32]==0);
    RouteFeedback decoded; assert(RouteFeedback::Decode(wire,decoded));
    assert(decoded.entries[0].rateBps==500000000 && decoded.entries[0].queueBytes==32768);
    assert(decoded.sampleNs==original.sampleNs && decoded.Encode()==wire);
    assert(decoded.entries[0].Contains(0x0b001f01) && !decoded.entries[0].Contains(0x0b002001));

    auto pressure=MakeReport(true); auto pressureWire=pressure.Encode();
    assert(pressureWire.size()==49 && RouteFeedback::Decode(pressureWire,decoded));
    assert(decoded.entries[0].prefixBytes==8192 && decoded.entries[0].prefixFlows==3);

    // Malformed/truncated packets cannot partly overwrite a valid message.
    for (size_t n=0; n<pressureWire.size(); ++n) {
        auto shortWire=pressureWire; shortWire.resize(n);
        assert(!RouteFeedback::Decode(shortWire,decoded));
    }
    for (size_t index:{size_t(0),size_t(4),size_t(5),size_t(6),size_t(28)}) {
        auto corrupt=wire; corrupt[index]=255; assert(!RouteFeedback::Decode(corrupt,decoded));
    }
    auto trailing=wire; trailing.push_back(0); assert(!RouteFeedback::Decode(trailing,decoded));
    auto zeroRate=wire; for(size_t i=33;i<41;++i) zeroRate[i]=0;
    assert(!RouteFeedback::Decode(zeroRate,decoded));
    auto overlapping=MakeReport(); overlapping.entries.push_back(overlapping.entries[0]); assert(!overlapping.Valid());
    auto invalid=MakeReport(); invalid.entries[0].prefix=33; assert(!invalid.Valid());

    RouteFeedbackCache cache;
    assert(!cache.Find(1,0x0b001001,100,1000));
    assert(RouteFeedback::Decode(wire,decoded));
    assert(cache.Accept(1,decoded,100));
    original.entries[0].queueBytes=999999; original.entries[0].rateBps=1000000000;
    // A remote change without packet arrival cannot change the receiver.
    assert(cache.Find(1,0x0b001001,101,1000)->queueBytes==32768);
    assert(cache.Find(1,0x0b001001,101,1000)->rateBps==500000000);
    assert(!cache.Find(2,0x0b001001,101,1000)); // independent ingress links
    assert(!cache.Find(1,0x0b002001,101,1000)); // unknown destination
    uint64_t age=0; double delta=0;
    cache.Find(1,0x0b001001,150,1000,&delta,&age); assert(age==50 && delta==0);
    // Sender time has an arbitrary offset; no clock synchronization required.
    original.sequence=2; original.sampleNs+=200000;
    auto secondWire=original.Encode(); RouteFeedback next;
    assert(RouteFeedback::Decode(secondWire,next));
    assert(cache.Find(1,0x0b001001,200,1000)->rateBps==500000000); // delayed, not delivered
    assert(cache.Accept(1,next,300));
    assert(cache.Find(1,0x0b001001,310,1000,&delta,&age)->rateBps==1000000000);
    assert(age==10 && delta==999999-32768);
    assert(!cache.Accept(1,next,400)); // duplicate must not refresh the TTL
    assert(!cache.Accept(1,decoded,500)); // reordered old packet cannot overwrite
    assert(cache.Find(1,0x0b001001,1300,1000));
    assert(!cache.Find(1,0x0b001001,1301,1000)); // loss/expiry never reads live state
    assert(!cache.Find(1,0x0b001001,299,1000));
    assert(cache.Accept(1,next,1400)==false);
    original.sequence=3; original.sampleNs+=200000;
    assert(cache.Accept(1,original,1500)); // a genuinely new arrival recovers

    // Prefix mapping is derived only from the sender's local forwarding table.
    std::map<uint32_t,uint32_t> routes;
    for(unsigned i=0;i<128;++i) routes[0x0b000001+i*256]=i/16+1;
    auto prefixes=RouteFeedbackPrefixes(routes); assert(prefixes.size()==8);
    for(unsigned i=0;i<8;++i) assert(prefixes[i].network==0x0b000000+i*4096 && prefixes[i].prefix==20 && prefixes[i].port==i+1);
    std::map<uint32_t,uint32_t> interleaved{{0x0b000001,1},{0x0b000101,2},{0x0b000201,1}};
    auto exact=RouteFeedbackPrefixes(interleaved); assert(exact.size()==3);
    for(auto p:exact) assert(p.prefix==32);
    RouteFeedback eight; eight.sequence=1;
    for(auto prefix:prefixes) { auto e=MakeReport().entries[0];e.network=prefix.network;e.prefix=prefix.prefix;eight.entries.push_back(e); }
    assert(eight.Encode().size()==160); // +20 B IPv4 +14 B simulated L2 =194 B
    std::mt19937 rng(9101);
    for(unsigned n=0;n<400;++n) {
        std::vector<uint8_t> bytes(n); for(auto &b:bytes)b=rng();
        RouteFeedback ignored; assert(!RouteFeedback::Decode(bytes,ignored));
    }
    std::cout << "PASS: wire fields, malformed packets, causal arrival, expiry, reordering, clock offset, ingress isolation and prefix mapping\n";
}
