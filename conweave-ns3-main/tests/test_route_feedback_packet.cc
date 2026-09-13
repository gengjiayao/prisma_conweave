#include "../src/point-to-point/model/route-feedback.h"
#include "../src/point-to-point/model/ppp-header.h"
#include "ns3/packet.h"
#include "ns3/ipv4-header.h"
#include "ns3/flow-id-tag.h"
#include "ns3/simulator.h"
#include <cassert>
#include <iostream>
using namespace ns3;

static RouteFeedbackCache cache;
static void Receive(Ptr<Packet> packet) {
    PppHeader ppp; Ipv4Header ip;
    packet->RemoveHeader(ppp); packet->RemoveHeader(ip);
    assert(ip.GetProtocol()==RouteFeedback::PROTOCOL);
    std::vector<uint8_t> payload(packet->GetSize()); packet->CopyData(payload.data(),payload.size());
    RouteFeedback report; assert(RouteFeedback::Decode(payload,report));
    assert(cache.Accept(1,report,Simulator::Now().GetNanoSeconds()));
}
static void BeforeDelivery() { assert(!cache.Find(1,0x0b001001,Simulator::Now().GetNanoSeconds(),1000000)); }
static void AfterDelivery() {
    uint64_t age=0; const auto *e=cache.Find(1,0x0b001001,Simulator::Now().GetNanoSeconds(),1000000,nullptr,&age);
    assert(e && e->queueBytes==32768 && e->rateBps==500000000 && e->prefixBytes==8192 && e->prefixFlows==3);
    assert(age==10000);
}
int main() {
    Ptr<Packet> carrier;
    {
        RouteFeedback r; r.sequence=1; r.sampleNs=9000000000ULL; r.pressure=true;
        RouteFeedback::Entry e; e.network=0x0b001000; e.prefix=20;e.queueBytes=32768;e.rateBps=500000000;e.prefixBytes=8192;e.prefixFlows=3;
        r.entries.push_back(e); auto payload=r.Encode();
        auto packet=Create<Packet>(payload.data(),payload.size());
        Ipv4Header ip; ip.SetProtocol(RouteFeedback::PROTOCOL);ip.SetPayloadSize(packet->GetSize());packet->AddHeader(ip);
        PppHeader ppp;ppp.SetProtocol(0x0021);packet->AddHeader(ppp);
        packet->AddPacketTag(FlowIdTag(12345));
        // Reconstruct a new ns-3 packet from frame bytes alone. It has a
        // different UID and no tags; the sender's report then goes out of scope.
        std::vector<uint8_t> frame(packet->GetSize());packet->CopyData(frame.data(),frame.size());
        carrier=Create<Packet>(frame.data(),frame.size());
        assert(carrier->GetUid()!=packet->GetUid());
        FlowIdTag absent;assert(!carrier->PeekPacketTag(absent));
        assert(carrier->GetSize()==83); // 49-byte pressure payload +34-byte framing
    }
    Simulator::Schedule(MicroSeconds(50),&BeforeDelivery);
    Simulator::Schedule(MicroSeconds(100),&Receive,carrier);
    Simulator::Schedule(MicroSeconds(110),&AfterDelivery);
    Simulator::Run();Simulator::Destroy();
    std::cout<<"PASS: ns-3 wire frame survives new packet UID and removal of all metadata; measurements appear only after delivery\n";
}
