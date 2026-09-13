#include "switch-node.h"
#include "weighted-ecmp.h"
#include "routing-diagnostics.h"
#include "route-learning.h"
#include <random>
#include <unordered_set>
#include "qbb-header.h"
#include <fstream>
#include <iomanip>
#include <limits>

#include "assert.h"
#include "ns3/boolean.h"
#include "ns3/conweave-routing.h"
#include "ns3/double.h"
#include "ns3/flow-id-tag.h"
#include "ns3/flow-id-num-tag.h"
#include "ns3/int-header.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4.h"
#include "ns3/letflow-routing.h"
#include "ns3/packet.h"
#include "ns3/pause-header.h"
#include "ns3/settings.h"
#include "ns3/uinteger.h"
#include "ns3/conweave-obs-manager.h" 
#include "ppp-header.h"
#include "qbb-net-device.h"
#include <atomic>
#include <chrono>


NS_LOG_COMPONENT_DEFINE("SwitchNode");

namespace {
  struct DiagnosticCounters {
    uint64_t packets = 0, bytes = 0, routeChanges = 0, decisions = 0;
    uint64_t suboptimal = 0, staleReads = 0, missingHistory = 0;
    uint64_t rxPackets = 0, rxBytes = 0, outOfOrder = 0, duplicates = 0;
    uint64_t ecnPackets = 0, reorderCnp = 0, reorderWithoutEcn = 0;
    uint64_t proposedChanges = 0, guardedChanges = 0, missingReports = 0;
    double queueRegretNs = 0, chosenDelayNs = 0, informationAgeNs = 0;
    double localQueueNs = 0, remoteQueueNs = 0;
  };
  DiagnosticCounters S_diagnostic[2];
  uint64_t S_reportPackets = 0, S_reportBytes = 0, S_reportsDelivered = 0;
  uint64_t S_reportsRejected = 0, S_reportsMalformed = 0;
  std::map<uint32_t, uint32_t> S_backgroundPaths;
  // RL拦截在交换机侧的统计（进程级聚合）
  std::atomic<uint64_t> S_rlSeen{0};         // 满足候选条件、尝试拦截的包（非控制、非目的ToR）
  std::atomic<uint64_t> S_rlHeld{0};         // 实际被挂起并请求动作
  std::atomic<uint64_t> S_rlBusySkip{0};     // 因 m_rlBusy 放行
  std::atomic<uint64_t> S_rlCtrlSkip{0};     // 因控制报文放行
  std::atomic<uint64_t> S_rlDstTorSkip{0};   // 因为到本ToR（下行）放行
  std::atomic<uint64_t> S_rlTimeoutFallback{0}; // 因动作超时回退（ECMP 放行）

  // 非 RL 模式：记录整次仿真的墙钟耗时（进程级一次性）
  struct WallClockTimer {
    std::chrono::steady_clock::time_point t0;
    WallClockTimer() : t0(std::chrono::steady_clock::now()) {}
    ~WallClockTimer() {
      // 仅在“全局 lb_mode != 7 (RL 覆写)”时打印
      if (ns3::Settings::lb_mode != 7) {
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        NS_LOG_UNCOND("[RUNTIME] lb_mode=" << ns3::Settings::lb_mode << " wall_clock_sec=" << sec);
      }
    }
  } s_wallClockTimer;

  struct RlSwitchPrinter {
    ~RlSwitchPrinter() {
      const double seen = static_cast<double>(S_rlSeen.load());
      const double held = static_cast<double>(S_rlHeld.load());
      const double busy = static_cast<double>(S_rlBusySkip.load());
      const double ctrl = static_cast<double>(S_rlCtrlSkip.load());
      const double dst  = static_cast<double>(S_rlDstTorSkip.load());
      const double tofb = static_cast<double>(S_rlTimeoutFallback.load());
      const double hold_ratio = (seen > 0.0 ? held / seen : 0.0);
      const double busy_rate  = (seen > 0.0 ? busy / seen : 0.0);
      NS_LOG_UNCOND("[RL-SW] seen=" << (uint64_t)seen
                     << " held=" << (uint64_t)held
                     << " busy_skip=" << (uint64_t)busy
                     << " ctrl_skip=" << (uint64_t)ctrl
                     << " dsttor_skip=" << (uint64_t)dst
                     << " timeout_fb=" << (uint64_t)tofb
                     << " hold_ratio=" << hold_ratio
                     << " busy_rate=" << busy_rate);
    }
  } s_rlSwitchPrinter;
}

namespace ns3 {

/* ****************  RL per-hop hold & release  **************** */

// -- 静态函数实现 --
void
SwitchNode::IncrementRlHeld() { S_rlHeld++; }

void
SwitchNode::IncrementRlBusySkip() { S_rlBusySkip++; }


void
SwitchNode::RlHandover(Ptr<NetDevice> inDev, Ptr<Packet> p, CustomHeader &ch)
{
    // 1. 重新加入过滤逻辑：控制包直接放行，不计入RL统计
    const bool control_pkt = (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE ||
                              ch.l3Prot == 0xFD || ch.l3Prot == 0xFC);
    if (control_pkt) { S_rlCtrlSkip++; SendToDev(p, ch); return; }

    // 2. 重新加入过滤逻辑：目的 ToR的包直接下发，不计入RL统计
    if (m_isToR && m_isToR_hostIP.find(ch.dip) != m_isToR_hostIP.end()) {
        S_rlDstTorSkip++;
        SendToDev(p, ch);
        return;
    }

    // 3. 统计进入候选的包（到此已排除控制与目的ToR）
    S_rlSeen++;

    if (!m_rlMgr) { SendToDev(p, ch); return; }
    Ptr<ConweaveObsManager> mgr = DynamicCast<ConweaveObsManager>(m_rlMgr);
    if (!mgr || !mgr->Ready()) { SendToDev(p, ch); return; }

    // 4. 将包交给 Manager 处理，由其内部 Flowlet 逻辑决定是否挂起
    // Manager会负责调用 SendToDev 或 RlRelease
    mgr->OnPerHopPacket(this, inDev, p, ch);
}

void
SwitchNode::RlRelease(Ptr<Packet> p, CustomHeader& ch, uint32_t outIf)
{
    if (m_rlTimeoutEv.IsRunning()) m_rlTimeoutEv.Cancel();

    // 选队列优先级（完全复用原有逻辑）
    uint32_t qIndex;
    if (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE ||
        (m_ackHighPrio && (ch.l3Prot == 0xFD || ch.l3Prot == 0xFC))) {
        qIndex = 0;
    } else {
        qIndex = (ch.l3Prot == 0x06 ? 1 : ch.udp.pg);
    }

    // 用传入的包从指定端口发出
    DoSwitchSend(p, ch, outIf, qIndex);

    // 【诊断】首次RL释放（中文一次性打印）
    {
        static bool s_printed = false;
        if (!s_printed) {
            NS_LOG_UNCOND("【RL释放】首次触发 outIf=" << outIf << " sw=" << GetId());
            s_printed = true;
        }
    }
}


void
SwitchNode::RlTimeoutFallback()
{
    // 这个函数的逻辑也需要重写，因为它依赖 m_rlHeld
    // 超时也应该通知 Manager，由 Manager 来决定如何处理
    if (!m_rlMgr) return;
    Ptr<ConweaveObsManager> mgr = DynamicCast<ConweaveObsManager>(m_rlMgr);
    if (mgr) {
        // 我们需要在 ConweaveObsManager 中添加一个处理超时的函数
        // mgr->OnRlTimeout();
    }
    // S_rlTimeoutFallback++; // 统计移到 Manager
}
/* ************************************************************* */


TypeId SwitchNode::GetTypeId(void) {
    static TypeId tid =
        TypeId("ns3::SwitchNode")
            .SetParent<Node>()
            .AddConstructor<SwitchNode>()
            .AddTraceSource("MacRx",
                            "Packet just arrived at the switch from a port (after parsing, non PFC).",
                            MakeTraceSourceAccessor(&SwitchNode::m_traceMacRx))
            .AddAttribute("EcnEnabled", "Enable ECN marking.", BooleanValue(false),
                          MakeBooleanAccessor(&SwitchNode::m_ecnEnabled), MakeBooleanChecker())
            .AddAttribute("CcMode", "CC mode.", UintegerValue(0),
                          MakeUintegerAccessor(&SwitchNode::m_ccMode),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("LbMode",
                            "Load balancer mode for this switch. 0: follow global Settings::lb_mode; 7: RL override",
                            UintegerValue(0),
                            MakeUintegerAccessor(&SwitchNode::m_lbMode),
                            MakeUintegerChecker<uint32_t>())
            .AddAttribute("AckHighPrio", "Set high priority for ACK/NACK or not", UintegerValue(0),
                          MakeUintegerAccessor(&SwitchNode::m_ackHighPrio),
                          MakeUintegerChecker<uint32_t>());
    return tid;
}

SwitchNode::SwitchNode() {
    m_ecmpSeed = m_id;
    m_isToR = false;
    m_node_type = 1;
    m_isToR = false;
    m_drill_candidate = 2;
    m_mmu = CreateObject<SwitchMmu>();
    // Conga's Callback for switch functions
    m_mmu->m_congaRouting.SetSwitchSendCallback(MakeCallback(&SwitchNode::DoSwitchSend, this));
    m_mmu->m_congaRouting.SetSwitchSendToDevCallback(
        MakeCallback(&SwitchNode::SendToDevContinue, this));
    // ConWeave's Callback for switch functions
    m_mmu->m_conweaveRouting.SetSwitchSendCallback(MakeCallback(&SwitchNode::DoSwitchSend, this));
    m_mmu->m_conweaveRouting.SetSwitchSendToDevCallback(
        MakeCallback(&SwitchNode::SendToDevContinue, this));

    for (uint32_t i = 0; i < pCnt; i++) {
        m_txBytes[i] = 0;
    }
}

/**
 * @brief Load Balancing
 */
uint32_t SwitchNode::DoLbFlowECMP(Ptr<const Packet> p, const CustomHeader &ch,
                                  const std::vector<int> &nexthops) {
    // pick one next hop based on hash
    union {
        uint8_t u8[4 + 4 + 2 + 2];
        uint32_t u32[3];
    } buf;
    buf.u32[0] = ch.sip;
    buf.u32[1] = ch.dip;
    if (ch.l3Prot == 0x6)
        buf.u32[2] = ch.tcp.sport | ((uint32_t)ch.tcp.dport << 16);
    else if (ch.l3Prot == 0x11)  // XXX RDMA traffic on UDP
        buf.u32[2] = ch.udp.sport | ((uint32_t)ch.udp.dport << 16);
    else if (ch.l3Prot == 0xFC || ch.l3Prot == 0xFD)  // ACK or NACK
        buf.u32[2] = ch.ack.sport | ((uint32_t)ch.ack.dport << 16);
    else {
        std::cout << "[ERROR] Sw(" << m_id << ")," << PARSE_FIVE_TUPLE(ch)
                  << "Cannot support other protoocls than TCP/UDP (l3Prot:" << ch.l3Prot << ")"
                  << std::endl;
        assert(false && "Cannot support other protoocls than TCP/UDP");
    }

    uint32_t hashVal = EcmpHash(buf.u8, 12, m_ecmpSeed);
    if (Settings::lb_mode == 1 && (ch.l3Prot == 0x6 || ch.l3Prot == 0x11)) {
        std::vector<uint64_t> rates;
        for (int outIf : nexthops) {
            Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(m_devices[outIf]);
            NS_ASSERT_MSG(dev, "Weighted ECMP requires QbbNetDevice next hops");
            rates.push_back(dev->GetDataRate().GetBitRate());
        }
        return nexthops[CapacityWeightedIndex(hashVal, rates)];
    }
    uint32_t idx = hashVal % nexthops.size();
    return nexthops[idx];
}

/*-----------------CONGA-----------------*/
uint32_t SwitchNode::DoLbConga(Ptr<Packet> p, CustomHeader &ch, const std::vector<int> &nexthops) {
    return DoLbFlowECMP(p, ch, nexthops);  // flow ECMP (dummy)
}

/*-----------------Letflow-----------------*/
uint32_t SwitchNode::DoLbLetflow(Ptr<Packet> p, CustomHeader &ch,
                                 const std::vector<int> &nexthops) {
    if (m_isToR && nexthops.size() == 1) {
        if (m_isToR_hostIP.find(ch.sip) != m_isToR_hostIP.end() &&
            m_isToR_hostIP.find(ch.dip) != m_isToR_hostIP.end()) {
            return nexthops[0];  // intra-pod traffic
        }
    }

    /* ONLY called for inter-Pod traffic */
    uint32_t outPort = m_mmu->m_letflowRouting.RouteInput(p, ch);
    if (outPort == LETFLOW_NULL) {
        assert(nexthops.size() == 1);  // Receiver's TOR has only one interface to receiver-server
        outPort = nexthops[0];         // has only one option
    }
    assert(std::find(nexthops.begin(), nexthops.end(), outPort) !=
           nexthops.end());  // Result of Letflow cannot be found in nexthops
    return outPort;
}

/*-----------------DRILL-----------------*/
uint32_t SwitchNode::CalculateInterfaceLoad(uint32_t interface) {
    Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(m_devices[interface]);
    NS_ASSERT_MSG(!!device && !!device->GetQueue(),
                  "Error of getting a egress queue for calculating interface load");
    return device->GetQueue()->GetNBytesTotal();  // also used in HPCC
}

uint32_t SwitchNode::DoLbDrill(Ptr<const Packet> p, const CustomHeader &ch,
                               const std::vector<int> &nexthops) {
    // find the Egress (output) link with the smallest local Egress Queue length
    uint32_t leastLoadInterface = 0;
    uint32_t leastLoad = std::numeric_limits<uint32_t>::max();
    auto rand_nexthops = nexthops;
    std::random_shuffle(rand_nexthops.begin(), rand_nexthops.end());

    std::map<uint32_t, uint32_t>::iterator itr = m_previousBestInterfaceMap.find(ch.dip);
    if (itr != m_previousBestInterfaceMap.end()) {
        leastLoadInterface = itr->second;
        leastLoad = CalculateInterfaceLoad(itr->second);
    }

    uint32_t sampleNum =
        m_drill_candidate < rand_nexthops.size() ? m_drill_candidate : rand_nexthops.size();
    for (uint32_t samplePort = 0; samplePort < sampleNum; samplePort++) {
        uint32_t sampleLoad = CalculateInterfaceLoad(rand_nexthops[samplePort]);
        if (sampleLoad < leastLoad) {
            leastLoad = sampleLoad;
            leastLoadInterface = rand_nexthops[samplePort];
        }
    }
    m_previousBestInterfaceMap[ch.dip] = leastLoadInterface;
    return leastLoadInterface;
}

/*------------------ConWeave Dummy ----------------*/
uint32_t SwitchNode::DoLbConWeave(Ptr<const Packet> p, const CustomHeader &ch,
                                  const std::vector<int> &nexthops) {
    return DoLbFlowECMP(p, ch, nexthops);  // flow ECMP (dummy)
}
/*----------------------------------*/

//9.11 RL
void
SwitchNode::SetRlPreferredOutIf(uint32_t dstTorId, uint32_t outIf)
{
  // 覆写为“待消费”的一次性端口
  m_rlPreferOnce[dstTorId] = outIf;
}

bool
SwitchNode::TryConsumeRlPreferredOutIf(uint32_t dstTorId, uint32_t &outIfOut)
{
  auto it = m_rlPreferOnce.find(dstTorId);
  if (it == m_rlPreferOnce.end()) {
    return false;
  }
  outIfOut = it->second;
  m_rlPreferOnce.erase(it); // 只消费一次
  return true;
}

// [Design A 2026-05-14] Per-flowKey 路由表实现
// 目标：让同一 flowlet 内所有包都跟随首包走 RL 选的 port，消除 intra-flowlet reorder

void
SwitchNode::SetRlPreferredForFlow(uint64_t flowKey, uint32_t outIf)
{
  // 触发清理：若 map 太大，先扫描过期 entry
  if (m_rlPrefByFlow.size() > kFlowletPrefSizeCap) {
    CleanupExpiredFlowPref();
  }
  FlowletRoutePref pref;
  pref.outIf = outIf;
  pref.last_access_sec = Simulator::Now().GetSeconds();
  m_rlPrefByFlow[flowKey] = pref;
}

bool
SwitchNode::LookupRlPreferredForFlow(uint64_t flowKey, uint32_t &outIfOut)
{
  auto it = m_rlPrefByFlow.find(flowKey);
  if (it == m_rlPrefByFlow.end()) return false;

  double now = Simulator::Now().GetSeconds();
  // 过期则清掉并 fallback
  if (now - it->second.last_access_sec > kFlowletPrefTTL) {
    m_rlPrefByFlow.erase(it);
    return false;
  }

  // refresh-on-access：长 flowlet 持续访问会刷新 last_access_sec，不会被截断
  it->second.last_access_sec = now;
  outIfOut = it->second.outIf;
  return true;
}

void
SwitchNode::CleanupExpiredFlowPref()
{
  double now = Simulator::Now().GetSeconds();
  for (auto it = m_rlPrefByFlow.begin(); it != m_rlPrefByFlow.end(); ) {
    if (now - it->second.last_access_sec > kFlowletPrefTTL) {
      it = m_rlPrefByFlow.erase(it);
    } else {
      ++it;
    }
  }
}


void SwitchNode::CheckAndSendPfc(uint32_t inDev, uint32_t qIndex) {
    Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(m_devices[inDev]);
    bool pClasses[qCnt] = {0};
    m_mmu->GetPauseClasses(inDev, qIndex, pClasses);
    for (int j = 0; j < qCnt; j++) {
        if (pClasses[j]) {
            uint32_t paused_time = device->SendPfc(j, 0);
            m_mmu->SetPause(inDev, j, paused_time);
            m_mmu->m_pause_remote[inDev][j] = true;
            /** PAUSE SEND COUNT ++ */
        }
    }

    for (int j = 0; j < qCnt; j++) {
        if (!m_mmu->m_pause_remote[inDev][j]) continue;

        if (m_mmu->GetResumeClasses(inDev, j)) {
            device->SendPfc(j, 1);
            m_mmu->SetResume(inDev, j);
            m_mmu->m_pause_remote[inDev][j] = false;
        }
    }
}
void SwitchNode::CheckAndSendResume(uint32_t inDev, uint32_t qIndex) {
    Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(m_devices[inDev]);
    if (!device) return;
    if (m_mmu->GetResumeClasses(inDev, qIndex)) {
        // device->SendPfc(qIndex, 1);
        // m_mmu->SetResume(inDev, qIndex);
        if (device->IsPaused(qIndex)) {           // 只有真暂停过才发 RESUME
            device->SendPfc(qIndex, 1);
            m_mmu->SetResume(inDev, qIndex);
        } else {
            NS_LOG_LOGIC("Skip RESUME: queue " << qIndex << " not paused on dev " << inDev);
        }
    }
}

/********************************************
 *              MAIN LOGICS                 *
 *******************************************/

// This function can only be called in switch mode
//9.9修改
bool SwitchNode::SwitchReceiveFromDevice(Ptr<NetDevice> device, Ptr<Packet> packet,
                                         CustomHeader &ch) {
    m_traceMacRx(device,packet);  // 9.9trace callback
    if (RouteLearning::mode && m_isToR && m_isToR_hostIP.count(ch.dip) &&
        (ch.l3Prot==0xFC || ch.l3Prot==0xFD)) {
        uint64_t key=ConWeaveRouting::GetFlowKey(ch.dip,ch.sip,ch.ack.dport,ch.ack.sport);
        auto &ack=m_routeAcks[key]; int64_t now=Simulator::Now().GetNanoSeconds();
        if (ack.lastNs>=0 && now>ack.lastNs && ch.ack.seq>ack.sequence) {
            double rate=(ch.ack.seq-ack.sequence)*8./double(now-ack.lastNs);
            ack.rate=.8*ack.rate+.2*std::min(1.,rate);
        }
        ack.sequence=std::max(ack.sequence,ch.ack.seq);ack.lastNs=now;
        // IRN uses the NACK protocol number even for ordinary cumulative ACKs.
        // A nonzero selective range identifies out-of-order reception feedback.
        if(ch.l3Prot==0xFD && ch.ack.irnNackSize>0) ack.nackNs=now;
    }
    if (ch.l3Prot == RouteFeedback::PROTOCOL) {
        // The only remote-state update path: bytes received on this link.
        if (!m_isToR || !RoutingDiagnostics::enabled || !RoutingDiagnostics::feedback) return true;
        Ptr<Packet> payload=packet->Copy();
        PppHeader ppp; Ipv4Header ip;
        payload->RemoveHeader(ppp); payload->RemoveHeader(ip);
        std::vector<uint8_t> wire(payload->GetSize());
        payload->CopyData(wire.data(),wire.size());
        RouteFeedback report;
        if (!RouteFeedback::Decode(wire,report) || report.pressure!=RouteLearning::pressureEnabled) {
            ++S_reportsMalformed; return true;
        }
        ++S_reportsDelivered;
        if (!m_routeFeedback.Accept(device->GetIfIndex(),report,Simulator::Now().GetNanoSeconds())) {
            ++S_reportsRejected; return true;
        }
        if (RouteLearning::pressureEnabled)
            for (const auto &e:report.entries)
                RouteLearning::RecordPressure('R',{uint64_t(Simulator::Now().GetNanoSeconds()),GetId(),
                    device->GetIfIndex(),e.network,e.prefixBytes,e.prefixFlows,report.sampleNs});
        return true;
    }

    if (RoutingDiagnostics::enabled && RoutingDiagnostics::backgroundMode &&
        ch.l3Prot == 0x11 && RoutingDiagnostics::Group(ch.sip) == 1) {
        SendToDev(packet, ch);
        return true;
    }

    // === RL 每跳拦截 ===
    uint32_t lbModeLocal = (m_lbMode == 0) ? Settings::lb_mode : m_lbMode;
    if (lbModeLocal == 7 && m_rlMgr) {
        // 直接将包交给RL管理器处理。
        // Manager内部的Flowlet逻辑会决定是立即使用缓存转发，还是挂起等待Python决策。
        // 不再需要返回值，因为Manager接管了包的生命周期。
        RlHandover(device, packet, ch);
        return true; // Manager会负责后续处理（转发或丢弃），所以这里直接返回
    } 


    SendToDev(packet, ch);
    NS_LOG_LOGIC("MacRx fired uid=" << packet->GetUid());
    return true;
}

void SwitchNode::SendToDev(Ptr<Packet> p, CustomHeader &ch) {
    // Exogenous diagnostic background follows the same fixed path under every
    // treatment. Bypass path-tag algorithms at every hop for these data packets.
    if (RoutingDiagnostics::enabled && RoutingDiagnostics::backgroundMode &&
        ch.l3Prot == 0x11 && RoutingDiagnostics::Group(ch.sip) == 1) {
        SendToDevContinue(p, ch);
        return;
    }
    /** HIJACK: hijack the packet and run DoSwitchSend internally for Conga and ConWeave.
     * Note that DoLbConWeave() and DoLbConga() are flow-ECMP function for control packets
     * or intra-ToR traffic.
     */

    // Conga
    if (Settings::lb_mode == 3) {
        m_mmu->m_congaRouting.RouteInput(p, ch);
        return;
    }

    // ConWeave
    if (Settings::lb_mode == 9 && m_lbMode != 7) { // 全局 ConWeave 模式且非 RL 覆写模式
        m_mmu->m_conweaveRouting.RouteInput(p, ch);
        return;
    }

    // Others
    SendToDevContinue(p, ch);
}
uint32_t SwitchNode::DoLbRl(Ptr<Packet> p, CustomHeader &ch, const std::vector<int> &nexthops) {
    // 只对数据包进行 RL 选择
    bool control_pkt = (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE || ch.l3Prot == 0xFD || ch.l3Prot == 0xFC);
    if (control_pkt) return DoLbFlowECMP(p, ch, nexthops);

    // 在目的 ToR 强制走本地 ECMP（nexthops 通常只有唯一下行端口）
    if (m_isToR && m_isToR_hostIP.find(ch.dip) != m_isToR_hostIP.end()) {
        return DoLbFlowECMP(p, ch, nexthops);
    }

    // [Design A 2026-05-14] 优先：按 flowKey 查找 per-flow RL preference
    // 与 conweave-obs-manager.cc 中 flowKey 计算保持完全一致
    uint64_t flowKey = 0;
    if (ch.udp.sport || ch.udp.dport) {
      flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
    } else if (ch.tcp.sport || ch.tcp.dport) {
      flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.tcp.sport, ch.tcp.dport);
    } else {
      flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, 0, 0);
    }
    if (flowKey != 0) {
      uint32_t rlOutIf = 0;
      if (LookupRlPreferredForFlow(flowKey, rlOutIf) && rlOutIf > 0) {
        return rlOutIf; // 命中 per-flow RL 选择（保证同 flowlet 内所有包路径一致）
      }
    }

    // 兼容老接口：dstToR 一次性 RL 选择（实际已 dead，保留为防御）
    auto itTor = Settings::hostIp2SwitchId.find(ch.dip);
    if (itTor != Settings::hostIp2SwitchId.end()) {
        uint32_t dstToR = itTor->second;
        uint32_t rlOutIf = 0;
        if (TryConsumeRlPreferredOutIf(dstToR, rlOutIf) && rlOutIf > 0) {
            return rlOutIf;
        }
    }
    // 未命中：回退到 Flow ECMP（同 flow 始终走同一 port，保证 baseline 行为）
    return DoLbFlowECMP(p, ch, nexthops);
}


void SwitchNode::SendToDevContinue(Ptr<Packet> p, CustomHeader &ch) {
    int idx = GetOutDev(p, ch);
    if (idx >= 0) {
        NS_ASSERT_MSG(m_devices[idx]->IsLinkUp(),
                      "The routing table look up should return link that is up");

        // determine the qIndex
        uint32_t qIndex;
        if (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE ||
            (m_ackHighPrio &&
             (ch.l3Prot == 0xFD ||
              ch.l3Prot == 0xFC))) {  // QCN or PFC or ACK/NACK, go highest priority
            qIndex = 0;               // high priority
        } else {
            qIndex = (ch.l3Prot == 0x06 ? 1 : ch.udp.pg);  // if TCP, put to queue 1. Otherwise, it
                                                           // would be 3 (refer to trafficgen)
        }

        DoSwitchSend(p, ch, idx, qIndex);  // m_devices[idx]->SwitchSend(qIndex, p, ch);
        return;
    }
    std::cout << "WARNING - Drop occurs in SendToDevContinue()" << std::endl;
    return;  // Drop otherwise
}

int SwitchNode::GetOutDev(Ptr<Packet> p, CustomHeader &ch) {
    // look up entries
    auto entry = m_rtTable.find(ch.dip);

    // no matching entry
    if (entry == m_rtTable.end()) {
        std::cout << "[ERROR] Sw(" << m_id << ")," << PARSE_FIVE_TUPLE(ch)
                  << "No matching entry, so drop this packet at SwitchNode (l3Prot:" << ch.l3Prot
                  << ")" << std::endl;
        assert(false);
    }

    // entry found
    const auto &nexthops = entry->second;
    if (RoutingDiagnostics::enabled && RoutingDiagnostics::backgroundMode &&
        ch.l3Prot == 0x11 && RoutingDiagnostics::Group(ch.sip) == 1) {
        if (m_isToR && m_isToR_hostIP.count(ch.sip) && nexthops.size() > 1) {
            uint32_t source = Settings::hostIp2IdMap.at(ch.sip) % 128;
            uint32_t index = RoutingDiagnostics::backgroundMode == 2 && source >= 64 ? 1 : 0;
            if (RoutingDiagnostics::backgroundMode == 3) {
                FlowIDNUMTag tag;
                if (!p->PeekPacketTag(tag)) NS_FATAL_ERROR("Pinned background packet lacks flow ID");
                index = S_backgroundPaths.at(tag.GetId());
            }
            return nexthops.at(index);
        }
        return DoLbFlowECMP(p, ch, nexthops);
    }
    bool control_pkt =
        (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE || ch.l3Prot == 0xFD || ch.l3Prot == 0xFC);

    // RL 覆写在 DoLbRl 中统一处理
    // === RL 未命中 -> 落回原先逻辑 ===

    if (Settings::lb_mode == 0 || control_pkt) {  // control packet (ACK, NACK, PFC, QCN)
        return DoLbFlowECMP(p, ch, nexthops);     // ECMP routing path decision (4-tuple)
    }

    switch (Settings::lb_mode) {
        case 12:
            return DiagnosticRoute(p, ch, nexthops);
        case 1:
            return DoLbFlowECMP(p, ch, nexthops);
        case 7:
            return DoLbRl(p, ch, nexthops);
        case 2:
            return DoLbDrill(p, ch, nexthops);
        case 3:
            return DoLbConga(p, ch, nexthops); /** DUMMY: Do ECMP */
        case 6:
            return DoLbLetflow(p, ch, nexthops);
        case 9:
            return DoLbConWeave(p, ch, nexthops); /** DUMMY: Do ECMP */
        default:
            std::cout << "Unknown lb_mode(" << Settings::lb_mode << ")" << std::endl;
            assert(false);
    }
}

/*
 * The (possible) callback point when conweave dequeues packets from buffer
 */
void SwitchNode::DoSwitchSend(Ptr<Packet> p, CustomHeader &ch, uint32_t outDev, uint32_t qIndex) {
    if (RoutingDiagnostics::enabled && RoutingDiagnostics::backgroundMode &&
        Settings::lb_mode == 3 && ch.l3Prot == 0x11 && RoutingDiagnostics::Group(ch.sip) == 1) {
        // Pinning changes route selection only. Background bytes must still
        // enter CONGA's physical-link load estimator, including at the spine.
        auto &conga = m_mmu->m_congaRouting;
        if (!conga.m_dreEvent.IsRunning())
            conga.m_dreEvent = Simulator::Schedule(conga.m_dreTime, &CongaRouting::DreEvent, &conga);
        conga.UpdateLocalDre(p, ch, outDev);
        RoutingDiagnostics::congaBackgroundBytes += p->GetSize();
    }
    if (RoutingDiagnostics::enabled) DiagnosticObserve(p, ch, outDev);
    // admission control
    FlowIdTag t;
    p->PeekPacketTag(t);
    uint32_t inDev = t.GetFlowId();

    /** NOTE:
     * ConWeave control packets have the high priority as ACK/NACK/PFC/etc with qIndex = 0.
     */
    if (inDev == Settings::CONWEAVE_CTRL_DUMMY_INDEV) { // sanity check
        // ConWeave reply is on ACK protocol with high priority, so qIndex should be 0
        assert(qIndex == 0 && m_ackHighPrio == 1 && "ConWeave's reply packet follows ACK, so its qIndex should be 0");
    }

    if (qIndex != 0) {  // not highest priority
        if (m_mmu->CheckEgressAdmission(outDev, qIndex,
                                        p->GetSize())) {  // Egress Admission control
            if (m_mmu->CheckIngressAdmission(inDev, qIndex,
                                             p->GetSize())) {  // Ingress Admission control
                m_mmu->UpdateIngressAdmission(inDev, qIndex, p->GetSize());
                m_mmu->UpdateEgressAdmission(outDev, qIndex, p->GetSize());
            } else { /** DROP: At Ingress */
#if (0)
                // /** NOTE: logging dropped pkts */
                // std::cout << "LostPkt ingress - Sw(" << m_id << ")," << PARSE_FIVE_TUPLE(ch)
                //           << "L3Prot:" << ch.l3Prot
                //           << ",Size:" << p->GetSize()
                //           << ",At " << Simulator::Now() << std::endl;
#endif
                Settings::dropped_pkt_sw_ingress++;
                return;  // drop
            }
        } else { /** DROP: At Egress */
#if (0)
            // /** NOTE: logging dropped pkts */
            // std::cout << "LostPkt egress - Sw(" << m_id << ")," << PARSE_FIVE_TUPLE(ch)
            //           << "L3Prot:" << ch.l3Prot << ",Size:" << p->GetSize() << ",At "
            //           << Simulator::Now() << std::endl;
#endif
            Settings::dropped_pkt_sw_egress++;
            return;  // drop
        }

        CheckAndSendPfc(inDev, qIndex);
    }

    m_devices[outDev]->SwitchSend(qIndex, p, ch);
}

void SwitchNode::SwitchNotifyDequeue(uint32_t ifIndex, uint32_t qIndex, Ptr<Packet> p) {
    if (RouteLearning::pressureEnabled) {
        CustomHeader header(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);
        p->PeekHeader(header);ObservePrefixQueue(ifIndex,p,header,false);
    }
    FlowIdTag t;
    p->PeekPacketTag(t);
    if (qIndex != 0) {
        uint32_t inDev = t.GetFlowId();
        if (inDev != Settings::CONWEAVE_CTRL_DUMMY_INDEV) {
            // NOTE: ConWeave's probe/reply does not need to pass inDev interface,
            // so skip for conweave's queued packets
            m_mmu->RemoveFromIngressAdmission(inDev, qIndex, p->GetSize());
        }
        m_mmu->RemoveFromEgressAdmission(ifIndex, qIndex, p->GetSize());
        if (m_ecnEnabled) {
            bool egressCongested = m_mmu->ShouldSendCN(ifIndex, qIndex);
            if (egressCongested) {
                PppHeader ppp;
                Ipv4Header h;
                p->RemoveHeader(ppp);
                p->RemoveHeader(h);
                h.SetEcn((Ipv4Header::EcnType)0x03);
                p->AddHeader(h);
                p->AddHeader(ppp);
            }
        }
        // NOTE: ConWeave's probe/reply does not need to pass inDev interface
        if (inDev != Settings::CONWEAVE_CTRL_DUMMY_INDEV) {
            CheckAndSendResume(inDev, qIndex);
        }
    }

    // HPCC's INT
    if (1) {
        uint8_t *buf = p->GetBuffer();
        if (buf[PppHeader::GetStaticSize() + 9] == 0x11) {  // udp packet
            IntHeader *ih = (IntHeader *)&buf[PppHeader::GetStaticSize() + 20 + 8 +
                                              6];  // ppp, ip, udp, SeqTs, INT
            Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(m_devices[ifIndex]);
            if (m_ccMode == 3) {  // HPCC
                ih->PushHop(Simulator::Now().GetTimeStep(), m_txBytes[ifIndex],
                            dev->GetQueue()->GetNBytesTotal(), dev->GetDataRate().GetBitRate());
            }
        }
    }
    m_txBytes[ifIndex] += p->GetSize();
}

uint32_t SwitchNode::EcmpHash(const uint8_t *key, size_t len, uint32_t seed) {
    uint32_t h = seed;
    if (len > 3) {
        const uint32_t *key_x4 = (const uint32_t *)key;
        size_t i = len >> 2;
        do {
            uint32_t k = *key_x4++;
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h += (h << 2) + 0xe6546b64;
        } while (--i);
        key = (const uint8_t *)key_x4;
    }
    if (len & 3) {
        size_t i = len & 3;
        uint32_t k = 0;
        key = &key[i - 1];
        do {
            k <<= 8;
            k |= *key--;
        } while (--i);
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

void SwitchNode::SetEcmpSeed(uint32_t seed) { m_ecmpSeed = seed; }

void SwitchNode::AddTableEntry(Ipv4Address &dstAddr, uint32_t intf_idx) {
    uint32_t dip = dstAddr.Get();
    m_rtTable[dip].push_back(intf_idx);
}

void SwitchNode::ClearTable() { m_rtTable.clear(); }

uint64_t SwitchNode::GetTxBytesOutDev(uint32_t outdev) {
    assert(outdev < pCnt);
    return m_txBytes[outdev];
}

bool RoutingDiagnostics::enabled = false;
bool RoutingDiagnostics::remote = false;
bool RoutingDiagnostics::oooCnp = true;
bool RoutingDiagnostics::feedback = false;
bool RoutingDiagnostics::guard = false;
uint64_t RoutingDiagnostics::feedbackPeriodNs = 50000;
uint64_t RoutingDiagnostics::guardMarginNs = 0;
uint32_t RoutingDiagnostics::reportQuantumBytes = 0;
bool RoutingDiagnostics::ageGuard = false;
bool RoutingDiagnostics::paddedReports = false;
bool RoutingDiagnostics::congaAckFeedback = false;
uint64_t RoutingDiagnostics::congaAckUpdates = 0;
uint64_t RoutingDiagnostics::congaBackgroundBytes = 0;
uint64_t RoutingDiagnostics::delayNs = 0;
uint64_t RoutingDiagnostics::gapNs = 100000;
uint64_t RoutingDiagnostics::sampleNs = 10000;
uint64_t RoutingDiagnostics::backgroundPeriodNs = 1000000;
uint32_t RoutingDiagnostics::backgroundMode = 0;
double RoutingDiagnostics::startSeconds = 2.0;
std::string RoutingDiagnostics::output;
std::string RoutingDiagnostics::backgroundPathsFile;

void RoutingDiagnostics::LoadBackgroundPaths() {
    if (!enabled || backgroundMode != 3) return;
    std::ifstream file(backgroundPathsFile.c_str());
    if (!file.good()) NS_FATAL_ERROR("Missing exogenous background path schedule");
    uint32_t id, path;
    while (file >> id >> path) {
        if (path >= 8 || !S_backgroundPaths.emplace(id, path).second)
            NS_FATAL_ERROR("Invalid or duplicate background path assignment");
    }
    if (!file.eof()) NS_FATAL_ERROR("Malformed background path schedule");
}

unsigned RoutingDiagnostics::Group(uint32_t sourceIp) {
    if (!backgroundMode) return 0;
    uint32_t source = Settings::hostIp2IdMap.at(sourceIp) % 128;
    return source >= 32 && source < 96 ? 1 : 0;
}

void RoutingDiagnostics::Receive(uint32_t sourceIp, uint32_t seq, uint32_t expected,
                                 uint32_t size, bool ecn, bool cnp) {
    if (!enabled) return;
    auto &s = S_diagnostic[Group(sourceIp)];
    ++s.rxPackets; s.rxBytes += size;
    if (seq > expected) ++s.outOfOrder;
    if (seq + size <= expected) ++s.duplicates;
    if (ecn) ++s.ecnPackets;
    if (cnp) { ++s.reorderCnp; if (!ecn) ++s.reorderWithoutEcn; }
}

void RoutingDiagnostics::Write() {
    if (!enabled) return;
    std::ofstream f(output.c_str());
    if (!f.good()) NS_FATAL_ERROR("Cannot write routing diagnosis output");
    f << std::setprecision(17) << "{\n\"groups\":[\n";
    for (unsigned i = 0; i < 2; ++i) {
        const auto &s = S_diagnostic[i];
        f << "{\"packets\":" << s.packets << ",\"bytes\":" << s.bytes
          << ",\"route_changes\":" << s.routeChanges << ",\"decisions\":" << s.decisions
          << ",\"suboptimal_packets\":" << s.suboptimal
          << ",\"queue_regret_ns_sum\":" << s.queueRegretNs
          << ",\"chosen_delay_ns_sum\":" << s.chosenDelayNs
          << ",\"local_queue_ns_sum\":" << s.localQueueNs
          << ",\"remote_queue_ns_sum\":" << s.remoteQueueNs
          << ",\"historical_reads\":" << s.staleReads
          << ",\"missing_history\":" << s.missingHistory
          << ",\"information_age_ns_sum\":" << s.informationAgeNs
          << ",\"rx_packets\":" << s.rxPackets << ",\"rx_bytes\":" << s.rxBytes
          << ",\"out_of_order\":" << s.outOfOrder << ",\"duplicates\":" << s.duplicates
          << ",\"ecn_packets\":" << s.ecnPackets << ",\"reorder_cnp\":" << s.reorderCnp
          << ",\"reorder_without_ecn\":" << s.reorderWithoutEcn
          << ",\"proposed_changes\":" << s.proposedChanges << ",\"guarded_changes\":" << s.guardedChanges
          << ",\"missing_reports\":" << s.missingReports << "}" << (i ? "\n" : ",\n");
    }
    f << "],\"remote\":" << remote << ",\"delay_ns\":" << delayNs
      << ",\"gap_ns\":" << gapNs << ",\"ooo_cnp_enabled\":" << oooCnp
      << ",\"background_mode\":" << backgroundMode
      << ",\"feedback\":" << feedback << ",\"guard\":" << guard
      << ",\"feedback_period_ns\":" << feedbackPeriodNs << ",\"guard_margin_ns\":" << guardMarginNs
      << ",\"report_quantum_bytes\":" << reportQuantumBytes << ",\"age_guard\":" << ageGuard
      << ",\"padded_reports\":" << paddedReports
      << ",\"conga_ack_feedback\":" << congaAckFeedback << ",\"conga_ack_updates\":" << congaAckUpdates
      << ",\"conga_background_hop_bytes\":" << congaBackgroundBytes
      << ",\"report_packets\":" << S_reportPackets << ",\"report_bytes\":" << S_reportBytes
      << ",\"reports_delivered\":" << S_reportsDelivered
      << ",\"reports_rejected\":" << S_reportsRejected
      << ",\"reports_malformed\":" << S_reportsMalformed
      << ",\"reports_sent_minus_received\":" << S_reportPackets-S_reportsDelivered
      << ",\"oracle_counters_valid\":" << (!RouteLearning::mode && !feedback) << "}\n";
}

void SwitchNode::DiagnosticEmitFeedback() {
    if (m_isToR) return;
    // The spine advertises only its OWN forwarding table and output ports.
    std::map<uint32_t,uint32_t> localRoutes;
    for (const auto &route:m_rtTable) {
        if (route.second.size()!=1) NS_FATAL_ERROR("Feedback requires single spine-to-leaf next hops");
        localRoutes[route.first]=route.second.front();
    }
    RouteFeedback report;
    report.sequence=++m_routeFeedbackSequence;
    report.sampleNs=Simulator::Now().GetNanoSeconds();
    report.pressure=RouteLearning::pressureEnabled;
    for (const auto &prefix:RouteFeedbackPrefixes(localRoutes)) {
        RouteFeedback::Entry e; e.network=prefix.network; e.prefix=prefix.prefix;
        const uint32_t port=prefix.port;
        uint64_t bytes=CalculateInterfaceLoad(port);
        if (RoutingDiagnostics::reportQuantumBytes) {
            const uint64_t q=RoutingDiagnostics::reportQuantumBytes;
            bytes=((bytes+q-1)/q)*q;
        }
        if (bytes>uint64_t(UINT32_MAX)) NS_FATAL_ERROR("Queue report overflow");
        e.queueBytes=bytes;
        e.rateBps=DynamicCast<QbbNetDevice>(GetDevice(port))->GetDataRate().GetBitRate();
        if (report.pressure) {
            const auto &prefixQueue=m_prefixQueues[port];
            const uint64_t quantized=((prefixQueue.bytes+1023)/1024)*1024;
            if (quantized>UINT32_MAX || prefixQueue.packetsByFlow.size()>UINT32_MAX)
                NS_FATAL_ERROR("Prefix report overflow");
            e.prefixBytes=quantized; e.prefixFlows=prefixQueue.packetsByFlow.size();
            RouteLearning::RecordPressure('S',{report.sampleNs,GetId(),port,e.network,
                prefixQueue.bytes,e.prefixFlows,e.prefixBytes});
        }
        report.entries.push_back(e);
    }
    const std::vector<uint8_t> wire=report.Encode();
    for (uint32_t port=1; port<GetNDevices(); ++port) {
        Ptr<Packet> packet=Create<Packet>(wire.data(),wire.size());
        Ipv4Header ip; ip.SetSource(Ipv4Address(uint32_t(0))); ip.SetDestination(Ipv4Address(uint32_t(0)));
        ip.SetProtocol(RouteFeedback::PROTOCOL); ip.SetTtl(1); ip.SetPayloadSize(packet->GetSize()); packet->AddHeader(ip);
        PppHeader ppp; ppp.SetProtocol(0x0021); packet->AddHeader(ppp);
        if (wire.size()+ip.GetSerializedSize()>GetDevice(port)->GetMtu())
            NS_FATAL_ERROR("Feedback exceeds link MTU; use fewer advertised prefixes");
        CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);
        packet->PeekHeader(ch);
        // This local ingress tag serves existing MMU accounting only. The
        // feedback decoder neither reads it nor needs any simulator UID.
        FlowIdTag tag; tag.SetFlowId(Settings::CONWEAVE_CTRL_DUMMY_INDEV); packet->AddPacketTag(tag);
        ++S_reportPackets; S_reportBytes+=packet->GetSize();
        DoSwitchSend(packet,ch,port,0);
    }
}

const RouteFeedback::Entry *SwitchNode::FindRouteFeedback(uint32_t port,uint32_t destination,
                                                         double *delta,uint64_t *age) const {
    const uint64_t ttl=std::max(uint64_t(1000000),5*RoutingDiagnostics::feedbackPeriodNs);
    return m_routeFeedback.Find(port,destination,Simulator::Now().GetNanoSeconds(),ttl,delta,age);
}

double SwitchNode::ReportedPathDelay(uint32_t outIf,CustomHeader &ch,bool remote) {
    const double localRate=DynamicCast<QbbNetDevice>(m_devices[outIf])->GetDataRate().GetBitRate();
    const double local=(CalculateInterfaceLoad(outIf)+1000.)*8e9/localRate;
    uint64_t age=0;
    const auto *report=FindRouteFeedback(outIf,ch.dip,nullptr,&age);
    auto &stats=S_diagnostic[RoutingDiagnostics::Group(ch.sip)];
    if (!report) ++stats.missingReports;
    else {
        ++stats.staleReads; stats.informationAgeNs+=age;
        m_diagnosticDecisionAgeNs=std::max(m_diagnosticDecisionAgeNs,age);
    }
    // Before a report arrives, use only the local link as a nominal estimate.
    // An absent/expired report never falls back to remote simulator objects.
    const double rate=report ? report->rateBps : localRate;
    return local+((remote && report ? report->queueBytes : 0)+1000.)*8e9/rate;
}

void SwitchNode::DiagnosticSamplePorts() {
    const int64_t now = Simulator::Now().GetNanoSeconds();
    const int64_t retention=std::max(uint64_t(RoutingDiagnostics::delayNs+2*RoutingDiagnostics::sampleNs),
                                    uint64_t(RouteLearning::mode>=8 ? 1000000 : 0));
    const int64_t oldest = now - retention;
    for (uint32_t i = 1; i < GetNDevices(); ++i) {
        auto &h = m_diagnosticHistory[i];
        h.emplace_back(now, CalculateInterfaceLoad(i));
        while (h.size() > 2 && h[1].first < oldest) h.pop_front();
    }
}

double SwitchNode::DiagnosticPathDelay(uint32_t outIf, CustomHeader &ch,
                                      bool remote, bool historical) {
    if (historical && RoutingDiagnostics::feedback) return ReportedPathDelay(outIf,ch,remote);
    if (RouteLearning::mode) NS_FATAL_ERROR("Oracle remote access is forbidden during RL routing");
    Ptr<QbbNetDevice> local = DynamicCast<QbbNetDevice>(m_devices[outIf]);
    double delay = (CalculateInterfaceLoad(outIf) + 1000.0) * 8e9 / local->GetDataRate().GetBitRate();
    Ptr<Channel> channel = local->GetChannel();
    Ptr<SwitchNode> next;
    for (uint32_t i = 0; i < channel->GetNDevices(); ++i)
        if (channel->GetDevice(i)->GetNode()->GetId() != GetId())
            next = DynamicCast<SwitchNode>(channel->GetDevice(i)->GetNode());
    if (!next) NS_FATAL_ERROR("Diagnostic oracle requires a two-tier leaf-spine topology");
    const auto &nextHops = next->m_rtTable.at(ch.dip);
    if (nextHops.size() != 1) NS_FATAL_ERROR("Diagnostic oracle expects one spine-to-destination-leaf hop");
    uint32_t nextIf = nextHops[0];
    uint32_t bytes = 0;
    if (remote) bytes=next->CalculateInterfaceLoad(nextIf);
    if (remote && historical && !RoutingDiagnostics::feedback && RoutingDiagnostics::delayNs) {
        auto &s = S_diagnostic[RoutingDiagnostics::Group(ch.sip)];
        ++s.staleReads;
        const int64_t now = Simulator::Now().GetNanoSeconds();
        const int64_t cutoff = now - RoutingDiagnostics::delayNs;
        auto &history = next->m_diagnosticHistory[nextIf];
        bool found = false;
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            if (it->first <= cutoff) {
                bytes = it->second; s.informationAgeNs += now - it->first;
                found = true; break;
            }
        }
        if (!found) { bytes = 0; ++s.missingHistory; }
    }
    Ptr<QbbNetDevice> second = DynamicCast<QbbNetDevice>(next->GetDevice(nextIf));
    return delay + (bytes + 1000.0) * 8e9 / second->GetDataRate().GetBitRate();
}

uint32_t SwitchNode::DiagnosticRoute(Ptr<Packet> p, CustomHeader &ch,
                                    const std::vector<int> &nexthops) {
    if (!RoutingDiagnostics::enabled) NS_FATAL_ERROR("LB_MODE 12 requires DIAG_ENABLE");
    if (!m_isToR || !m_isToR_hostIP.count(ch.sip) || nexthops.size() == 1)
        return DoLbFlowECMP(p, ch, nexthops);
    const uint64_t key = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
    auto &flow = m_diagnosticFlows[key];
    const int64_t now = Simulator::Now().GetNanoSeconds();
    if (RouteLearning::mode==8 && flow.port) {
        flow.lastNs=now;return flow.port;
    }
    if (RouteLearning::mode==9 && flow.port) {
        const auto &ack=m_routeAcks[key];
        const bool fresh=uint64_t(ch.udp.seq)>=flow.sentEnd;
        const bool spaced=now-flow.decisionNs>=int64_t(RouteLearning::intervalNs);
        const bool drained=ack.lastNs>=0 && uint64_t(ack.sequence)>=flow.sentEnd;
        const bool gap=now-flow.lastNs>=int64_t(RouteLearning::intervalNs);
        const bool bounded=ack.lastNs>=0 && flow.sentEnd>=64000 &&
            flow.sentEnd<=uint64_t(ack.sequence)+8000 && flow.pathChanges<1;
        const bool allowed=RouteLearning::pathReselect==1 ? drained :
            RouteLearning::pathReselect==2 ? gap : RouteLearning::pathReselect==3 && bounded;
        if (!fresh || !spaced || !allowed) {flow.lastNs=now;return flow.port;}
    }
    if (RouteLearning::mode) flow.maxSequence=std::max(flow.maxSequence,ch.udp.seq);
    if (flow.firstNs < 0) { flow.firstNs=now; flow.switchNs=now; }
    if (RouteLearning::mode<8 && RouteLearning::mode && flow.port && flow.decisionNs>=0 &&
        now-flow.decisionNs < int64_t(RouteLearning::intervalNs)) {
        m_diagnosticDecisionAgeNs=0;
        flow.arrivalNs=std::max(flow.arrivalNs,now+DiagnosticPathDelay(flow.port,ch,RoutingDiagnostics::remote,true));
        flow.informationAgeNs=m_diagnosticDecisionAgeNs;
        flow.lastNs=now; return flow.port;
    }
    if (RouteLearning::mode<8 && flow.port && RoutingDiagnostics::gapNs && now - flow.lastNs <= int64_t(RoutingDiagnostics::gapNs)) {
        flow.lastNs = now;
        return flow.port;
    }
    ++S_diagnostic[RoutingDiagnostics::Group(ch.sip)].decisions;
    // Rotate tie order by a deterministic flow hash. No additional RNG draws.
    uint32_t offset = EcmpHash(reinterpret_cast<const uint8_t*>(&key), sizeof(key), m_ecmpSeed) % nexthops.size();
    double best = std::numeric_limits<double>::infinity();
    m_diagnosticDecisionAgeNs = 0;
    uint32_t port = nexthops[offset];
    std::map<uint32_t,double> scores;
    for (uint32_t i = 0; i < nexthops.size(); ++i) {
        uint32_t candidate = nexthops[(i + offset) % nexthops.size()];
        double score = DiagnosticPathDelay(candidate, ch, RoutingDiagnostics::remote, true);
        if (RouteLearning::mode) scores[candidate]=score;
        if (score < best) { best = score; port = candidate; }
    }
    if (RouteLearning::mode>=8) {
        FlowIDNUMTag id;
        if (!p->PeekPacketTag(id)) NS_FATAL_ERROR("Initial-path learning requires flow identity");
        const uint32_t destination=ch.dip;
        std::vector<RouteLearning::State> states;
        std::vector<RouteLearning::PressureState> pressure;
        std::vector<uint32_t> ports;
        double mean=0,var=0;
        for(auto s:scores) mean+=s.second/scores.size();
        for(auto s:scores) var+=(s.second-mean)*(s.second-mean)/scores.size();
        for(uint32_t i=0;i<nexthops.size();++i) {
            uint32_t candidate=nexthops[(i+offset)%nexthops.size()];ports.push_back(candidate);
            Ptr<QbbNetDevice> link=DynamicCast<QbbNetDevice>(m_devices[candidate]);
            double rate=link->GetDataRate().GetBitRate();
            double localNs=(CalculateInterfaceLoad(candidate)+1000.)*8e9/rate;
            double remoteNs=scores[candidate]-localNs;
            double remoteSerial=DiagnosticPathDelay(candidate,ch,false,true)-localNs;
            double deltaBytes=0.; uint64_t ageNs=0;
            const auto *report=FindRouteFeedback(candidate,destination,&deltaBytes,&ageNs);
            const bool known=report!=nullptr;
            if (RouteLearning::pressureEnabled) {
                const auto &localPrefix=m_prefixQueues[candidate];
                const uint32_t remoteBytes=known ? report->prefixBytes : 0;
                const uint32_t remoteFlows=known ? report->prefixFlows : 0;
                pressure.push_back(RouteLearning::PressureState{{localPrefix.bytes*8e3/rate/.2,
                    remoteBytes*remoteSerial*1e-9/.2,
                    std::log1p(double(localPrefix.packetsByFlow.size()))/4.,std::log1p(double(remoteFlows))/4.}});
                if (flow.port) RouteLearning::RecordPressure('P',{uint64_t(now),GetId(),candidate,id.GetId(),i,
                    localPrefix.bytes,localPrefix.packetsByFlow.size(),remoteBytes,remoteFlows,
                    destination,ageNs,uint64_t(rate),uint64_t(std::llround(remoteSerial))});
            }
            const double age=known ? ageNs*1e-6 : .2;
            const double remoteDelta=known ? deltaBytes*remoteSerial/1000.*1e-6 : 0.;
            // Published actors were trained with this input identically zero.
            // Enabling a local trend requires new normalization and training.
            const double localDelta=0.;
            auto work=[now](const RoutePathTraffic &t) {return t.lastNs<0 ? 0. : t.workNs*std::exp(-(now-t.lastNs)/200000.)*1e-6;};
            // Store traffic by visible destination IP, then aggregate using
            // the destination prefix learned from a RECEIVED advertisement.
            double pathWork=0.; int64_t assignedNs=-1;
            const uint32_t first=known ? report->network : destination;
            const uint32_t last=known ? report->network | ~report->Mask() : destination;
            auto it=m_routePathTraffic.lower_bound(std::make_pair(candidate,first));
            for (;it!=m_routePathTraffic.end() && it->first.first==candidate && it->first.second<=last;++it) {
                pathWork+=work(it->second); assignedNs=std::max(assignedNs,it->second.assignedNs);
            }
            const auto &portTraffic=m_routePathTraffic[std::make_pair(candidate,uint32_t(-1))];
            const double portWork=work(portTraffic);
            const double local=localNs*1e-6,remote=remoteNs*1e-6;
            const double idle=assignedNs<0 ? .2 : std::min(2.,(now-assignedNs)*1e-6);
            RouteLearning::State state{{local,remote,8e6/rate,remoteSerial*1e-6,
                std::min(2.,age),remoteDelta,localDelta,pathWork,portWork,
                local/(local+remote+1e-6),remote*std::min(2.,age),remoteDelta*std::min(2.,age),
                pathWork*std::min(2.,age),portWork*std::min(2.,age),double(known),
                std::tanh(remoteDelta/.05),std::log1p(local/.016),std::log1p(remote/.016),
                std::sqrt(var)*1e-6,(scores[candidate]-best)*1e-6,
                pathWork/(portWork+.001),idle,mean*1e-6,
                RouteLearning::mode==9 ? (8e6/rate+remoteSerial*1e-6)*std::log1p(flow.sentEnd/64000.) : 0.}};
            states.push_back(state);
        }
        int previous=-1;
        for(unsigned i=0;i<ports.size();++i) if(ports[i]==flow.port) previous=i;
        const auto &gateAck=m_routeAcks[key];
        RouteLearning::GateContext gateContext{{double(flow.sentEnd),
            std::max(0.,double(flow.sentEnd)-gateAck.sequence),
            double(gateAck.lastNs<0 ? now-flow.firstNs : now-gateAck.lastNs),gateAck.rate,
            double(now-flow.firstNs),double(flow.lastNs<0 ? 0 : now-flow.lastNs),
            gateAck.nackNs<0 ? 0. : std::exp(-double(now-gateAck.nackNs)/200000.)}};
        unsigned selected=RouteLearning::ChoosePath(id.GetId(),now,states,previous,gateContext,pressure,key);
        if (RouteLearning::mode==9)
            RouteLearning::RecordPathBoundary(id.GetId(),now,flow.sentEnd,m_routeAcks[key].sequence,ch.udp.seq,
                flow.lastNs<0 ? 0 : now-flow.lastNs,flow.port,ports[selected]);
        port=ports[selected];best=scores[port];flow.decisionNs=now;
        if (flow.port && port!=flow.port) ++flow.pathChanges;
        // A no-op boundary must not masquerade as a new path assignment.
        if (!flow.port || port!=flow.port)
            m_routePathTraffic[std::make_pair(port,destination)].assignedNs=now;
    } else if (RouteLearning::mode && flow.port && port!=flow.port) {
        FlowIDNUMTag id;
        if (!p->PeekPacketTag(id)) NS_FATAL_ERROR("Route learning requires a flow identity");
        const double old=scores.at(flow.port), base=16000.;
        double mean=0, var=0;
        for (auto s:scores) mean+=s.second/scores.size();
        for (auto s:scores) var+=(s.second-mean)*(s.second-mean)/scores.size();
        auto scaled=[](double ns) {return std::min(1.0,std::log1p(std::max(0.,ns)/16000.)/std::log(1001.));};
        double oldTrend=0, bestTrend=0;
        if(flow.previousScores.count(flow.port)) oldTrend=std::tanh((old-flow.previousScores[flow.port])/(base+flow.previousScores[flow.port]));
        if(flow.previousScores.count(port)) bestTrend=std::tanh((best-flow.previousScores[port])/(base+flow.previousScores[port]));
        double slack=now+best-flow.arrivalNs;
        uint64_t age=std::max(flow.informationAgeNs,m_diagnosticDecisionAgeNs);
        const auto &ack=m_routeAcks[key];
        double ackProgress=flow.decisionNs>=0 && now>flow.decisionNs ?
            std::min(1.,std::max(0.,double(ack.sequence)-flow.previousAck)*8./(now-flow.decisionNs)) : 0.;
        RouteLearning::State state{{scaled(old),scaled(best),scaled(mean),scaled(std::sqrt(var)),
            (old-best)/(old+base),std::tanh(slack/200000.),scaled(now-flow.lastNs),scaled(age),
            oldTrend,bestTrend,scaled(DiagnosticPathDelay(flow.port,ch,false,true)),
            scaled(DiagnosticPathDelay(port,ch,false,true)),scaled(now-flow.firstNs),scaled(now-flow.switchNs),
            double(flow.lastAction),scaled(flow.decisionNs<0 ? 0 : now-flow.decisionNs),
            scaled(flow.maxSequence*8.),scaled(ack.sequence*8.),
            scaled(std::max(0.,double(flow.maxSequence)-ack.sequence)*8.),
            scaled(ack.lastNs<0 ? now-flow.firstNs : now-ack.lastNs),ack.rate,
            ack.nackNs<0 ? 0. : std::exp(-double(now-ack.nackNs)/200000.),ackProgress,
            double(ack.lastNs>=0)}};
        unsigned action=RouteLearning::Decide(id.GetId(),now,state,slack,age,oldTrend);
        flow.lastAction=action; flow.decisionNs=now;
        flow.previousAck=ack.sequence;
        auto &s=S_diagnostic[RoutingDiagnostics::Group(ch.sip)]; ++s.proposedChanges;
        if(!action) {port=flow.port;best=old;++s.guardedChanges;}
        else flow.switchNs=now;
        flow.previousScores=scores;
    } else if (!RouteLearning::mode && RoutingDiagnostics::guard && flow.port && port != flow.port) {
        auto &s = S_diagnostic[RoutingDiagnostics::Group(ch.sip)]; ++s.proposedChanges;
        uint64_t margin = RoutingDiagnostics::guardMarginNs;
        if (RoutingDiagnostics::ageGuard)
            margin = std::max(margin, std::max(flow.informationAgeNs, m_diagnosticDecisionAgeNs));
        if (now + best < flow.arrivalNs + margin) {
            port = flow.port; ++s.guardedChanges;
            best = DiagnosticPathDelay(port, ch, RoutingDiagnostics::remote, true);
        }
    }
    if (RouteLearning::mode && flow.previousScores.empty()) flow.previousScores=scores;
    flow.arrivalNs = std::max(flow.arrivalNs, now + best);
    flow.informationAgeNs = m_diagnosticDecisionAgeNs;
    flow.port = port; flow.lastNs = now;
    return port;
}

#include "route-learning.inc"

void SwitchNode::ObservePrefixQueue(uint32_t port, Ptr<Packet> packet, const CustomHeader &ch, bool enqueue) {
    // Observable prefix packets are a proxy for short flows, never a final-size label.
    if (ch.l3Prot!=0x11 || ch.udp.seq>=8000) return;
    const uint64_t flowKey=ConWeaveRouting::GetFlowKey(ch.sip,ch.dip,ch.udp.sport,ch.udp.dport);
    auto &queue=m_prefixQueues[port];const uint32_t bytes=packet->GetSize();
    if (enqueue) {queue.bytes+=bytes;++queue.packetsByFlow[flowKey];}
    else {
        auto flow=queue.packetsByFlow.find(flowKey);
        if (queue.bytes<bytes || flow==queue.packetsByFlow.end() || !flow->second)
            NS_FATAL_ERROR("Prefix queue accounting underflow");
        queue.bytes-=bytes;
        if (!--flow->second) queue.packetsByFlow.erase(flow);
    }
    RouteLearning::RecordPressure(enqueue ? 'E' : 'D',{uint64_t(Simulator::Now().GetNanoSeconds()),GetId(),port,flowKey,
        bytes,ch.udp.seq,queue.bytes,queue.packetsByFlow.size(),packet->GetUid()});
}

void SwitchNode::DiagnosticObserve(Ptr<Packet> p, CustomHeader &ch, uint32_t outIf) {
    if (ch.l3Prot != 0x11 || !m_isToR || !m_isToR_hostIP.count(ch.sip)) return;
    auto route = m_rtTable.find(ch.dip);
    if (route == m_rtTable.end() || route->second.size() <= 1) return;
    if (RouteLearning::mode>=8) {
        int64_t now=Simulator::Now().GetNanoSeconds();
        double work=p->GetSize()*8e9/DynamicCast<QbbNetDevice>(m_devices[outIf])->GetDataRate().GetBitRate();
        for(uint32_t destination : {ch.dip,uint32_t(-1)}) {
            auto &t=m_routePathTraffic[std::make_pair(outIf,destination)];
            t.workNs=(t.lastNs<0 ? 0. : t.workNs*std::exp(-(now-t.lastNs)/200000.))+work;t.lastNs=now;
        }
        if (RouteLearning::mode==9) {
            const uint64_t flowKey=ConWeaveRouting::GetFlowKey(ch.sip,ch.dip,ch.udp.sport,ch.udp.dport);
            auto flow=m_diagnosticFlows.find(flowKey);
            if (flow!=m_diagnosticFlows.end())
                flow->second.sentEnd=std::max(flow->second.sentEnd,uint64_t(ch.udp.seq)+p->GetSize()-ch.GetSerializedSize());
        }
    }
    auto &s = S_diagnostic[RoutingDiagnostics::Group(ch.sip)];
    ++s.packets; s.bytes += p->GetSize();
    const uint64_t key = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
    auto inserted = m_diagnosticObservedPorts.emplace(key, outIf);
    if (!inserted.second && inserted.first->second != outIf) {
        ++s.routeChanges; inserted.first->second = outIf;
    }
    // Oracle counters are excluded from every RL/report-based run.
    if (RouteLearning::mode || RoutingDiagnostics::feedback) return;
    // Read-only shadow counterfactual at the actual source egress, including
    // cached-flowlet packets. Queue regret is descriptive, not an FCT bound.
    double chosen = DiagnosticPathDelay(outIf, ch, true, false);
    Ptr<QbbNetDevice> local = DynamicCast<QbbNetDevice>(m_devices[outIf]);
    s.localQueueNs += CalculateInterfaceLoad(outIf) * 8e9 / local->GetDataRate().GetBitRate();
    s.remoteQueueNs += chosen - DiagnosticPathDelay(outIf, ch, false, false);
    double best = chosen;
    for (int candidate : route->second)
        best = std::min(best, DiagnosticPathDelay(candidate, ch, true, false));
    s.chosenDelayNs += chosen;
    s.queueRegretNs += chosen - best;
    if (chosen > best + 1) ++s.suboptimal;
}

} /* namespace ns3 */
