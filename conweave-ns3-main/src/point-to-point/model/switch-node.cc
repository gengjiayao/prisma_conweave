#include "switch-node.h"

#include "assert.h"
#include "ns3/boolean.h"
#include "ns3/conweave-routing.h"
#include "ns3/double.h"
#include "ns3/flow-id-tag.h"
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
    bool control_pkt =
        (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE || ch.l3Prot == 0xFD || ch.l3Prot == 0xFC);

    // RL 覆写在 DoLbRl 中统一处理
    // === RL 未命中 -> 落回原先逻辑 ===

    if (Settings::lb_mode == 0 || control_pkt) {  // control packet (ACK, NACK, PFC, QCN)
        return DoLbFlowECMP(p, ch, nexthops);     // ECMP routing path decision (4-tuple)
    }

    switch (Settings::lb_mode) {
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

} /* namespace ns3 */
