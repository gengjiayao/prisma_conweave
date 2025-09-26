/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "conweave-obs-manager.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include <algorithm>
#include "ns3/switch-node.h" 
// additional includes
#include "ns3/packet.h"
#include "ns3/qbb-net-device.h"
#include "ns3/qbb-channel.h"
#include "ns3/data-rate.h"
#include "ns3/conweave-routing.h"
#include <array>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("ConweaveObsManager");

// Free-function thunk so MakeBoundCallback can bind 'this' and outIf
static void Conweave_OnMacTxThunk(ConweaveObsManager* self, uint32_t outIf, Ptr<const Packet> p)
{
  if (self) self->OnMacTx(outIf, p);
}


/* ****************  1. 每跳入口：挂起 + 构造观测  **************** */
void
ConweaveObsManager::OnPerHopPacket(Ptr<SwitchNode> sw,
                                   Ptr<NetDevice> inDev,
                                   Ptr<Packet> p,
                                   const CustomHeader& ch)
{
  if (m_busy) return;          // 单并发保护

  /* 1. 保存挂起上下文 */
  m_busy        = true;
  m_swHeld      = sw;
  m_pktHeld     = p;
  m_chHeld      = ch;
  m_pktUidHeld  = p->GetUid();

  /* 2. 解析目的 overlay */
  uint32_t dip = ch.dip;
  auto itTor = Settings::hostIp2SwitchId.find(dip);
  int dstOverlay = -1;
  if (itTor != Settings::hostIp2SwitchId.end()) {
    uint32_t dstTor = itTor->second;
    if (dstTor < m_nodeIdToOverlay.size()) dstOverlay = m_nodeIdToOverlay[dstTor];
  }

  /* 3. 更新包上下文（用于GetExtraInfo）*/
  SetCurrentPacketContext(m_pktUidHeld, p->GetSize());
  m_globalInjected++;  // 更新统计

  /* 4. 构造特征：[dstOverlay] + 各邻居出口队列长度 */
  std::vector<uint32_t> feats;
  feats.reserve(1 + m_overlayNeighbors.size());
  feats.push_back((uint32_t)std::max(0, dstOverlay));

  for (size_t i = 0; i < m_overlayNeighbors.size(); ++i) {
    uint32_t ifx = (i < m_egressIfs.size() ? m_egressIfs[i] : 0);
    uint32_t qlen = 0;
    if (ifx > 0 && ifx < sw->GetNDevices()) {
      Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(ifx));
      if (dev && dev->GetQueue()) qlen = dev->GetQueue()->GetNBytesTotal();
    }
    feats.push_back(qlen);
  }
  SetCurrentPacketContext(m_pktUidHeld, p->GetSize());
  /* 4. 直接走你现有的 ZMQ 发送路径 */
  PrepareAndSendObservation(dstOverlay, feats, m_pktUidHeld, sw->GetId());
}

/* ****************  2. 把观测喂给 Python（复用你现有路径）  **************** */
void
ConweaveObsManager::PrepareAndSendObservation(int dstOverlay,
                                              const std::vector<uint32_t>& feats,
                                              uint64_t pktUid,
                                              uint32_t swId)
{
  m_lastObsFeats   = feats;
  m_lastDstOverlay = dstOverlay;
  m_lastPktId      = pktUid;
  m_lastSwId       = swId;
  m_lastPktSize    = m_curPktSize;
  m_hasPrepared    = true;

  // === 构造 Python 期望的 info 串（数据包：pkt_type=0） ===
  const double now = Simulator::Now().GetSeconds();
  // 这里我们没有逐跳 e2e 延迟/成本统计，用 0 占位即可；全局计数用已有成员
  std::ostringstream oss;
  oss.setf(std::ios::fixed); oss.precision(9); // 时间用秒，小数位给足
  oss << "delay_time=" << 0.0
      << ",pkt_size=" << m_lastPktSize
      << ",curr_time=" << now
      << ",pkt_id=" << pktUid
      << ",pkt_type=" << 0
      << ",avg_e2e=" << 0.0
      << ",cost=" << 0.0
      << ",global_avg_e2e=" << 0.0
      << ",global_cost=" << 0.0
      << ",dropped=" << 0
      << ",delivered=" << 0
      << ",injected=" << m_globalInjected
      << ",buffered=" << m_globalBuffered
      << ",global_dropped=" << m_globalDropped
      << ",global_delivered=" << m_globalDelivered
      << ",global_injected=" << m_globalInjected
      << ",global_buffered=" << m_globalBuffered
      << ",signaling_overhead=" << 0
      << ",lost_list=" << DrainLostPacketsSemicolon(); // 形如 "12;45;"

  m_lastInfo = oss.str();

  // 发通知
  NotifyGymCurrentState();
}

// -- 补全：overlay 邻居索引 -> 对齐的 egress ifIndex 映射（防御：越界返回 0）
uint32_t
ConweaveObsManager::OverlayNeighborToIfIndex(int overlayNbr) const
{
  if (overlayNbr < 0) return 0;
  size_t idx = static_cast<size_t>(overlayNbr);
  if (idx >= m_egressIfs.size()) return 0;
  return m_egressIfs[idx];
}

// ======= New: Egress tracing registration and simple stats ======
void
ConweaveObsManager::RegisterEgressTracing(Ptr<SwitchNode> sw)
{
  if (!sw) return;
  // Iterate devices and try to register MacTx trace where possible
  for (uint32_t i = 1; i < sw->GetNDevices(); ++i) {
    Ptr<NetDevice> dev = sw->GetDevice(i);
    if (!dev) continue;
    // remember bw estimate
    double bw = ResolveLinkBandwidthBps(i);
    m_portStats[i].bwBps = bw;
    // try to connect to a trace named MacTx (best-effort)
    // If this trace does not exist for the device, TraceConnectWithoutContext will fail silently.
    // Use a free-function thunk because MakeBoundCallback doesn't accept member function pointers with extra bound args
    try {
      dev->TraceConnectWithoutContext("MacTx", MakeBoundCallback(&Conweave_OnMacTxThunk, this, i));
    } catch (...) {
      // ignore if the trace source doesn't exist
    }
  }
}

void
ConweaveObsManager::OnMacTx(uint32_t outIf, Ptr<const Packet> p)
{
  if (!p) return;
  auto it = m_portStats.find(outIf);
  if (it == m_portStats.end()) {
    m_portStats[outIf] = EgressPortStats();
  }
  auto &st = m_portStats[outIf];
  st.accTxBytes += p->GetSize();

  // Queue integration for fixed window (ACC style)
  double now = Simulator::Now().GetSeconds();
  int64_t q = ReadQueueBytes(outIf);
  if (q >= 0) {
    if (st.lastQueueSampleSec == 0.0) st.lastQueueSampleSec = now;
    st.queueIntBytes += double(q) * (now - st.lastQueueSampleSec);
    st.lastQueueSampleSec = now;
    
    // Keep EMA for backward compatibility
    double alpha = 0.2;
    st.avgQueueBytes = (1.0 - alpha) * st.avgQueueBytes + alpha * double(q);
  }
}

int64_t
ConweaveObsManager::ReadQueueBytes(uint32_t outIf)
{
  if (!m_sw) return -1;
  if (outIf >= m_sw->GetNDevices()) return -1;
  Ptr<NetDevice> dev = m_sw->GetDevice(outIf);
  if (!dev) return -1;
  Ptr<QbbNetDevice> qdev = DynamicCast<QbbNetDevice>(dev);
  if (qdev && qdev->GetQueue()) {
    return int64_t(qdev->GetQueue()->GetNBytesTotal());
  }
  return -1;
}

double
ConweaveObsManager::MapQueueToScore(double avgQueueBytes, double qMaxBytes)
{
  if (qMaxBytes <= 0) return 1.0;
  double q = std::max(0.0, std::min(1.0, avgQueueBytes / qMaxBytes));
  if (q < 0.01) return 1.0;
  if (q < 0.05) return 0.8;
  if (q < 0.10) return 0.6;
  if (q < 0.20) return 0.4;
  if (q < 0.40) return 0.2;
  return 0.0;
}

// ACC-style 10-step queue scoring
double ConweaveObsManager::MapQueueToScoreAcc(double L, const std::array<double,10>& E)
{
  int n = 9; // default to worst case
  for (int i = 0; i < 10; ++i) {
    if (L <= E[i]) {
      n = i;
      break;
    }
  }
  return 1.0 - 0.1 * n; // 1.0, 0.9, ..., 0.0
}

// Build 10-step thresholds based on BDP
std::array<double,10> ConweaveObsManager::BuildQueueStepsBDP(double bwBps, double rttSec)
{
  std::array<double,10> E{};
  double Lmin = 64 * 1024; // 64KB minimum
  double bdpBytes = bwBps * rttSec / 8.0;
  double Lmax = std::max(256.0*1024, std::min(bdpBytes, 16.0*1024*1024)); // BDP-based max, capped at 16MB
  double base = Lmax / Lmin;
  if (!std::isfinite(base) || base <= 1.0) base = 1.000001;
  double ratio = std::pow(base, 1.0/9.0);
  
  E[0] = Lmin;
  for (int i = 1; i < 10; ++i) {
    E[i] = E[i-1] * ratio;
  }
  return E;
}

double
ConweaveObsManager::ResolveLinkBandwidthBps(uint32_t outIf)
{
  (void) outIf;
  // Minimal safe implementation: return a reasonable default (100 Gbps)
  // to avoid depending on non-portable device/channel APIs. This ensures
  // the code compiles and yields usable reward signals (util-based)
  // while a follow-up patch can extract the real DataRate from device
  // attributes when available.
  return 100e9; // 100 Gbps
}

double
ConweaveObsManager::ComputeReward()
{
  const double kEps = 1e-9;
  auto itps = m_portStats.find(m_lastActionOutIf);
  if (itps == m_portStats.end()) {
    return 0.0; // 该端口目前还没有统计，直接返回
  }
  EgressPortStats& st = itps->second;

  double now = Simulator::Now().GetSeconds();

  if (st.winStartSec == 0.0) { // 冷启动
      st.winStartSec = now - std::max(1e-6, m_rewardWinSec);
  }
  double span = now - st.winStartSec;
  if (span <= kEps) span = kEps;      // 绝不让窗口为 0

  // === 1) 窗口未就绪：直接返回“上一次窗口值”（不更新任何状态） ===
  const double W = m_rewardWinSec;
  double byteTrigger = (st.ackBytesWin >= m_rewardByteMin);
  if (span < W && !byteTrigger) {
    auto itp = m_prevRByIf.find(m_lastActionOutIf);
    return (itp != m_prevRByIf.end()) ? itp->second : 0.0;
  }

  // === 2) 计算本窗口的 r （与你原逻辑一致） ===
  double bwBps = (st.bwBps > kEps) ? st.bwBps : ResolveLinkBandwidthBps(m_lastActionOutIf);

  double T_win = 0.0;
  {
    double denom = std::max(kEps, bwBps) * std::max(kEps, span);
    double numBits = 0.0;
    if (st.ackBytesWin > 0) numBits = 8.0 * st.ackBytesWin;
    else if (st.accTxBytes > 0) numBits = 8.0 * st.accTxBytes;
    T_win = std::min(1.0, numBits / denom);
  }

  double T_ema = 0.0;
  auto itAw2 = m_ackWins.find(m_lastActionOutIf);
  if (itAw2 == m_ackWins.end()) {
    itAw2 = m_ackWins.insert({m_lastActionOutIf, AckWin()}).first;
  }
  T_ema = std::min(1.0, std::max(0.0, itAw2->second.emaGoodput / std::max(kEps, bwBps)));
  // 调稳：更信任 EMA，减小瞬时窗口的噪声影响
  double T = (T_ema > 0.0) ? (0.3 * T_win + 0.7 * T_ema) : T_win;

  double Lavg = st.queueIntBytes / std::max(kEps, span);
  double qScore = MapQueueToScoreAcc(Lavg, st.qSteps);
  st.qSmooth = (1.0 - m_qSmoothLambda) * st.qSmooth + m_qSmoothLambda * qScore;

  double R_sw = 0.0;
  if (auto itAw = m_ackWins.find(m_lastActionOutIf); itAw != m_ackWins.end())
    R_sw = std::min(1.0, std::max(0.0, itAw->second.emaDup));

  double R_flow = -1.0;
  {
    double sumDup = 0.0; uint32_t cnt = 0;
    for (const auto &kv : m_flow2OutIf) {
      const auto &fm = kv.second;
      if (fm.outIf != m_lastActionOutIf) continue;
      if ((now - fm.ts) > m_flowMapTtlSec) continue;
      auto itfs = m_flowStats.find(kv.first);
      if (itfs == m_flowStats.end()) continue;
      sumDup += std::max(0.0, std::min(1.0, itfs->second.emaDup));
      cnt += 1;
    }
    if (cnt > 0) R_flow = sumDup / double(cnt);
  }
  double R_norm = (R_flow >= 0.0 ? R_flow : R_sw);

  double w_dup = std::max(0.0, (m_rewardWDup > 0.0 ? m_rewardWDup : 1.0 - (m_rewardWUtil + m_rewardWQueue)));
  double r_inst = m_rewardWUtil * T + m_rewardWQueue * st.qSmooth + w_dup * (1.0 - R_norm);

  double r_prev = 0.0;
  if (auto itPrev = m_prevRByIf.find(m_lastActionOutIf); itPrev != m_prevRByIf.end()) r_prev = itPrev->second;
  double r = 0.4 * r_inst + 0.6 * r_prev;

  // sanitize before logging
  auto clamp01 = [](double x){ return std::min(1.0, std::max(0.0, x)); };
  auto safe     = [](double x){ return std::isfinite(x) ? x : 0.0; };
  T_win = clamp01(safe(T_win));
  T_ema = clamp01(safe(T_ema));
  T     = clamp01(safe(T));
  st.qSmooth = clamp01(safe(st.qSmooth));
  R_sw = clamp01(safe(R_sw));
  if (R_flow >= 0.0) R_flow = clamp01(safe(R_flow));
  r = safe(r);

  // === 3) 把“窗口内增量”缓存到 pending：delta = r - r_prev ===
  double delta = r - r_prev;
  m_pendingDeltaByIf[m_lastActionOutIf] += delta;
  m_prevRByIf[m_lastActionOutIf] = r;

  // === 4) 重置窗口累计 ===
  st.winStartSec = now;
  st.lastQueueSampleSec = now;
  st.ackBytesWin = 0;
  st.queueIntBytes = 0.0;
  st.accTxBytes = 0;

  // === 5) 日志关闭：禁用 [RW] win_close 打印，避免同步 I/O 降速 ===
  // （保留空块以便将来恢复）
  if (true) { // 启用周期性进度日志（降频）
    static double lastPrint = 0.0;
    const double printInterval = 0.00050; // 每 0.5ms 打印一次仿真进度
    if (now - lastPrint >= printInterval) {
      NS_LOG_UNCOND("[PROGRESS] t=" << now);
      lastPrint = now;
    }
  }
  return r;
}

void
ConweaveObsManager::ReportAckOnIngress(Ptr<const Packet> p)
{
  if (!p) return;
  // Try to parse the CustomHeader to attribute ACKs to a flow
  Ptr<Packet> pc = p->Copy();
  CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);
  pc->PeekHeader(ch);

  uint64_t flowKey = 0;
  // Use UDP/TCP 4-tuple if available, otherwise fallback to sip/dip alone
  if (ch.udp.sport || ch.udp.dport) {
    flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
  } else if (ch.tcp.sport || ch.tcp.dport) {
    flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.tcp.sport, ch.tcp.dport);
  } else {
    // ports zeroed; use IPs only by packing them
    flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, 0, 0);
  }

  uint32_t outIf = 0;
  auto it = m_flow2OutIf.find(flowKey);
  if (it != m_flow2OutIf.end()) {
    double age = Simulator::Now().GetSeconds() - it->second.ts;
    if (age > m_flowMapTtlSec) {
      // mapping too old, erase it and fall back
      // NS_LOG_INFO("[ACK] mapping expired key=" << flowKey << " age=" << age);
      m_flow2OutIf.erase(it);
    } else {
      outIf = it->second.outIf;
    }
  }
  if (outIf == 0) {
    //NS_LOG_UNCOND("[ACK] miss key=" << flowKey << " (no outIf)");
    return;
  }
  if (m_portStats.find(outIf) == m_portStats.end()) m_portStats[outIf] = EgressPortStats();
  auto &st = m_portStats[outIf];
  auto &awin = m_ackWins[outIf];
  // Per-flow stats record
  auto &fst = m_flowStats[flowKey];

  // Count ack packet, but compute confirmed payload bytes using ackNo delta
  st.accAckPkts += 1;
  awin.acks += 1;
  fst.ackCount += 1;

  // Dup/ACK detection: prefer tcp.ack if present
  uint64_t ackNo = 0;
  if (ch.tcp.ack) ackNo = ch.tcp.ack;
  else if (ch.udp.seq) ackNo = ch.udp.seq; // fallback

  uint64_t deltaBytes = 0;
  double now = Simulator::Now().GetSeconds();
  if (ackNo != 0) {
    if (awin.lastAckNo == ackNo) {
      awin.dupacks += 1;
      fst.dupAckCount += 1;
    } else {
      // sequence wrap/ordering safe diff: only count positive advancement
      if (ackNo > awin.lastAckNo) {
        deltaBytes = ackNo - awin.lastAckNo;
      } else {
        // ackNo <= lastAckNo -> treat as dup/ordering
        awin.dupacks += 1;
        fst.dupAckCount += 1;
        deltaBytes = 0;
      }
      awin.lastAckNo = ackNo;
      fst.lastAckNo = ackNo;
    }
  }

  if (deltaBytes > 0) {
    st.accAckBytes += deltaBytes;
    awin.bytes += deltaBytes;
    st.ackBytesWin += deltaBytes; // Add to fixed window
    
    // compute instantaneous goodput estimate for this ack
    if (awin.lastAckTs == 0.0) awin.lastAckTs = now - std::max(1e-6, m_rewardWinSec);
    double dt = std::max(5e-6, now - awin.lastAckTs); // floor dt to avoid divide-by-tiny
    double goodput_bps = (double)deltaBytes * 8.0 / dt;
    // 调稳：ACK 吞吐 EMA 放慢，降低抖动
    double beta = 0.05;
    awin.emaGoodput = (1.0 - beta) * awin.emaGoodput + beta * goodput_bps;
    awin.lastAckTs = now;
    
  // 关闭 ACK 打印，避免大量 I/O
  if (false) {
    static uint64_t ackPrint = 0; (void)ackPrint;
    NS_LOG_UNCOND("[ACK] outIf=" << outIf << " key=" << flowKey << " ack=" << ackNo
                   << " dbytes=" << deltaBytes << " dt=" << dt << " bps=" << goodput_bps << " ema=" << awin.emaGoodput);
  }
  }

  // Update EMA estimates (beta set to 0.15) using window seconds
  // 调稳：窗内 T/dup 的 EMA 放慢
  double beta2 = 0.05;
  double bw = st.bwBps > 0 ? st.bwBps : ResolveLinkBandwidthBps(outIf);
  double instT = 0.0;
  double winSec = std::max(1e-6, m_rewardWinSec); // avoid zero
  if (bw > 0) {
    instT = std::min(1.0, (double(awin.bytes) * 8.0) / (bw * winSec));
  }
  awin.emaT = (1.0 - beta2) * awin.emaT + beta2 * instT;
  double dupFrac = awin.acks > 0 ? double(awin.dupacks) / double(std::max<uint32_t>(1, awin.acks)) : 0.0;
  awin.emaDup = (1.0 - beta2) * awin.emaDup + beta2 * dupFrac;
  // Flow-level emaDup (optional, mainly for debugging per-flow disorder)
  {
    double flowDupFrac = fst.ackCount > 0 ? double(fst.dupAckCount) / double(std::max<uint64_t>(1, fst.ackCount)) : 0.0;
    fst.emaDup = (1.0 - beta2) * fst.emaDup + beta2 * flowDupFrac;
    fst.lastTs = now;
  }

  // Reset window accumulators so next ACKs build a fresh window
  awin.bytes = 0;
  awin.acks = 0;
  awin.dupacks = 0;
}


void
ConweaveObsManager::NotifyGymCurrentState()
{
  if (!m_notifyCb.IsNull()) {
    m_notifyCb();  // 等价于 env->Notify()
  } else {
    NS_LOG_WARN("NotifyGymCurrentState(): notify callback is null; step not sent.");
  }
}

void
ConweaveObsManager::Configure(Ptr<SwitchNode> sw,
                              const std::vector<int>& overlayNeighbors,
                              const std::map<int,uint32_t>& indexToSwitch,
                              const std::vector<int>& nodeIdToOverlay)
{
  m_sw = sw;
  m_overlayNeighbors = overlayNeighbors;
  m_indexToSwitch = indexToSwitch;
  m_nodeIdToOverlay = nodeIdToOverlay;
  m_egressIfs.clear();

  NS_ASSERT_MSG(m_sw, "ConweaveObsManager: SwitchNode is null");

  // 1) 计算观测上界（优先用 MMU 总字节数）
  if (m_sw->m_mmu && m_sw->m_mmu->GetMmuBufferBytes() > 0) {
    m_upperBound = (double)m_sw->m_mmu->GetMmuBufferBytes();
  } else {
    m_upperBound = 1e9; // 兜底，不会太小
  }

  // 2) 把 overlay 邻居映射成出端口 ifIndex（与 overlayNeighbors 顺序严格对齐）
  for (int ovIdx : m_overlayNeighbors)
  {
    auto it = m_indexToSwitch.find(ovIdx);
    NS_ASSERT_MSG(it != m_indexToSwitch.end(), "overlay index not in index->switch map");
    uint32_t neighborId = it->second;

    uint32_t ifx = ResolveEgressIfToNeighbor(neighborId);
    NS_ASSERT_MSG(ifx > 0, "No egress ifIndex found from this switch to neighborId=" << neighborId);
    m_egressIfs.push_back(ifx);

    //调试用
    NS_LOG_UNCOND("[ObsCfg] sw=" << m_sw->GetId()
    << " overlayNbr=" << ovIdx
    << " -> nbrSwId=" << neighborId
    << " ifIndex=" << ifx);
  }

  // 对每个端口尝试挂MacTxtrace统计accTxBytes还有队列积分,也把bwBps赋初值
  RegisterEgressTracing(m_sw);
  m_winStartSec = Simulator::Now().GetSeconds();
  m_lastRewardTimeSec = m_winStartSec;

  // Initialize ACC thresholds for each port
  double now = Simulator::Now().GetSeconds();
  for (uint32_t outIf : m_egressIfs) {
    auto& st = m_portStats[outIf];
    st.bwBps = ResolveLinkBandwidthBps(outIf);
    st.winStartSec = now;
    st.lastQueueSampleSec = now;
    st.qSteps = BuildQueueStepsBDP(st.bwBps, m_rttGuessSec);
    // Pre-initialize ack window state for this port to avoid later lookups on missing keys
    (void)m_ackWins[outIf];
  }

  // Set adaptive TTL
  m_flowMapTtlSec = std::max(5e-4, 200.0 * std::max(m_rttGuessSec, m_rewardWinSec));
  NS_LOG_UNCOND("[Config] Set adaptive flowMapTtlSec to " << m_flowMapTtlSec * 1000 << " ms");

  m_ready = true;
  NS_LOG_INFO("ConweaveObsManager configured for sw " << m_sw->GetId()
              << " with " << m_egressIfs.size() << " neighbors.");
}

//下放逻辑
Ptr<OpenGymSpace>
ConweaveObsManager::GetActionSpace() const
{
  uint32_t n = m_overlayNeighbors.empty() ? 1u : (uint32_t)m_overlayNeighbors.size();
  return CreateObject<OpenGymDiscreteSpace>(n);
}

Ptr<OpenGymSpace>
ConweaveObsManager::GetObservationSpace() const
{
  uint32_t dim = 1u + (uint32_t)m_overlayNeighbors.size();
  double high = std::max(1.0, m_upperBound);
  return CreateObject<OpenGymBoxSpace>(0.0, high, std::vector<uint32_t>{dim}, TypeNameGet<float>());
}

uint32_t
ConweaveObsManager::ResolveEgressIfToNeighbor(uint32_t neighborNodeId) const
{
  // 遍历本机所有设备，查该 NetDevice 的 QbbChannel 对端是否是 neighborNodeId
  for (uint32_t i = 1; i < m_sw->GetNDevices(); ++i) {
    auto dev = DynamicCast<QbbNetDevice>(m_sw->GetDevice(i));
    if (!dev) continue;
    auto ch = DynamicCast<QbbChannel>(dev->GetChannel());
    if (!ch) continue;

    Ptr<NetDevice> d0 = ch->GetDevice(0);
    Ptr<NetDevice> d1 = ch->GetDevice(1);

    Ptr<Node> peer = (d0 == dev) ? d1->GetNode() : d0->GetNode();
    if (peer && peer->GetId() == neighborNodeId) {
      return i; // ifIndex
    }
  }
  return 0; // not found
}

// 构造观测：[dstOverlay] + [每邻居端口的队列字节数]
std::vector<float>
ConweaveObsManager::BuildObservation(int currentDstOverlay) const
{
  std::vector<float> out;
  out.reserve(1 + m_egressIfs.size());

  // [0] 放目的 overlay id（你之前约定好的）
  out.push_back(static_cast<float>(currentDstOverlay));

  if (!m_ready) {
    // 若未 ready，邻居部分填 0
    for (size_t k = 0; k < m_overlayNeighbors.size(); ++k) out.push_back(0.0f);
    return out;
  }

  // 按照 overlayNeighbors 顺序，对齐取 ifIndex 对应端口的总队列字节数
  for (uint32_t ifx : m_egressIfs) {
    float qbytes = 0.0f;
    auto dev = DynamicCast<QbbNetDevice>(m_sw->GetDevice(ifx));
    if (dev && dev->GetQueue()) {
      qbytes = static_cast<float>(dev->GetQueue()->GetNBytesTotal());
    }
    out.push_back(qbytes);
    
  }
  return out;
}

float
ConweaveObsManager::GetReward()
{
  // 先尝试结算一次窗口（若未到触发条件，ComputeReward 会立即返回且不改状态）
  (void)ComputeReward();

  // 改为：按 Agent 聚合发放 reward —— 汇总并清零所有端口的 pending 增量
  double reward = 0.0;
  for (auto &kv : m_pendingDeltaByIf) {
    reward += kv.second;
    kv.second = 0.0; // 清零，避免重复累计
  }
  return static_cast<float>(reward);
}

bool
ConweaveObsManager::ApplyAction(uint32_t actionId, int dstOverlay)
{
  // if (m_overlayNeighbors.empty()) return true;
  // if (actionId >= m_overlayNeighbors.size()) actionId = 0;

  // // 1) 选中的邻居 overlay
  // const int nbrOverlay = m_overlayNeighbors[actionId];

  // // 2) 对应本机出口 ifIndex（与 overlay 顺序对齐）
  // const uint32_t outIf = m_egressIfs[actionId];
  // if(outIf == 0) return true; // 邻居未解析到有效端口,忽略本次action

  // // 3) 目的 ToR（由 dstOverlay 找真实 switch id）
  // auto it = m_indexToSwitch.find(dstOverlay);
  // if (it == m_indexToSwitch.end()) return true;
  // uint32_t dstToR = it->second;

  // // 4) 下发给 ConWeave 路由逻辑9.11
  // if(m_sw) {
  //   m_sw->SetRlPreferredOutIf(dstToR, outIf);
  // }
  // // m_sw->m_mmu->m_conweaveRouting.SetPreferredOutIf(dstToR, outIf);
  // NS_LOG_UNCOND("[Action] t=" << Simulator::Now().GetSeconds()
  //   << " sw=" << m_sw->GetId()
  //   << " dstToR=" << dstToR
  //   << " actionId=" << actionId
  //   << " outIf=" << outIf);

  // return true;
  if (!m_busy || !m_swHeld) return true;

  /* 3.1 合法性检查 */
  if (m_overlayNeighbors.empty()) { m_busy = false; return true; }
  if (actionId >= m_overlayNeighbors.size()) actionId = 0;

  /* 3.2 选出口 */
  const uint32_t outIf = (actionId < m_egressIfs.size() ? m_egressIfs[actionId] : 0);
  if (outIf == 0) { m_busy = false; return true; }

  /* 3.3 直接放行：调用 SwitchNode 的 RlRelease */
  // attribute last action to this outIf for reward accounting
  m_lastActionOutIf = outIf;
  // map flow -> outIf so later ACKs can be attributed deterministically
  uint64_t flowKey = 0;
  uint64_t revKey = 0;
  try {
    // try to compute flowKey from held header if available
    if (m_chHeld.udp.sport || m_chHeld.udp.dport || m_chHeld.udp.seq) {
      flowKey = ConWeaveRouting::GetFlowKey(m_chHeld.sip, m_chHeld.dip, m_chHeld.udp.sport, m_chHeld.udp.dport);
      revKey = ConWeaveRouting::GetFlowKey(m_chHeld.dip, m_chHeld.sip, m_chHeld.udp.dport, m_chHeld.udp.sport);
    } else {
      flowKey = ConWeaveRouting::GetFlowKey(m_chHeld.sip, m_chHeld.dip, 0, 0);
      revKey = ConWeaveRouting::GetFlowKey(m_chHeld.dip, m_chHeld.sip, 0, 0);
    }
    if (flowKey != 0) {
      ConweaveObsManager::FlowMapping fm;
      fm.outIf = outIf;
      fm.ts = Simulator::Now().GetSeconds();
      m_flow2OutIf[flowKey] = fm;
      if (revKey != 0) {
        m_flow2OutIf[revKey] = fm;
      }
    }
  } catch(...) {
    // ignore if header malformed
  }
  m_swHeld->RlRelease(outIf);

  // Verbose-only: action trace disabled by default
  // NS_LOG_UNCOND("[Action] t=" << Simulator::Now().GetSeconds()
  //               << " sw=" << m_swHeld->GetId()
  //               << " pktUid=" << m_pktUidHeld
  //               << " dstOverlay=" << dstOverlay
  //               << " actionId=" << actionId
  //               << " outIf=" << outIf);

  /* 3.4 清理挂起上下文 */
  m_busy        = false;
  m_swHeld      = nullptr;
  m_pktHeld     = nullptr;
  m_pktUidHeld  = 0;
  ClearCurrentPacketContext();
  ResetPreparedObservation();
  // NS_LOG_UNCOND("[RL] snapshot reset after action");
  return true;
}

void
ConweaveObsManager::ResetPreparedObservation()
{
  m_lastObsFeats.clear();
  m_lastDstOverlay = -1;
  m_lastPktId = 0;
  m_lastPktSize = 0;
  m_hasPrepared = false;
}

//9.11新增
void
ConweaveObsManager::OnDelivered(uint64_t uid, double /*nowSec*/)
{
  // 最小实现：只产生“交付事件脉冲”并累计全局计数；
  // 注入计数在无损场景下与交付等同（避免 Python 端 injected=0 导致除零）
  m_eventDelivered = true;
  m_eventDelaySec = 0.0; // 如需 e2e 延迟，后续在源 ToR 记录首见时间再计算
  m_globalDelivered += 1;
  m_globalInjected  += 1;
  // 如需 drop：在 ReportPacketDrop 里累计 m_globalDropped
  (void)uid;
}

bool
ConweaveObsManager::ConsumeDeliveryEvent(double& delayOut)
{
  if (!m_eventDelivered) {
    delayOut = 0.0;
    return false;
  }
  delayOut = m_eventDelaySec;
  m_eventDelivered = false;
  m_eventDelaySec = 0.0;
  return true;
}

std::string
ConweaveObsManager::DrainLostPacketsSemicolon()
{
  if (m_lostUids.empty()) return std::string(); // 让上层用 ";" 兜底
  
  std::string s;
  s.reserve(m_lostUids.size() * 12);
  for (auto uid : m_lostUids) {
    s += std::to_string(uid);
    s += ';';
  }
  m_lostUids.clear();
  return s;
}

} // namespace ns3
