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
  #include "ns3/conga-routing.h"
  #include <array>
  #include <cmath>
  #include <sstream>
  #include <unordered_map>
  #include <atomic>
  #include <mutex>
  #include <fstream>
  #include <cstdlib> // for std::getenv
  #include <iomanip> // for std::setprecision

  namespace {
    std::atomic<uint64_t> S_seen{0}, S_gated{0}, S_bypass{0}, S_back{0}, S_fwd{0};
    // Flowlet decision coverage and explicit routing outcomes.
    std::atomic<uint64_t> S_flowletNew{0};
    std::atomic<uint64_t> S_flowletReusePkts{0};
    std::atomic<uint64_t> S_flowletDwellReuse{0};
    std::atomic<uint64_t> S_flowletFallbackPkts{0};
    // NEW: 采纳一致性 + 长环路抽样，进程级聚合
    std::atomic<uint64_t> S_decChosen{0}, S_decMatch{0}, S_decMismatch{0};
    std::atomic<uint64_t> S_longLoopSample{0}, S_longLoopHit{0};
    // NEW: ADH 字节加权统计
    std::atomic<uint64_t> S_decMatchBytes{0}, S_decMismatchBytes{0};

    // NEW: 按端口统计 dupACK/NACK，用于诊断乱序来源 (按流聚合)
    struct FlowSeq {
      uint32_t last_seq = 0;
      uint32_t streak   = 0;   // 连续 repeat 的长度（可留作诊断）
      double   ts       = 0.0;
    };
    struct DupAckPortStat {
      uint64_t total_ctrl = 0;
      uint64_t ack_total  = 0;
      uint64_t nack_total = 0;

      // 新增：分解 repeat / backstep / advance
      uint64_t repeat_total   = 0;   // == 的总次数
      uint64_t backstep_cnt   = 0;   // < 的总次数（我们认为的“dup(乱序)”）
      uint64_t backstep_bytes = 0;   // 可选：观测回退幅度
      uint64_t advance_cnt    = 0;   // > 的总次数

      uint64_t max_streak = 0;
      std::unordered_map<uint64_t, FlowSeq> flow_seq; // flowKey -> FlowSeq
    };
    std::map<uint32_t, DupAckPortStat> S_dupByIf; // key = outIf
    std::mutex S_dupMutex;
  }

  namespace ns3 {

  NS_LOG_COMPONENT_DEFINE("ConweaveObsManager");
  void
  ConweaveObsManager::EnsureRewardCsvOpen()
  {
    if (m_rewardCsvOpened) return;
    const char* outEnv = std::getenv("MIX_OUTPUT_DIR");
    std::string dir = (outEnv && *outEnv) ? std::string(outEnv) : std::string(".");

    std::ostringstream oss;
    oss << dir << "/reward_breakdown_sw"
        << (m_sw ? m_sw->GetId() : 0)
        << ".csv";
    
    m_rewardCsv.open(oss.str(), std::ios::out | std::ios::trunc);
    if (!m_rewardCsv.is_open()){
      NS_LOG_WARN("ConweaveObsManager: Failed to open reward CSV at " << oss.str());
      return;
    }

    // 写CSV表头
    m_rewardCsv << "time_sec,out_if,T,capacity_norm,headroom,qSmooth,avg_queue_bytes,q_score,queue_trend,R_norm,R_qcn,r_inst,r_level,delta,"
                  "bw_bps,rtt_sec,bdp_bytes,q_lmin,q_lmax\n";
    m_rewardCsv.flush();
    m_rewardCsvOpened = true;
    NS_LOG_INFO(" [RewardCSV] opened " << oss.str());
  }

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
                                    CustomHeader& ch)
  {
    m_seenPkts++; S_seen++;
    if (m_busy) { 
      m_bypassPkts++; S_bypass++; 
      SwitchNode::IncrementRlBusySkip(); // <--- 在这里更新全局统计
      sw->SendToDevContinue(p, ch); // <---【重要】被绕过的包需要继续转发
      return;
    }

    // 轻量采样：使用局部 uid，不依赖挂起上下文
    uint64_t uidLocal = p->GetUid();
    if ((uidLocal % 1024) == 0) { // 抽样 1/1024
      m_longLoopSample++; S_longLoopSample++;
      auto &rec = m_seenSwByUid[uidLocal];
      if (rec.first == sw->GetId()) { m_longLoopHit++; S_longLoopHit++; }  // 再次进入同一交换机
      rec.first = sw->GetId();
      rec.second++;
      if (rec.second > 64) m_seenSwByUid.erase(uidLocal); // 避免泄露
    }

    if (inDev) {
      m_lastInIf = inDev->GetIfIndex();
    }

    /* 2. 解析目的 overlay + flowKey */
    uint32_t dip = ch.dip;
    auto itTor = Settings::hostIp2SwitchId.find(dip);
    int dstOverlay = -1;
    if (itTor != Settings::hostIp2SwitchId.end()) {
      uint32_t dstTorId = itTor->second;
      if (dstTorId < m_nodeIdToOverlay.size()) dstOverlay = m_nodeIdToOverlay[dstTorId];
    }

    // 生成 flowKey（UDP/TCP，否则回退为 IP 对）
    uint64_t flowKey = 0;
    if (ch.udp.sport || ch.udp.dport) {
      flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.udp.sport, ch.udp.dport);
    } else if (ch.tcp.sport || ch.tcp.dport) {
      flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, ch.tcp.sport, ch.tcp.dport);
    } else {
      flowKey = ConWeaveRouting::GetFlowKey(ch.sip, ch.dip, 0, 0);
    }

    // ===== Flowlet gating: 仅在新 flowlet 首包且允许切换时挂起，请求动作 =====
    double now = Simulator::Now().GetSeconds();
    auto &ctx = m_flowlets[flowKey];
    bool firstEver = (ctx.lastSeenSec <= 0.0);
    bool newFlowlet = false;
    if (firstEver) newFlowlet = true; else if ((now - ctx.lastSeenSec) > m_flowletGapSec) newFlowlet = true;
    if (newFlowlet) { S_flowletNew++; }
    
    // v8 keeps the dwell gate explicit for diagnostics.  Its configured
    // value is zero because the flowlet gap already prevents switching
    // inside a flowlet; suppressing decisions for another 1.5 ms silently
    // reused stale routes across dozens of real flowlet boundaries.
    bool canSwitch = (now >= ctx.lockedUntilSec);
    
    ctx.lastSeenSec = now;
    ctx.pktsInFlowlet += 1;
    ctx.bytesInFlowlet += p->GetSize();

    if (newFlowlet && canSwitch && !ctx.awaitingAction) {
      ctx.awaitingAction = true;
      ctx.startSec = now;
      ctx.pktsInFlowlet = 1;
      ctx.bytesInFlowlet = p->GetSize();

      // -- 更新全局 held 统计 --
      SwitchNode::IncrementRlHeld();

      // —— 原有“挂起 + 构造观测”路径 ——
      m_busy        = true;
      m_swHeld      = sw;
      m_pktHeld     = p;
      m_chHeld      = ch;
      m_pktUidHeld  = p->GetUid();
      m_flowletKeyHeld = flowKey;

    /* 3. 更新包上下文（用于GetExtraInfo）；仅首次见到该包累计 injected 并记录首见时间 */
    SetCurrentPacketContext(m_pktUidHeld, p->GetSize());
    if (m_seen.find(m_pktUidHeld) == m_seen.end()) {
      m_seen[m_pktUidHeld] = true;
      m_firstSeenSec[m_pktUidHeld] = Simulator::Now().GetSeconds();
      m_globalInjected++;
    }

    /* 4. Build the rl-core-v8 observation from statistics that are actually
     * maintained while lb=rl:
     * [dstOverlay, lastAction] + per-egress
     * (queueOccupancy, queueEma, dreUtilization, queueTrend,
     *  capacityHeadroom, relativeCapacity).
     *
     * CONGA's GetOneHopMetrics cannot be used here.  CONGA initialization and
     * RouteInput run only for lb_mode=3, whereas RL uses lb_mode=7 and takes
     * over before the CONGA path.  Reading it from RL produced four constants
     * in output 568075741 despite appearing to be a real congestion signal.
     */
    std::vector<uint32_t> feats;
    feats.reserve(2 + m_overlayNeighbors.size() * kFeatsPerEgress);
    feats.push_back((uint32_t)std::max(0, dstOverlay));

    const double obsScale = 5000.0;
    double maxEgressBandwidthBps = 1.0;
    for (uint32_t egressIf : m_egressIfs) {
      maxEgressBandwidthBps = std::max(
        maxEgressBandwidthBps,
        ResolveLinkBandwidthBps(egressIf)
      );
    }
    
    // 新增：上一步动作 ID（首次为 0，后续为实际动作）
    uint32_t lastAction = 0;
    if (flowKey != 0) {
      auto itFlow = m_flowlets.find(flowKey);
      if (itFlow != m_flowlets.end() && itFlow->second.lastActionOutIf > 0) {
        // 找到上次选的 outIf 对应的 overlay neighbor index
        for (size_t i = 0; i < m_egressIfs.size(); ++i) {
          if (m_egressIfs[i] == itFlow->second.lastActionOutIf) {
            lastAction = (uint32_t)i;
            break;
          }
        }
      }
    }
    feats.push_back(lastAction);

    const double observationTimeSec = Simulator::Now().GetSeconds();
    auto clamp01 = [](double x) {
      if (!std::isfinite(x)) return 0.0;
      return std::min(1.0, std::max(0.0, x));
    };
    auto quantize = [&](double x) {
      return static_cast<uint32_t>(std::lround(clamp01(x) * obsScale));
    };

    for (size_t i = 0; i < m_overlayNeighbors.size(); ++i) {
      uint32_t ifx = (i < m_egressIfs.size() ? m_egressIfs[i] : 0);
      double egressBandwidthBps = ResolveLinkBandwidthBps(ifx);
      EgressPortStats& st = m_portStats[ifx];
      st.bwBps = egressBandwidthBps;

      // Sample every candidate at decision time.  MacTx still supplies the
      // high-rate samples, while this read makes idle-port decay observable.
      UpdateQueueStats(ifx, observationTimeSec);
      double utilization = DecayAndGetDreUtilization(ifx, observationTimeSec);
      double capacity = clamp01(egressBandwidthBps / maxEgressBandwidthBps);
      double headroom = clamp01(capacity * (1.0 - utilization));
      double qRef = QueueReferenceBytes(st);
      double queueOccupancy = clamp01(st.lastQueueBytes / (st.lastQueueBytes + qRef));
      double queueEma = clamp01(st.avgQueueBytes / (st.avgQueueBytes + qRef));
      double queueTrend = clamp01(0.5 * (QueueTrendNormalized(st) + 1.0));

      feats.push_back(quantize(queueOccupancy));
      feats.push_back(quantize(queueEma));
      feats.push_back(quantize(utilization));
      feats.push_back(quantize(queueTrend));
      feats.push_back(quantize(headroom));
      feats.push_back(quantize(capacity));
    }
      
      /* 4. 直接走你现有的 ZMQ 发送路径 */
      PrepareAndSendObservation(dstOverlay, feats, m_pktUidHeld, sw->GetId());
      return; // 仅首包挂起，其余包不挂起
    }

    // Every packet that does not need a fresh decision must still use the
    // action owned by this RL flowlet context.  Sending it through
    // SendToDevContinue used a second, shorter cache in SwitchNode; when that
    // cache expired during the old dwell lock the packet silently fell back
    // to ECMP.  Direct release makes the ownership contract unambiguous.
    if (ctx.lastActionOutIf > 0) {
      if (m_rlHeldUids.size() > 50000) m_rlHeldUids.clear();
      m_rlHeldUids.insert(p->GetUid());
      if (newFlowlet) {
        S_flowletDwellReuse++;
      } else {
        S_flowletReusePkts++;
      }
      sw->RlRelease(p, ch, ctx.lastActionOutIf);
      return;
    }

    // This is an explicitly counted defensive fallback, never an implicit
    // part of the normal RL route.  A correct v8 run must report zero.
    S_flowletFallbackPkts++;
    sw->SendToDevContinue(p, ch);
  }

  /* ****************  2. 把观测喂给 Python（复用你现有路径）  **************** */
  void
  ConweaveObsManager::PrepareAndSendObservation(int dstOverlay,
                                                const std::vector<uint32_t>& feats,
                                                uint64_t pktUid,
                                                uint32_t swId)
  {
    m_gatedPkts++; S_gated++;
    m_lastObsFeats   = feats;
    m_lastDstOverlay = dstOverlay;
    m_lastPktId      = pktUid;
    m_lastSwId       = swId;
    m_lastPktSize    = m_curPktSize;
    m_hasPrepared    = true;

    // === 构造 Python 期望的 info 串（数据包：pkt_type=0） ===
    const double now = Simulator::Now().GetSeconds();
    // 计算全局平均 e2e 与 cost（示例定义：与 gAvgE2E 一致）
    const double gAvgE2E = (m_globalDelivered > 0) ? (m_sumE2E / std::max(1.0, (double)m_globalDelivered)) : 0.0;
    const double gCost   = gAvgE2E;
    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss.precision(9); // 时间用秒，小数位给足
    oss << "delay_time=" << m_lastDeliveredDelaySec
        << ",pkt_size=" << m_lastPktSize
        << ",curr_time=" << now
        << ",pkt_id=" << pktUid
        << ",pkt_type=" << 0
        << ",avg_e2e=" << gAvgE2E
        << ",cost=" << gCost
        << ",global_avg_e2e=" << gAvgE2E
        << ",global_cost=" << gCost
        << ",dropped=" << m_stepDropped
        << ",delivered=" << m_stepDelivered
        << ",injected=" << m_globalInjected
        << ",buffered=" << m_stepBuffered
        << ",global_dropped=" << m_globalDropped
        << ",global_delivered=" << m_globalDelivered
        << ",global_injected=" << m_globalInjected
        << ",global_buffered=" << m_globalBuffered
        << ",signaling_overhead=" << 0
        << ",lost_list=" << DrainLostPacketsSemicolon() // 形如 "12;45;"
        // [Design B 2026-05-14] Warm-start IL: 计算 ECMP 期望动作并加入 info
        // Python 端用作监督学习的 target action
        << ",ecmp_action=" << ComputeEcmpActionForHeldPacket();

    m_lastInfo = oss.str();

    // 发通知
    NotifyGymCurrentState();
    // 重置步进统计，确保下一步只反映增量
    m_stepDelivered = 0;
    m_stepDropped   = 0;
    m_stepBuffered  = 0;
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
    if (p) {
      auto uid = p->GetUid();
      auto it = m_decisionByUid.find(uid);
      if (it != m_decisionByUid.end()) {
        if (it->second == outIf) {
          m_decMatch++; S_decMatch++;
          S_decMatchBytes += p->GetSize();
        } else {
          m_decMismatch++; S_decMismatch++;
          S_decMismatchBytes += p->GetSize();
        }
        m_decisionByUid.erase(it); // 只统计第一次命中
      }
    }
    if (!p) return;
    auto it = m_portStats.find(outIf);
    if (it == m_portStats.end()) {
      m_portStats[outIf] = EgressPortStats();
    }
    auto &st = m_portStats[outIf];
    st.accTxBytes += p->GetSize();

    // Update the RL-native DRE.  Decay is shared with observation/reward reads
    // so an idle port cannot retain stale utilization indefinitely.
    {
      bool isRlHeld = false;
      auto rit = m_rlHeldUids.find(p->GetUid());
      if (rit != m_rlHeldUids.end()) {
        isRlHeld = true;
        m_rlHeldUids.erase(rit);  // 一次性消费
      }
      double now_dre = Simulator::Now().GetSeconds();
      (void)DecayAndGetDreUtilization(outIf, now_dre);
      if (isRlHeld) {
        st.dre_bytes += static_cast<double>(p->GetSize());
      }
    }

    UpdateQueueStats(outIf, Simulator::Now().GetSeconds());
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
    double bdpBytes = bwBps * rttSec / 8.0;

    // 新刻度：在 BDP 的 1/4 ~ 4 倍之间等比 10 段，并夹在 [32KB, 16MB]
    double Lmin = std::max(32.0*1024, 0.25 * bdpBytes);
    double Lmax = std::min(16.0*1024*1024, 4.0 * bdpBytes);
    if (!std::isfinite(Lmax) || Lmax <= Lmin) {
      Lmin = 64*1024; Lmax = 512*1024; // 兜底
    }
    
    double ratio = std::pow(Lmax / Lmin, 1.0/9.0);
    
    E[0] = Lmin;
    for (int i = 1; i < 10; ++i) {
      E[i] = E[i-1] * ratio;
    }
    return E;
  }

  double
  ConweaveObsManager::ResolveLinkBandwidthBps(uint32_t outIf)
  {
    // [Asym Topology Fix 2026-05-19] Query the actual device DataRate so per-port
    // bwBps is correct under heterogeneous link speeds (e.g. slow spines).
    // Old behavior: hardcoded 1Gbps regardless of outIf -> T_port wrong on slow links.
    if (!m_sw || outIf == 0 || outIf >= m_sw->GetNDevices()) return 1e9;
    Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(m_sw->GetDevice(outIf));
    if (!dev) return 1e9;
    uint64_t bps = dev->GetDataRate().GetBitRate();
    return (bps > 0) ? static_cast<double>(bps) : 1e9;
  }

  void
  ConweaveObsManager::UpdateQueueStats(uint32_t outIf, double nowSec)
  {
    int64_t queueBytes = ReadQueueBytes(outIf);
    if (queueBytes < 0 || !std::isfinite(nowSec)) return;

    EgressPortStats& st = m_portStats[outIf];
    double currentQueue = static_cast<double>(queueBytes);
    if (st.lastQueueSampleSec <= 0.0 || nowSec <= st.lastQueueSampleSec) {
      st.lastQueueSampleSec = nowSec;
      st.lastQueueBytes = currentQueue;
      if (st.avgQueueBytes <= 0.0) st.avgQueueBytes = currentQueue;
      return;
    }

    double dt = nowSec - st.lastQueueSampleSec;
    double tau = std::max(1e-9, m_queueEmaTau);
    double alpha = 1.0 - std::exp(-dt / tau);
    alpha = std::min(1.0, std::max(0.0, alpha));
    double derivative = (currentQueue - st.lastQueueBytes) / dt;
    st.queueDerivEma = (1.0 - alpha) * st.queueDerivEma + alpha * derivative;
    st.avgQueueBytes = (1.0 - alpha) * st.avgQueueBytes + alpha * currentQueue;
    st.queueIntBytes += currentQueue * dt;
    st.lastQueueSampleSec = nowSec;
    st.lastQueueBytes = currentQueue;
  }

  double
  ConweaveObsManager::DecayAndGetDreUtilization(uint32_t outIf, double nowSec)
  {
    EgressPortStats& st = m_portStats[outIf];
    if (st.bwBps <= 0.0) st.bwBps = ResolveLinkBandwidthBps(outIf);
    if (st.last_dre_update_time > 0.0 && nowSec > st.last_dre_update_time) {
      double tau = std::max(1e-9, m_dreTau);
      st.dre_bytes *= std::exp(-(nowSec - st.last_dre_update_time) / tau);
    }
    st.last_dre_update_time = nowSec;
    double denominator = std::max(1e-9, m_dreTau * st.bwBps);
    double utilization = st.dre_bytes * 8.0 / denominator;
    if (!std::isfinite(utilization)) return 0.0;
    return std::min(1.0, std::max(0.0, utilization));
  }

  double
  ConweaveObsManager::QueueReferenceBytes(const EgressPortStats& st) const
  {
    double qRefBdp = m_queueRefAlphaBdp * st.qBdpBytes;
    double qRefBuf = m_queueRefBetaBuf * m_upperBound;
    return std::max({qRefBdp, qRefBuf, m_queueRefMinBytes, 1.0});
  }

  double
  ConweaveObsManager::QueueTrendNormalized(const EgressPortStats& st) const
  {
    // queueDerivEma is bytes/second.  Normalize by the port service rate in
    // bytes/second; the former division by BDP bytes was dimensionally wrong
    // and saturated output 568075741 near one.
    double serviceBytesPerSec = std::max(1.0, st.bwBps / 8.0);
    double trend = st.queueDerivEma / serviceBytesPerSec;
    if (!std::isfinite(trend)) return 0.0;
    return std::min(1.0, std::max(-1.0, trend));
  }

  // double
  // ConweaveObsManager::ComputeReward()
  // {
  //   const double kEps = 1e-9;
  //   auto itps = m_portStats.find(m_lastActionOutIf);
  //   if (itps == m_portStats.end()) {
  //     return 0.0; // 该端口目前还没有统计，直接返回
  //   }
  //   EgressPortStats& st = itps->second;

  //   double now = Simulator::Now().GetSeconds();

  //   auto safe = [](double x) {
  //     return std::isfinite(x) ? x : 0.0;
  //   };
  //   auto clamp01 = [](double x) {
  //     return std::min(1.0, std::max(0.0, x));
  //   };

  //   if (st.winStartSec == 0.0) { // 冷启动
  //       st.winStartSec = now - std::max(1e-6, m_rewardWinSec);
  //   }
  //   double span = now - st.winStartSec;
  //   if (span <= kEps) span = kEps;      // 绝不让窗口为 0

  //   // === 1) 窗口未就绪：直接返回“上一次窗口值”（不更新任何状态） ===
  //   const double W = m_rewardWinSec;
  //   double byteTrigger = (st.ackBytesWin >= m_rewardByteMin);
  //   if (span < W && !byteTrigger) {
  //     auto itp = m_prevRByIf.find(m_lastActionOutIf);
  //     return (itp != m_prevRByIf.end()) ? itp->second : 0.0;
  //   }

  //   // // === 2) 计算本窗口的 r （与你原逻辑一致） ===
  //   // double bwBps = (st.bwBps > kEps) ? st.bwBps : ResolveLinkBandwidthBps(m_lastActionOutIf);

  //   // double T_win = 0.0;
  //   // {
  //   //   double denom = std::max(kEps, bwBps) * std::max(kEps, span);
  //   //   double numBits = 0.0;
  //   //   if (st.ackBytesWin > 0) numBits = 8.0 * st.ackBytesWin;
  //   //   // A1) 去掉 accTxBytes 回退
  //   //   // else if (st.accTxBytes > 0) numBits = 8.0 * st.accTxBytes;
  //   //   T_win = std::min(1.0, numBits / denom);
  //   // }

  //   // double T_ema = 0.0;
  //   // auto itAw2 = m_ackWins.find(m_lastActionOutIf);
  //   // if (itAw2 == m_ackWins.end()) {
  //   //   itAw2 = m_ackWins.insert({m_lastActionOutIf, AckWin()}).first;
  //   // }
  //   // T_ema = std::min(1.0, std::max(0.0, itAw2->second.emaGoodput / std::max(kEps, bwBps)));
  //   // // 调稳：更信任 EMA，减小瞬时窗口的噪声影响
  //   // double T = (T_ema > 0.0) ? (0.3 * T_win + 0.7 * T_ema) : T_win;

  //     // === 2) 计算本窗口的 r ===
  //     double bwBps = (st.bwBps > kEps) ? st.bwBps : ResolveLinkBandwidthBps(m_lastActionOutIf);

  //     // --- NEW: per-flow EMA aggregation for throughput ---
  //     double sumFlowBps = 0.0;
  //     uint32_t activeFlows = 0;
    
  //     for (const auto &kv : m_flow2OutIf) {
  //       const auto &fm = kv.second;
  //       // 只看当前动作端口上的映射
  //       if (fm.outIf != m_lastActionOutIf) continue;
  //       // 丢弃过老的映射，避免僵尸流
  //       if ((now - fm.ts) > m_flowMapTtlSec) continue;
    
  //       auto itFs = m_flowStats.find(kv.first);
  //       if (itFs == m_flowStats.end()) continue;
    
  //       double bps = itFs->second.emaBps;
  //       if (!std::isfinite(bps) || bps <= 0.0) continue;
    
  //       sumFlowBps += bps;
  //       activeFlows += 1;
  //     }
    
  //     // 归一化：仍然用链路带宽，物理意义 = 端口利用率
  //     double T = 0.0;
  //     if (bwBps > kEps && sumFlowBps > 0.0) {
  //       T = sumFlowBps / bwBps;
  //     }
  //     T = clamp01(safe(T));
    
  //     // 占位：为兼容后面 sanitize 逻辑，T_win/T_ema 保留变量但置 0
  //     double T_win = 0.0;
  //     double T_ema = 0.0;
    
    
  //   double Lavg = st.queueIntBytes / std::max(kEps, span);
  //   double qScore = MapQueueToScoreAcc(Lavg, st.qSteps);
  //   st.qSmooth = (1.0 - m_qSmoothLambda) * st.qSmooth + m_qSmoothLambda * qScore;

  //   double R_sw = 0.0;
  //   auto itAw = m_ackWins.find(m_lastActionOutIf);
  //   if (itAw != m_ackWins.end())
  //     R_sw = std::min(1.0, std::max(0.0, itAw->second.emaDup));

  //   double R_flow = -1.0;
  //   {
  //     double sumDup = 0.0; uint32_t cnt = 0;
  //     for (const auto &kv : m_flow2OutIf) {
  //       const auto &fm = kv.second;
  //       if (fm.outIf != m_lastActionOutIf) continue;
  //       if ((now - fm.ts) > m_flowMapTtlSec) continue;
  //       auto itfs = m_flowStats.find(kv.first);
  //       if (itfs == m_flowStats.end()) continue;
  //       sumDup += std::max(0.0, std::min(1.0, itfs->second.emaDup));
  //       cnt += 1;
  //     }
  //     if (cnt > 0) R_flow = sumDup / double(cnt);
  //   }
  //   double R_norm = (R_flow >= 0.0 ? R_flow : R_sw);

  //   double w_dup = std::max(0.0, (m_rewardWDup > 0.0 ? m_rewardWDup : 1.0 - (m_rewardWUtil + m_rewardWQueue)));

  //   // // 端口均衡项：以“本窗已观测端口的 qSmooth 与全端口均值”的差异抑制扎堆11.28注释掉暂时无用，可能起反作用
  //   // double balance = 0.0;
  //   // if (m_rewardWBalance > 0.0) {
  //   //   double sumQ = 0.0; double cntQ = 0.0;
  //   //   for (const auto &kv : m_portStats) {
  //   //     // 只统计最近活跃过的端口（有窗口起点）
  //   //     if (kv.second.winStartSec > 0.0) { sumQ += kv.second.qSmooth; cntQ += 1.0; }
  //   //   }
  //   //   if (cntQ > 0.0) {
  //   //     double meanQ = sumQ / cntQ;
  //   //     // qSmooth 高于均值的端口给一个负向偏置（鼓励走低负载端口）
  //   //     balance = std::max(-1.0, std::min(1.0, meanQ - st.qSmooth));
  //   //   }
  //   // }
    
  //   // B1) 新增 QCN/NACK 惩罚
  //   double R_qcn = 0.0;
  //   if (st.ackWinCtrl + st.nackWinCtrl > 0) {
  //     R_qcn = double(st.nackWinCtrl) / double(st.ackWinCtrl + st.nackWinCtrl);
  //   }

  //   double r_inst = m_rewardWUtil * T 
  //                 + m_rewardWQueue * st.qSmooth 
  //                 + w_dup * (1.0 - R_norm) 
  //                 //+ m_rewardWBalance * balance
  //                 + m_rewardWQcn * (1.0 - R_qcn);

  //   double r_prev = 0.0;
  //   auto itPrev = m_prevRByIf.find(m_lastActionOutIf);
  //   if (itPrev != m_prevRByIf.end()) r_prev = itPrev->second;
  //   double r = 0.7 * r_inst + 0.3 * r_prev;

  //   // sanitize before logging
  //   //auto clamp01 = [](double x){ return std::min(1.0, std::max(0.0, x)); };
  //   //auto safe     = [](double x){ return std::isfinite(x) ? x : 0.0; };
  //   T_win = clamp01(safe(T_win));
  //   T_ema = clamp01(safe(T_ema));
  //   T     = clamp01(safe(T));
  //   st.qSmooth = clamp01(safe(st.qSmooth));
  //   R_sw = clamp01(safe(R_sw));
  //   if (R_flow >= 0.0) R_flow = clamp01(safe(R_flow));
  //   R_norm = clamp01(safe(R_norm));
  //   R_qcn = clamp01(safe(R_qcn));
  //   //r_inst = safe(r_inst);
  //   //r = safe(r);
  //   r_inst = clamp01(safe(r_inst));
  //   r = clamp01(safe(r));

  //   // === 3) 把“窗口内增量”缓存到 pending：delta = r - r_prev ===
  //   double delta = r - r_prev;
  //   m_pendingDeltaByIf[m_lastActionOutIf] += delta;
  //   m_prevRByIf[m_lastActionOutIf] = r;

  //   // ====3.5 记录CSV：每次窗口关闭输出一行 ====
  //   EnsureRewardCsvOpen();
  //   if (m_rewardCsvOpened) {
  //     m_rewardCsv << std::fixed << std::setprecision(9)
  //                 << now << ","
  //                 << m_lastActionOutIf << ","
  //                 << T_port << ","
  //                 << st.qSmooth << ","
  //                 << R_norm << ","
  //                 << R_qcn << ","
  //                 << r_inst << ","
  //                 << r << ","
  //                 << delta << "\n";
  //   }
    
  //   // === 4) 重置窗口累计 ===
  //   st.winStartSec = now;
  //   st.lastQueueSampleSec = now;
  //   st.ackBytesWin = 0;
  //   st.queueIntBytes = 0.0;
  //   st.accTxBytes = 0;
  //   // B1) 重置窗内计数
  //   st.ackWinCtrl = 0;
  //   st.nackWinCtrl = 0;

  //   // === 5) 日志关闭：禁用 [RW] win_close 打印，避免同步 I/O 降速 ===
  //   // （保留空块以便将来恢复）
  //   if (false) { // 启用周期性进度日志（降频）
  //     static double lastPrint = 0.0;
  //     const double printInterval = 0.00050; // 每 0.5ms 打印一次仿真进度
  //     if (now - lastPrint >= printInterval) {
  //       NS_LOG_UNCOND("[PROGRESS] t=" << now);
  //       lastPrint = now;
  //     }
  //   }
    
  //   // H2: 统计窗口关闭次数
  //   static std::unordered_map<uint32_t,uint64_t> s_winCloseCnt;
  //   s_winCloseCnt[m_lastActionOutIf]++;
    
  //   // 在析构函数中添加打印逻辑
  //   struct WinClosesPrinter {
  //     ~WinClosesPrinter() {
  //       for (auto& kv : s_winCloseCnt) {
  //         NS_LOG_UNCOND("[WIN] if=" << kv.first << " closes=" << kv.second);
  //       }
  //       // 追加 COVER/LOOP 汇总（进程级）
  //       const double cover = (S_seen > 0 ? double(S_gated) / double(S_seen) : 0.0);
  //       const double loopr = ((S_back + S_fwd) > 0 ? double(S_back) / double(S_back + S_fwd) : 0.0);
  //       NS_LOG_UNCOND("[COVER] seen=" << S_seen << " gated=" << S_gated
  //                       << " bypass=" << S_bypass << " cover=" << cover);
  //       // Flowlet 粒度覆盖情况
  //       const double hold_ratio = (S_flowletNew > 0 ? double(S_gated) / double(S_flowletNew) : 0.0);
  //       NS_LOG_UNCOND("[FLOWLET] new=" << S_flowletNew
  //                      << " held=" << S_gated
  //                      << " hold_ratio=" << hold_ratio);
  //       NS_LOG_UNCOND("[LOOP] back=" << S_back << " fwd=" << S_fwd
  //                       << " ratio=" << loopr);
  //       NS_LOG_UNCOND("[ADH] chosen=" << S_decChosen
  //                      << " match=" << S_decMatch
  //                      << " mismatch=" << S_decMismatch
  //                      << " ratio=" << (S_decChosen ? double(S_decMatch) / S_decChosen : 0.0)
  //                      << " observed=" << (S_decChosen ? double(S_decMatch + S_decMismatch) / S_decChosen : 0.0));

  //       NS_LOG_UNCOND("[LOOP-LONG] sample=" << S_longLoopSample
  //                      << " hit=" << S_longLoopHit
  //                      << " ratio=" << (S_longLoopSample ? double(S_longLoopHit) / S_longLoopSample : 0.0));

  //       // 字节加权版本，便于对齐不同大小包的影响
  //       const uint64_t obsBytes = S_decMatchBytes + S_decMismatchBytes;
  //       const double adh_bytes_ratio = (obsBytes ? double(S_decMatchBytes) / double(obsBytes) : 0.0);
  //       NS_LOG_UNCOND("[ADH-B] match_bytes=" << S_decMatchBytes
  //                      << " mismatch_bytes=" << S_decMismatchBytes
  //                      << " ratio=" << adh_bytes_ratio);
        
  //       // NEW: 打印按端口的 dupACK/NACK 统计 (按流聚合+三分支)
  //       NS_LOG_UNCOND("--- Per-Port DUP/NACK Stats (Flow-Aggregated, Backstep-as-DUP) ---");
  //       for (const auto& kv : S_dupByIf) {
  //         const auto& st = kv.second;
  //         if (st.total_ctrl == 0) continue;
  //         const double ack_backstep_ratio    = st.ack_total ? double(st.backstep_cnt) / st.ack_total : 0.0;
  //         const double ack_repeat_per_adv    = (st.advance_cnt ? double(st.repeat_total) / st.advance_cnt : 0.0);
  //         const double nack_ratio            = double(st.nack_total) / st.total_ctrl;

  //         NS_LOG_UNCOND("[DUPACK] if=" << kv.first
  //                        << " backstep_ratio=" << ack_backstep_ratio
  //                        << " repeat_per_adv=" << ack_repeat_per_adv
  //                        << " nack_ratio=" << nack_ratio
  //                        << " backsteps=" << st.backstep_cnt
  //                        << " repeats=" << st.repeat_total
  //                        << " advances=" << st.advance_cnt
  //                        << " total_ack=" << st.ack_total
  //                        << " nack=" << st.nack_total
  //                        << " max_streak_repeat=" << st.max_streak);
  //       }
  //     }
  //   };
  //   static WinClosesPrinter printer;

  //   return r;
  // }
  double
  ConweaveObsManager::ComputeReward()
  {
    const double kEps = 1e-9;

    // 1) 没有历史统计时直接返回 0
    auto itps = m_portStats.find(m_lastActionOutIf);
    if (itps == m_portStats.end()) {
      return 0.0;
    }
    EgressPortStats& st = itps->second;

    double now = Simulator::Now().GetSeconds();
    UpdateQueueStats(m_lastActionOutIf, now);

    auto safe = [](double x) {
      return std::isfinite(x) ? x : 0.0;
    };
    auto clamp01 = [](double x) {
      return std::min(1.0, std::max(0.0, x));
    };

    double bwBps = (st.bwBps > kEps) ? st.bwBps : ResolveLinkBandwidthBps(m_lastActionOutIf);
    st.bwBps = bwBps;

    // =========================
    // 2) 队列项：直接用 avgQueueBytes + ACC 阶梯打分
    // =========================

    double L = st.avgQueueBytes;  // OnMacTx 里已经做过 EMA
    // Hybrid scale: max(alpha*BDP, beta*BufferCap), with floor
    double qRef = QueueReferenceBytes(st);
    double qScore = (qRef > 0.0) ? (1.0 / (1.0 + (L / qRef))) : 0.0;
    st.qSmooth = (1.0 - m_qSmoothLambda) * st.qSmooth + m_qSmoothLambda * qScore;
    st.qSmooth = clamp01(safe(st.qSmooth));

    // =========================
    // 3) 乱序 / dup 项：R_norm
    // =========================

    double R_sw = 0.0;
    auto itAw = m_ackWins.find(m_lastActionOutIf);
    if (itAw != m_ackWins.end()) {
      R_sw = clamp01(safe(itAw->second.emaDup));  // 端口级 dup EMA
    }

    double R_flow = -1.0;
    {
      double sumDup = 0.0;
      uint32_t cnt = 0;
      for (const auto &kv : m_flow2OutIf) {
        const auto &fm = kv.second;
        if (fm.outIf != m_lastActionOutIf) continue;
        if ((now - fm.ts) > m_flowMapTtlSec) continue;
        auto itFs = m_flowStats.find(kv.first);
        if (itFs == m_flowStats.end()) continue;
        double dup = clamp01(safe(itFs->second.emaDup));
        sumDup += dup;
        cnt += 1;
      }
      if (cnt > 0) {
        R_flow = sumDup / double(cnt);
      }
    }

    double R_norm = (R_flow >= 0.0 ? R_flow : R_sw);

    // =========================
    // 4) QCN / NACK 惩罚项（当前权重已设为 0，可视为占位）
    // =========================

    double R_qcn = 0.0;
    if (st.ackWinCtrl + st.nackWinCtrl > 0) {
      R_qcn = double(st.nackWinCtrl) / double(st.ackWinCtrl + st.nackWinCtrl);
    }
    R_qcn = clamp01(safe(R_qcn));

    // =========================
    // 5) 拼 reward：T + 队列 + 乱序 + （可选）QCN
    // =========================

    // === Patch Start: Soften the R_norm penalty ===
    // 使用平方惩罚抑制低水平噪声
    double R_norm_sq = R_norm * R_norm; 
    // 确保范围安全
    if (R_norm_sq > 1.0) R_norm_sq = 1.0;

    // Queue growth normalized by service bytes/second.  The encoded trend is
    // shared with the v8 observation and safe action prior: zero means a
    // draining queue, 0.5 is neutral, and one means maximum growth.
    double queueTrendNorm = QueueTrendNormalized(st);
    double queueTrendEncoded = clamp01(0.5 * (queueTrendNorm + 1.0));
    double trendHealth = 1.0 - queueTrendEncoded;
    double instantQueueOccupancy = (qRef > 0.0)
      ? clamp01(st.lastQueueBytes / (st.lastQueueBytes + qRef))
      : 0.0;
    double instantQueueHealth = 1.0 - instantQueueOccupancy;

    // // [Fix v2] T 重设计：用 per-port 真实吞吐率替代失效的 flow EMA
    // // accTxBytes 在 OnMacTx 中累加，winStartSec 在上次 ComputeReward 末尾重置
    // double dt = now - st.winStartSec;
    // double T_port = 0.0;
    // if (dt > 1e-9 && bwBps > 1e-9) {
    //   T_port = (st.accTxBytes * 8.0) / (dt * bwBps);
    // }
    // T_port = clamp01(safe(T_port));
    // [Fix v3] T 重设计 v3: DRE 风格累加器（CONGA-inspired）
      // dre_bytes 在 OnMacTx 中按时间常数 m_dreTau 指数衰减+累加
      // 优势：统计窗口与 ComputeReward 触发频率解耦，反映最近 ~τ 时间的真实端口利用率
      double T_port = DecayAndGetDreUtilization(m_lastActionOutIf, now);

    // Capacity-aware absolute headroom.  The former (1-T_port) term gave the
    // same best-case reward to a 500 Mbps and a 1 Gbps uplink and historical
    // output 450009057 consequently rewarded slow links more highly.  Scaling
    // by the fastest local egress expresses available service rate in common
    // units while preserving congestion sensitivity.
    double maxEgressBandwidthBps = std::max(1.0, bwBps);
    for (uint32_t egressIf : m_egressIfs) {
      maxEgressBandwidthBps = std::max(
        maxEgressBandwidthBps,
        ResolveLinkBandwidthBps(egressIf)
      );
    }
    double capacityNorm = clamp01(safe(bwBps / maxEgressBandwidthBps));
    double headroom = clamp01(capacityNorm * (1.0 - T_port));

    // v8: reward and the non-trainable part of the Q advantage use the same
    // four RL-native signals.  Reordering remains a diagnostic: per-flowlet
    // route ownership already prevents intra-flowlet path changes, and the
    // corrected R_norm is normally zero, so including it only contributed an
    // action-independent constant to every TD target.
    double w_headroom = 0.6;
    double w_queue_ema = 0.2;
    double w_queue_instant = 0.1;
    double w_queue_trend = 0.1;
    double r_inst = w_headroom * headroom
                  + w_queue_ema * st.qSmooth
                  + w_queue_instant * instantQueueHealth
                  + w_queue_trend * trendHealth;
    r_inst = clamp01(safe(r_inst));

    // 6) 再做一层时间平滑：r = alpha * r_inst + (1-alpha) * r_prev
    double r_prev = 0.0;
    auto itPrev = m_prevRByIf.find(m_lastActionOutIf);
    if (itPrev != m_prevRByIf.end()) {
      r_prev = clamp01(safe(itPrev->second));
    }

    double alpha = m_rewardAlpha; // 头文件里默认 0.7
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    double r = alpha * r_inst + (1.0 - alpha) * r_prev;
    r = clamp01(safe(r));

    // 7) 为 delta 模式保留增量缓存；默认 m_useDeltaReward=false 不用
    double delta = r - r_prev;
    m_pendingDeltaByIf[m_lastActionOutIf] += delta;
    m_prevRByIf[m_lastActionOutIf] = r;

    // =========================
    // 8) CSV 记录：每 10 次 ComputeReward 才输出 1 行（采样减 I/O）
    // [2026-05-18] 原来每次写 → I/O 占用大；分析够用 7K 行/sw 即可
    // =========================

    static thread_local uint64_t s_csvCtr = 0;
    s_csvCtr++;
    bool writeCsv = (s_csvCtr % 10 == 0);

    EnsureRewardCsvOpen();
    if (m_rewardCsvOpened && writeCsv) {
      m_rewardCsv << std::fixed << std::setprecision(9)
                  << now << ","
                  << m_lastActionOutIf << ","
                  << T_port << ","
                  << capacityNorm << ","
                  << headroom << ","
                  << st.qSmooth << ","
                  << L << ","
                  << qScore << ","
                  << queueTrendEncoded << ","
                  << R_norm << ","
                  << R_qcn << ","
                  << r_inst << ","
                  << r << ","
                  << delta << ","
                  << st.bwBps << ","
                  << st.qRttSec << ","
                  << st.qBdpBytes << ","
                  << st.qLmin << ","
                  << st.qLmax << "\n";
    }

    // =========================
    // 9) 重置“本次奖励区间内”的一些计数（不再做窗口 gating）
    // =========================

    st.ackWinCtrl    = 0;
    st.nackWinCtrl   = 0;
    st.ackBytesWin   = 0;
    st.queueIntBytes = 0.0;
    st.accTxBytes    = 0;
    st.winStartSec   = now;
    st.lastQueueSampleSec = now;

    // =========================
    // 10) 统计次数（沿用你原来的 WinClosesPrinter）
    // =========================

    static std::unordered_map<uint32_t,uint64_t> s_winCloseCnt;
    s_winCloseCnt[m_lastActionOutIf]++;

    struct WinClosesPrinter {
      ~WinClosesPrinter() {
        for (auto& kv : s_winCloseCnt) {
          NS_LOG_UNCOND("[WIN] if=" << kv.first << " closes=" << kv.second);
        }
        // 追加 COVER/LOOP 汇总（进程级）
        const double cover = (S_seen > 0 ? double(S_gated) / double(S_seen) : 0.0);
        const double loopr = ((S_back + S_fwd) > 0 ? double(S_back) / double(S_back + S_fwd) : 0.0);
        NS_LOG_UNCOND("[COVER] seen=" << S_seen << " gated=" << S_gated
                        << " bypass=" << S_bypass << " cover=" << cover);
        // Flowlet-level route ownership.  fresh_ratio must be one in v8;
        // fallback_packets must be zero.  reused_packets are continuation
        // packets inside a flowlet and do not require another Python call.
        const double hold_ratio = (S_flowletNew > 0 ? double(S_gated) / double(S_flowletNew) : 0.0);
        NS_LOG_UNCOND("[FLOWLET] new=" << S_flowletNew
                      << " held=" << S_gated
                      << " fresh_ratio=" << hold_ratio
                      << " reused_packets=" << S_flowletReusePkts
                      << " dwell_reused_flowlets=" << S_flowletDwellReuse
                      << " fallback_packets=" << S_flowletFallbackPkts);
        NS_LOG_UNCOND("[LOOP] back=" << S_back << " fwd=" << S_fwd
                        << " ratio=" << loopr);
        NS_LOG_UNCOND("[ADH] chosen=" << S_decChosen
                      << " match=" << S_decMatch
                      << " mismatch=" << S_decMismatch
                      << " ratio=" << (S_decChosen ? double(S_decMatch) / S_decChosen : 0.0)
                      << " observed=" << (S_decChosen ? double(S_decMatch + S_decMismatch) / S_decChosen : 0.0));

        NS_LOG_UNCOND("[LOOP-LONG] sample=" << S_longLoopSample
                      << " hit=" << S_longLoopHit
                      << " ratio=" << (S_longLoopSample ? double(S_longLoopHit) / S_longLoopSample : 0.0));

        // 字节加权版本，便于对齐不同大小包的影响
        const uint64_t obsBytes = S_decMatchBytes + S_decMismatchBytes;
        const double adh_bytes_ratio = (obsBytes ? double(S_decMatchBytes) / double(obsBytes) : 0.0);
        NS_LOG_UNCOND("[ADH-B] match_bytes=" << S_decMatchBytes
                      << " mismatch_bytes=" << S_decMismatchBytes
                      << " ratio=" << adh_bytes_ratio);
        
        // NEW: 打印按端口的 dupACK/NACK 统计 (按流聚合+三分支)
        NS_LOG_UNCOND("--- Per-Port DUP/NACK Stats (Flow-Aggregated, Backstep-as-DUP) ---");
        for (const auto& kv : S_dupByIf) {
          const auto& st2 = kv.second;
          if (st2.total_ctrl == 0) continue;
          const double ack_backstep_ratio    = st2.ack_total ? double(st2.backstep_cnt) / st2.ack_total : 0.0;
          const double ack_repeat_per_adv    = (st2.advance_cnt ? double(st2.repeat_total) / st2.advance_cnt : 0.0);
          const double nack_ratio            = double(st2.nack_total) / st2.total_ctrl;

          NS_LOG_UNCOND("[DUPACK] if=" << kv.first
                        << " backstep_ratio=" << ack_backstep_ratio
                        << " repeat_per_adv=" << ack_repeat_per_adv
                        << " nack_ratio=" << nack_ratio
                        << " backsteps=" << st2.backstep_cnt
                        << " repeats=" << st2.repeat_total
                        << " advances=" << st2.advance_cnt
                        << " total_ack=" << st2.ack_total
                        << " nack=" << st2.nack_total
                        << " max_streak_repeat=" << st2.max_streak);
        }
      }
    };
    static WinClosesPrinter printer;

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

    // 正确的 ACK/NACK 判定与确认号
    const bool is_ack  = (ch.l3Prot == 0xFC);
    const bool is_nack = (ch.l3Prot == 0xFD);
    if (!is_ack && !is_nack) return; // 非确认报文，直接忽略
    uint32_t ack_seq   = ch.ack.seq;      // qbbHeader 里的确认号

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

    // ==== NEW: 按端口统计 dupACK/NACK (按流聚合) ====
    double now = Simulator::Now().GetSeconds();
    {
      std::lock_guard<std::mutex> lock(S_dupMutex);
      auto& st = S_dupByIf[outIf];
      st.total_ctrl++;
      if(is_ack) st.ack_total++; else st.nack_total++;

      // TTL/LRU, a simple version to control memory
      const double kFlowSeqTtl = 0.1; // 100ms
      if (st.flow_seq.size() > 10000) {
          for (auto it = st.flow_seq.begin(); it != st.flow_seq.end(); ) {
              if (now - it->second.ts > kFlowSeqTtl) {
                  it = st.flow_seq.erase(it);
              } else {
                  ++it;
              }
          }
      }

      auto &fs = st.flow_seq[flowKey];
      
      if (ack_seq > fs.last_seq) {
        st.advance_cnt++;
        fs.last_seq = ack_seq;
        fs.streak = 0;
      } else if (ack_seq != 0 && ack_seq == fs.last_seq) {
        st.repeat_total++;
        fs.streak++;
        st.max_streak = std::max<uint64_t>(st.max_streak, fs.streak);
      } else if (ack_seq < fs.last_seq) { // ack_seq < fs.last_seq  => 真正的乱序/回退
        st.backstep_cnt++;
        st.backstep_bytes += (uint64_t)(fs.last_seq - ack_seq);
        fs.streak = 0;
        // 不更新 fs.last_seq，保持“最近前进”的基准
      }
      fs.ts = now;
    }

    if (m_portStats.find(outIf) == m_portStats.end()) m_portStats[outIf] = EgressPortStats();
    auto &st = m_portStats[outIf];
    
    if (is_ack)  { st.ackWinCtrl++; }   // 窗内 ACK 计数
    if (is_nack) { st.nackWinCtrl++; }  // 窗内 NACK 计数

    auto &awin = m_ackWins[outIf];
    // Per-flow stats record
    auto &fst = m_flowStats[flowKey];

    // Count ack packet, but compute confirmed payload bytes using ackNo delta
    st.accAckPkts += 1;
    awin.acks += 1;
    fst.ackCount += 1;

    // ACK sequence numbers are scoped to a flow, not to an egress port.  v6
    // compared ack_seq with awin.lastAckNo shared by every flow on the port;
    // output 568075741 consequently reported R_norm around 0.5 even though
    // the forwarding path had no loops or action mismatches.
    uint64_t deltaBytes = 0;
    if (ack_seq != 0) {
      if (ack_seq > fst.lastAckNo) {
        deltaBytes = ack_seq - fst.lastAckNo;
        fst.lastAckNo = ack_seq;
      } else if (ack_seq < fst.lastAckNo) {
        // Only a per-flow backstep is treated as reordering.  Equal repeated
        // acknowledgements are common and remain neutral.
        awin.dupacks += 1;
        fst.dupAckCount += 1;
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

      //per-flow throughput EMA
      double dtFlow = (fst.lastAckTs > 0.0)
                    ? (now - fst.lastAckTs)
                    : m_rewardWinSec;
      if (dtFlow < 5e-6) dtFlow = 5e-6;
      double flow_bps = (double)deltaBytes * 8.0 / dtFlow;

      const double alpha = 0.05;
      fst.emaBps = (1.0 - alpha) * fst.emaBps + alpha * flow_bps;
      fst.lastAckTs = now;

      awin.lastAckTs = now;
      
    // 关闭 ACK 打印，避免大量 I/O
    if (false) {
      static uint64_t ackPrint = 0; (void)ackPrint;
      NS_LOG_UNCOND("[ACK] outIf=" << outIf << " key=" << flowKey << " ack=" << ack_seq
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


  // [Design B 2026-05-14] 计算当前 held packet 在 ECMP 哈希下应该走的 action_id
  // 与 SwitchNode::DoLbFlowECMP 一致的哈希逻辑（同 seed + 同 hash 函数）
  uint32_t
  ConweaveObsManager::ComputeEcmpActionForHeldPacket() const
  {
    if (m_overlayNeighbors.empty() || !m_swHeld) return 0;

    union {
      uint8_t  u8[12];
      uint32_t u32[3];
    } buf;
    buf.u32[0] = m_chHeld.sip;
    buf.u32[1] = m_chHeld.dip;
    if (m_chHeld.l3Prot == 0x06) {
      buf.u32[2] = m_chHeld.tcp.sport | ((uint32_t)m_chHeld.tcp.dport << 16);
    } else if (m_chHeld.l3Prot == 0x11) {
      buf.u32[2] = m_chHeld.udp.sport | ((uint32_t)m_chHeld.udp.dport << 16);
    } else if (m_chHeld.l3Prot == 0xFC || m_chHeld.l3Prot == 0xFD) {
      buf.u32[2] = m_chHeld.ack.sport | ((uint32_t)m_chHeld.ack.dport << 16);
    } else {
      return 0;  // 控制包默认 action 0
    }
    uint32_t hashVal = SwitchNode::EcmpHash(buf.u8, 12, m_swHeld->GetEcmpSeed());
    return hashVal % m_overlayNeighbors.size();
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
      int64_t initialQueueBytes = ReadQueueBytes(outIf);
      if (initialQueueBytes >= 0) {
        st.lastQueueBytes = static_cast<double>(initialQueueBytes);
        st.avgQueueBytes = static_cast<double>(initialQueueBytes);
      }
      const double rtt_sec = std::max(m_rttGuessSec, m_rewardWinSec);
      st.qSteps = BuildQueueStepsBDP(st.bwBps, rtt_sec);
      st.qLmin = st.qSteps.front();
      st.qLmax = st.qSteps.back();
      st.qRttSec = rtt_sec;
      st.qBdpBytes = st.bwBps * rtt_sec / 8.0;
      // Pre-initialize ack window state for this port to avoid later lookups on missing keys
      (void)m_ackWins[outIf];
    }

    // m_flowMapTtlSec = std::max(5e-4, 200.0 * std::max(m_rttGuessSec, m_rewardWinSec)); // Temporarily disable adaptive TTL
    //m_flowMapTtlSec = 5e-4; // Set to a fixed 500us, too small
    m_flowMapTtlSec = 0.02; // Set to a fixed 20ms
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
    // 新维度：[dstOverlay, lastAction] + N_neighbors * kFeatsPerEgress
    uint32_t dim = kObsHeaderDims + (uint32_t)m_overlayNeighbors.size() * kFeatsPerEgress;
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

  float
  ConweaveObsManager::GetReward()
  {
    (void)ComputeReward();

    if (m_useDeltaReward) {
      // 模式1：按 Agent 聚合发放 reward —— 汇总并清零所有端口的 pending 增量
      double reward = 0.0;
      for (auto &kv : m_pendingDeltaByIf) {
        reward += kv.second;
        kv.second = 0.0; // 清零，避免重复累计
      }
      return static_cast<float>(reward);
    } else {
      // 模式2（默认）：返回水平值 r
      // 1) 优先返回“上一拍动作端口”的水平值 r
      if (m_lastActionOutIf != 0) {
        auto it = m_prevRByIf.find(m_lastActionOutIf);
        if (it != m_prevRByIf.end()) {
          double r_level = it->second;
          if (!std::isfinite(r_level)) r_level = 0.0;
          r_level = std::min(1.0, std::max(0.0, r_level));
          return static_cast<float>(r_level);
        }
      }
      // 2) 若不可用（冷启动等），回退到全端口（带宽加权）平均
      double sum = 0.0, wsum = 0.0;
      for (const auto &kv : m_prevRByIf) {
        uint32_t ifx = kv.first;
        double r_prev = kv.second;
        double w = 1.0;
        auto itps = m_portStats.find(ifx);
        if (itps != m_portStats.end() && itps->second.bwBps > 0) w = itps->second.bwBps; // 带宽加权（更稳）
        sum  += w * r_prev;
        wsum += w;
      }
      double r_level = (wsum > 0.0 ? sum / wsum : 0.0);
      if (!std::isfinite(r_level)) r_level = 0.0;
      r_level = std::min(1.0, std::max(0.0, r_level));
      return static_cast<float>(r_level);
    }
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
    bool applied = false;
    if (!m_busy || !m_swHeld) { m_lastActionApplied = false; return false; }
    m_actionSeq += 1;

    /* 3.1 合法性检查 */
    if (m_overlayNeighbors.empty()) { m_busy = false; m_lastActionApplied = false; return false; }
    if (actionId >= m_overlayNeighbors.size()) actionId = 0;

    /* 3.2 选出口 */
    const uint32_t outIf = (actionId < m_egressIfs.size() ? m_egressIfs[actionId] : 0);
    if (outIf == 0) { m_busy = false; m_flowletKeyHeld = 0; m_lastActionApplied = false; return false; }
    applied = true;

    if (m_pktUidHeld != 0) {
      m_decisionByUid[m_pktUidHeld] = outIf;
      // [Hold-only 2026-05-13] 标记 RL 直接释放的首包，T_port 只累加这些包字节
      m_rlHeldUids.insert(m_pktUidHeld);
      m_decChosen++;
      S_decChosen++;
    }

    /* 3.3 直接放行：调用 SwitchNode 的 RlRelease */
    // H4: 判断是否“打回头路”
    if (m_lastInIf != 0 && outIf == m_lastInIf) {
        m_backHop++; S_back++;
    } else {
        m_fwdHop++; S_fwd++;
    }
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
    // 写回 flowlet 粘性与反抖：设置最小驻留时间锁定
    if (m_flowletKeyHeld != 0) {
      auto itf = m_flowlets.find(m_flowletKeyHeld);
      if (itf != m_flowlets.end()) {
        auto &ctx = itf->second;
        ctx.lastActionOutIf = outIf;
        ctx.awaitingAction  = false;
        // === 动作平滑：锁定到 now + minDwellSec，防止频繁切换 ===
        double now = Simulator::Now().GetSeconds();
        ctx.lockedUntilSec  = now + m_minDwellSec;
      }
    }

    // [Design A 2026-05-14] Per-flowKey RL preference 设置
    // 让同一 flowlet 内所有 packet 都跟随首包走 RL 选的 port，消除 intra-flowlet reorder
    // 这是 Bug #12 的正确修复方式：preference 按 flowKey 而非 dstToR 索引，避免跨 flow contamination
    if (m_flowletKeyHeld != 0) {
      m_swHeld->SetRlPreferredForFlow(m_flowletKeyHeld, outIf);
    }

    // 老接口保留为防御性 dead code（dstOverlay 实际永远 = -1，不会执行）
    auto itTor2 = m_indexToSwitch.find(dstOverlay);
    if (itTor2 != m_indexToSwitch.end()) {
      uint32_t dstToR = itTor2->second;
      m_swHeld->SetRlPreferredOutIf(dstToR, outIf);  // 永远不会被调用
    }
    m_swHeld->RlRelease(m_pktHeld, m_chHeld, outIf);

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
    m_flowletKeyHeld = 0;
    ClearCurrentPacketContext();
    ResetPreparedObservation();
    // NS_LOG_UNCOND("[RL] snapshot reset after action");
    m_lastActionApplied = applied;
    return applied;
  }

  void
  ConweaveObsManager::ResetPreparedObservation()
  {
    m_lastObsFeats.clear();
    m_lastDstOverlay = -1;
    m_lastPktId = 0;
    m_lastPktSize = 0;
    m_hasPrepared = false;
    m_flowletKeyHeld = 0;
  }

  //9.11新增
  void
  ConweaveObsManager::OnDelivered(uint64_t uid, double /*nowSec*/)
  {
    // 产生“交付事件脉冲”，并基于首见时间计算 e2e
    m_eventDelivered = true;
    double now = Simulator::Now().GetSeconds();
    double d = 0.0;
    auto it = m_firstSeenSec.find(uid);
    if (it != m_firstSeenSec.end()) {
      d = std::max(0.0, now - it->second);
      m_firstSeenSec.erase(it);
    }
    m_lastDeliveredDelaySec = d;
    m_eventDelaySec = d;
    m_sumE2E += d;
    m_globalDelivered += 1;
    m_seen.erase(uid);
    m_stepDelivered += 1;
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

  ConweaveObsManager::~ConweaveObsManager()
  {
    if (m_rewardCsvOpened) {
      m_rewardCsv.close();
    }
    // 类析构中不再打印，以免与进程级静态打印重复/被吞
  }

  } // namespace ns3
