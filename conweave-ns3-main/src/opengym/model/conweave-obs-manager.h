/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef CONWEAVE_OBS_MANAGER_H
#define CONWEAVE_OBS_MANAGER_H

#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/switch-node.h"
#include "ns3/qbb-net-device.h"
#include "ns3/qbb-channel.h"
#include "ns3/opengym-module.h"   //  新增：空间/容器类型

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <string>
#include <fstream>
#include "ns3/callback.h"  // 新增：Callback
namespace ns3 {
struct CustomHeader;

class ConweaveObsManager : public Object
{
public:
  static TypeId GetTypeId ()
  {
    static TypeId tid = TypeId ("ns3::ConweaveObsManager")
      .SetParent<Object>();
    return tid;
  }

  ConweaveObsManager () {}
  virtual ~ConweaveObsManager ();

  // Each egress is described only by statistics maintained by the RL path:
  // instantaneous queue, queue EMA, DRE utilization, queue trend,
  // capacity-aware headroom, and relative link capacity.
  static constexpr uint32_t kFeatsPerEgress = 6;
  // 观测维度：[dstOverlay, lastAction] + N_neighbors * kFeatsPerEgress
  static constexpr uint32_t kObsHeaderDims = 2;

  // 9.22 一次性初始化overlay邻居序号,overlay->ns3节点id映射,ns3节点id->overlay序号映射
  void Configure(Ptr<SwitchNode> sw,
                 const std::vector<int>& overlayNeighbors,
                 const std::map<int,uint32_t>& indexToSwitch,
                 const std::vector<int>& nodeIdToOverlay);

  // === 空间 ===
  Ptr<OpenGymSpace> GetActionSpace() const;
  Ptr<OpenGymSpace> GetObservationSpace() const;

  void SetNotifyCallback(Callback<void> cb) { m_notifyCb = cb; }

  std::vector<float> GetPreparedObservation() const {
    std::vector<float> out;
    out.reserve(m_lastObsFeats.size());
    for (auto v : m_lastObsFeats) out.push_back(static_cast<float>(v));
    return out;
  }
  bool HasPreparedObservation() const { return m_hasPrepared; }

  uint64_t GetLastPktId()   const { return m_lastPktId; }
  uint32_t GetLastPktSize() const { return m_lastPktSize; }
  int      GetLastDstOverlay() const { return m_lastDstOverlay; }
  bool     LastActionApplied() const { return m_lastActionApplied; }
  uint64_t GetActionSeq() const { return m_actionSeq; }
  
  // 清空本次逐跳缓存（动作后调用，防止 post-action 的下一状态复用旧 pkt）
  void ResetPreparedObservation();

  double GetObsUpperBound() const { return m_upperBound; }
  float GetReward();
  bool  ApplyAction(uint32_t actionId, int dstOverlay);

  void OnDelivered(uint64_t pktUid, double nowSec);
  bool ConsumeDeliveryEvent(double& delayOut);


  // === 工具 ===
  bool Ready() const { return m_ready; }
  uint32_t OverlayNeighborToIfIndex(int overlayNbr) const;

  // ---- 新增：最小契约 ----
  bool     HasCurrentPacketContext() const { return m_hasCtx; }
  void     SetCurrentPacketContext(uint64_t uid, uint32_t size) {
             m_curPktUid = uid; m_curPktSize = size; m_hasCtx = true;
           }
  void     ClearCurrentPacketContext() { m_hasCtx = false; }

  uint64_t GetCurrentPacketUid()  const { return m_curPktUid; }
  uint32_t GetCurrentPacketSize() const { return m_curPktSize; }

  // 取出并清空丢包 UID 列表，拼成 "12;45;"；无则返回空串
  std::string DrainLostPacketsSemicolon();
  // 可选：在任何 drop 回调处调用
  void ReportPacketDrop(uint64_t uid) { m_lostUids.push_back(uid); m_stepDropped += 1; m_globalDropped += 1; }
  // 便于步进观测的丢包计数
  void ReportPacketDropStep(uint64_t uid) { m_lostUids.push_back(uid); m_stepDropped += 1; }

  // 全局累计（给 GetExtraInfo 用）
  uint64_t GetGlobalInjected() const { return m_globalInjected; }
  uint64_t GetGlobalDelivered() const { return m_globalDelivered; }
  uint64_t GetGlobalDropped()  const { return m_globalDropped;  }
  uint64_t GetGlobalBuffered() const { return m_globalBuffered; }
  // 由 SwitchNode 调用：每跳来了包，准备 obs 并发给 Python
  void OnPerHopPacket(Ptr<SwitchNode> sw, Ptr<NetDevice> inDev,
                      Ptr<Packet> p, CustomHeader& ch);

  // 让 Env 能在收包回调里调用（公开接口，供 ConweaveRoutingEnv 使用）
  void ReportAckOnIngress(Ptr<const Packet> p);

  // Called by egress trace thunk when a packet leaves MAC/phy
  void OnMacTx(uint32_t outIf, Ptr<const Packet> p);

// private:
//   uint32_t FindOutIfTo (uint32_t neighborSwitchId) const;

private:
//   Ptr<SwitchNode> m_switch;
//   std::vector<int> m_overlayNeighbors;
//   std::map<int,uint32_t> m_indexToSwitch;
//   std::vector<int> m_nodeIdToOverlay;

//   // 与 m_overlayNeighbors 同索引的一一对应 outDev 索引（0 表示没找到，防御）
//   std::vector<uint32_t> m_neighborIfIdx;
  Ptr<SwitchNode> m_sw;
  std::vector<int> m_overlayNeighbors;
  std::map<int,uint32_t> m_indexToSwitch;
  std::vector<int> m_nodeIdToOverlay;

  // 与 m_overlayNeighbors 对齐的出端口 ifIndex
  std::vector<uint32_t> m_egressIfs;

  // 观测上界（例如 MMU 大小）
  double m_upperBound = 1e9;
  bool m_ready = false;



  //uint32_t ResolveEgressIfToNeighbor(uint32_t neighborNodeId);
  bool               m_hasCtx    = false;
  uint64_t           m_curPktUid = 0;
  uint32_t           m_curPktSize = 0;
  std::vector<uint64_t> m_lostUids;

  //==== 最小事件/累计状态 ====
  bool     m_eventDelivered = false;
  double   m_eventDelaySec = 0.0;

  uint64_t m_globalInjected = 0;
  uint64_t m_globalDelivered = 0;
  uint64_t m_globalDropped  = 0;
  uint64_t m_globalBuffered = 0;

  // 启用首见时间与已见标记，用于 e2e 统计
  std::unordered_map<uint64_t,double> m_firstSeenSec; // pktUid -> first seen time (sec)
  std::unordered_map<uint64_t,bool>   m_seen;         // pktUid -> seen flag
  double m_sumE2E = 0.0;                              // sum of delivered e2e delays (sec)
  double m_lastDeliveredDelaySec = 0.0;               // last delivered packet e2e (sec)
  // 每步（本步上报周期）局部统计，用于 info 中的 dropped/delivered/buffered
  uint64_t m_stepDelivered = 0;
  uint64_t m_stepDropped   = 0;
  uint64_t m_stepBuffered  = 0;
  // === 单并发挂起上下文 ===
  bool             m_busy        = false;
  Ptr<SwitchNode>  m_swHeld;
  Ptr<Packet>      m_pktHeld;
  CustomHeader     m_chHeld;
  uint64_t         m_pktUidHeld = 0;
  // ↓↓↓ 为 PrepareAndSendObservation 缓存 obs / info 的占位成员（编译需要）
  std::vector<uint32_t> m_lastObsFeats;
  int       m_lastDstOverlay = -1;
  uint64_t  m_lastPktId      = 0;
  uint32_t  m_lastPktSize    = 0;
  uint32_t  m_lastSwId       = 0;
  bool      m_hasPrepared    = false;
  std::string m_lastInfo;
  
  Callback<void> m_notifyCb;  // 指向 OpenGymEnv::Notify() 的回调
  uint32_t ResolveEgressIfToNeighbor(uint32_t neighborNodeId) const;
  void PrepareAndSendObservation(int dstOverlay,
                                const std::vector<uint32_t>& feats,
                                uint64_t pktUid,
                                uint32_t swId);
  // 你项目里实际发消息用的那一招（这里先声明，.cc 里给个安全空实现）
  void NotifyGymCurrentState();

  // ==== [ADD] Metrics & Params for ACC reward ====
  struct EgressPortStats {
    uint64_t lastTxBytes = 0;
    uint64_t accTxBytes = 0;
    uint64_t accAckPkts = 0;
    uint64_t accAckBytes = 0;
    double   bwBps = 0.0;
    double   avgQueueBytes = 0.0;
    // 现在的窗口机制已经移除
    double winStartSec = 0.0;
    double lastQueueSampleSec = 0.0;
    uint64_t ackBytesWin = 0;
    double queueIntBytes = 0.0;

    // === [DRE] T_port 重设计：指数衰减字节累加器 0505===
    double dre_bytes = 0.0;            // DRE 累加器（衰减字节）
    double last_dre_update_time = 0.0; // 上次更新时间（秒）
    
    // Ten-step ACC thresholds
    std::array<double,10> qSteps{};
    double qLmin = 0.0;
    double qLmax = 0.0;
    double qRttSec = 0.0;
    double qBdpBytes = 0.0;
    // Smoothed queue score (for low-pass filtering of qScore)
    double qSmooth = 1.0;

    // B1) QCN/NACK窗内计数
    uint64_t ackWinCtrl = 0, nackWinCtrl = 0;

    // === 新增：队列变化率 EMA（拥塞预警） ===
    double lastQueueBytes = 0.0;      // 上一次采样的队列字节数
    double queueDerivEma = 0.0;       // 队列变化率的 EMA (bytes/sec)
  };

  struct AckWin {
    uint64_t bytes = 0;      // accumulated acknowledged bytes in window
    uint32_t acks = 0;       // ack count
    uint32_t dupacks = 0;    // duplicate ack count
    double   emaT = 0.0;     // EMA of normalized throughput [0,1]
    double   emaDup = 0.0;   // EMA of dup fraction [0,1]
    double   lastAckTs = 0.0; // last ack timestamp (Simulator::Now)
  double   emaGoodput = 0.0; // EMA of goodput in bits/sec
  };

  std::unordered_map<uint32_t, AckWin> m_ackWins; // per-outIf ack/window stats
  // Per-flow ACK/NACK stats (for disorder measurement)
  struct FlowStats {
    uint64_t ackCount = 0;
    uint64_t dupAckCount = 0;
    uint64_t nackCount = 0;
    uint64_t lastAckNo = 0;
    double   emaDup = 0.0;   // EMA of dup ratio for the flow
    double   lastTs = 0.0;

    //新增 per-flow throughput estimation
    double emaBps = 0.0;
    double lastAckTs = 0.0;
  };
  std::unordered_map<uint64_t, FlowStats> m_flowStats; // key: flowKey
  struct FlowMapping {
    uint32_t outIf = 0;
    double   ts = 0.0; // seconds since sim start
  };
  // Map flowKey -> mapping (outIf + timestamp)
  std::unordered_map<uint64_t, FlowMapping> m_flow2OutIf;

  std::unordered_map<uint32_t, EgressPortStats> m_portStats; // key: outIf
  // TTL for flow->outIf mappings (seconds); mappings older than this are discarded
  double m_flowMapTtlSec = 0.05;  // 50ms, for flow->outIf mapping
  // 调参：侧重吞吐，降低队列权重，保持乱序权重
  // 修复 Reward 停滞问题：提高吞吐权重以突破物理地板
  double m_rewardWUtil = 0.6;   // 吞吐权重：0.6（主要优化目标）
  double m_rewardWQueue = 0.2;  // 队列权重：0.2（次要目标）
  double m_rewardWDup = 0.2;    // 乱序权重：0.2（保持）
  double m_dreTau = 1e-3;  // [DRE] T_port 衰减时间常数（秒），1ms 0505添加
  // C) 调参：启用一个小的均衡权重
  double m_rewardWBalance = 0.05; // 默认启用轻度均衡
  // B1) 新增：QCN/NACK惩罚项权重，由于和窗口强相关，先关闭，等稳定后可以再打开
  double m_rewardWQcn = 0.0;
  
  // C) 调参：缩短窗口，加快反馈
  double m_rewardWinSec = 40e-6; // 约5个RTT
  double m_rttGuessSec = 8.32e-6; // 8.32us, for BDP estimation
  // Time-based queue EMA avoids giving faster links a different smoothing
  // constant merely because their MacTx callback fires more often.
  double m_queueEmaTau = 100e-6;
  double m_rewardMinSpanSec = 0; //不再使用
  // C) 调参：字节触发阈值与BDP对齐
  uint64_t m_rewardByteMin = 8 * 1024; // 8KB (approx 1 BDP)
  // Queue score reference scale: max(alpha*BDP, beta*BufferCap), with floor
  // [Bug #6 Fix 2026-05-13] qRef 之前是 max(4×BDP, 0.1×totalBuffer=5MB, 32K)=5MB
  // 队列要堆到 5MB 才能让 qSmooth 跌——根本不可能 → qSmooth 永远 ≈ 1（30% 权重沉默）
  // 修复：让 qRef ≈ 16KB（≈3 个 BDP），队列 10K-100K 范围内 qScore 才有明显差距
  double m_queueRefAlphaBdp = 1.0;     // 4.0 → 1.0
  double m_queueRefBetaBuf = 0.001;    // 0.1 → 0.001（基本禁用 totalBuffer-based 项）
  double m_queueRefMinBytes = 16.0 * 1024;  // 32K → 16K
  // Low-pass factor for queue score smoothing
  // 调稳：队列平滑系数减小（慢一点）
  double m_qSmoothLambda = 0.1;
  double m_rewardAlpha = 0.7; // r = 0.7*r_inst + 0.3*r_prev

  double m_winStartSec = 0.0;
  double m_lastRewardTimeSec = 0.0;
  uint32_t m_lastActionOutIf = 0;
  bool m_lastActionApplied = false;
  uint64_t m_actionSeq = 0;
  // 切换：true=返回各端口增量之和（delta），false=返回水平值r（默认）
  bool m_useDeltaReward = false;

  // registration / callbacks
  void RegisterEgressTracing(Ptr<SwitchNode> sw);
  int64_t ReadQueueBytes(uint32_t outIf);
  double MapQueueToScore(double avgQueueBytes, double qMaxBytes);
  double MapQueueToScoreAcc(double L, const std::array<double,10>& E);
  std::array<double,10> BuildQueueStepsBDP(double bwBps, double rttSec);
  double ResolveLinkBandwidthBps(uint32_t outIf);
  void UpdateQueueStats(uint32_t outIf, double nowSec);
  double DecayAndGetDreUtilization(uint32_t outIf, double nowSec);
  double QueueReferenceBytes(const EgressPortStats& st) const;
  double QueueTrendNormalized(const EgressPortStats& st) const;
  double ComputeReward();

private:
  // [Design B 2026-05-14] Warm-start IL helper: 计算 held packet 的 ECMP target action
  uint32_t ComputeEcmpActionForHeldPacket() const;

  // H1: RL决策覆盖率探针 (Ad-hoc)
  std::unordered_map<uint64_t, uint32_t> m_decisionByUid;
  // [Hold-only 2026-05-13] 记录由本 switch 的 RL agent 决策过路径的包 UID。
  // OnMacTx 中只对这些 UID 累加 dre_bytes，让 T_port 反映 Agent 自己的决策后果。
  std::unordered_set<uint64_t> m_rlHeldUids;
  uint64_t m_decChosen{0}, m_decMatch{0}, m_decMismatch{0};

  // H3: 长环路采样探针
  std::unordered_map<uint64_t, std::pair<uint32_t,int>> m_seenSwByUid;
  uint64_t m_longLoopSample{0}, m_longLoopHit{0};

  // 在 private 里加：
  std::unordered_map<uint32_t, double> m_prevRByIf;      // 上一次窗口平滑reward（按端口）
  std::unordered_map<uint32_t, double> m_pendingDeltaByIf; // 尚未分发给“下一步”的增量（按端口）
  // H1: RL决策覆盖率探针
  uint64_t m_seenPkts = 0;
  uint64_t m_gatedPkts = 0;
  uint64_t m_bypassPkts = 0;

  // H4: “打回头路”探针
  uint32_t m_lastInIf = 0; // 记录包的入端口
  uint64_t m_backHop = 0;
  uint64_t m_fwdHop = 0;

  // === Flowlet-level decision support ===
  struct FlowletCtx {
    double   lastSeenSec = 0.0;
    double   startSec    = 0.0;
    uint32_t lastActionOutIf = 0;
    bool     awaitingAction  = false;
    double   lockedUntilSec  = 0.0;
    uint64_t bytesInFlowlet  = 0;
    uint32_t pktsInFlowlet   = 0;
  };
  std::unordered_map<uint64_t, FlowletCtx> m_flowlets; // flowKey -> ctx
  double m_flowletGapSec = 20e-6;    // flowlet的gap阈值，RTT的一半较为合适，config中看maxRTT
  // A flowlet boundary is already the safe switching boundary.  The former
  // 1.5 ms lock was 75x the 20 us gap and suppressed more than 90% of Agent
  // decisions, so v8 deliberately applies no additional dwell lock.
  double m_minDwellSec   = 0.0;
  uint64_t m_flowletKeyHeld = 0;     // flowKey for the currently suspended first packet

  // ==== Reward CSV logging ====
  std::ofstream m_rewardCsv;
  bool m_rewardCsvOpened = false;
  void EnsureRewardCsvOpen();
};

} // namespace ns3
#endif // CONWEAVE_OBS_MANAGER_H
