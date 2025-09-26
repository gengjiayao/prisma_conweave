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
#include "ns3/callback.h"  // 新增：Callback
namespace ns3 {

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
  virtual ~ConweaveObsManager () {}

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
  
  // 清空本次逐跳缓存（动作后调用，防止 post-action 的下一状态复用旧 pkt）
  void ResetPreparedObservation();

  std::vector<float> BuildObservation(int currentDstOverlay) const;
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
  void ReportPacketDrop(uint64_t uid) { m_lostUids.push_back(uid); }

  // 全局累计（给 GetExtraInfo 用）
  uint64_t GetGlobalInjected() const { return m_globalInjected; }
  uint64_t GetGlobalDelivered() const { return m_globalDelivered; }
  uint64_t GetGlobalDropped()  const { return m_globalDropped;  }
  uint64_t GetGlobalBuffered() const { return m_globalBuffered; }
  // 由 SwitchNode 调用：每跳来了包，准备 obs 并发给 Python
  void OnPerHopPacket(Ptr<SwitchNode> sw, Ptr<NetDevice> inDev,
                      Ptr<Packet> p, const CustomHeader& ch);

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

  // 如需 e2e 延迟可在后续扩展（这里先不启用）
  // std::unordered_map<uint64_t,double> m_firstSeenSec;
  // std::unordered_map<uint64_t,bool>   m_seen;
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
    double   bwBps = 0.0;
    double   avgQueueBytes = 0.0;
    uint64_t accAckBytes = 0;
    uint64_t accAckPkts = 0;
    uint64_t accDupAcks = 0;
    uint32_t lastAckSeq = 0; // best-effort seq tracking for dup detection
    
    // Fixed window aggregation (ACC style)
    double winStartSec = 0.0;
    double lastQueueSampleSec = 0.0;
    uint64_t ackBytesWin = 0;
    double queueIntBytes = 0.0;
    
    // Ten-step ACC thresholds
    std::array<double,10> qSteps{};
    // Smoothed queue score (for low-pass filtering of qScore)
    double qSmooth = 1.0;
  };

  struct AckWin {
    uint64_t bytes = 0;      // accumulated acknowledged bytes in window
    uint32_t acks = 0;       // ack count
    uint32_t dupacks = 0;    // duplicate ack count
    uint64_t lastAckNo = 0;  // last seen ack sequence number
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
  double m_rewardWUtil = 0.5;
  double m_rewardWQueue = 0.4;
  double m_rewardWDup = 0.1;
  // W_dup will be 1.0 - w_util - w_queue
  // 调稳：reward 统计窗口 1ms（原 80us）
  double m_rewardWinSec = 0.0005; // 0.5ms，加快窗口闭合
  double m_rttGuessSec = 8.32e-6; // 8.32us, for BDP estimation
  double m_rewardMinSpanSec = 0; //不再使用
  // Minimal ACK bytes to close window early (byte-trigger). 128KB default
  // 调稳：字节触发阈值 256KB（原 64KB），降低高频结算
  uint64_t m_rewardByteMin = 256 * 1024; // 256KB
  // Low-pass factor for queue score smoothing
  // 调稳：队列平滑系数减小（慢一点）
  double m_qSmoothLambda = 0.02;
  double m_rewardAlpha = 0.7; // r = 0.3*r_inst + 0.7*r_prev

  double m_winStartSec = 0.0;
  double m_lastRewardTimeSec = 0.0;
  uint32_t m_lastActionOutIf = 0;

  // registration / callbacks
  void RegisterEgressTracing(Ptr<SwitchNode> sw);
  int64_t ReadQueueBytes(uint32_t outIf);
  double MapQueueToScore(double avgQueueBytes, double qMaxBytes);
  double MapQueueToScoreAcc(double L, const std::array<double,10>& E);
  std::array<double,10> BuildQueueStepsBDP(double bwBps, double rttSec);
  double ResolveLinkBandwidthBps(uint32_t outIf);
  double ComputeReward();

private:
  // 在 private 里加：
  std::unordered_map<uint32_t, double> m_prevRByIf;      // 上一次窗口平滑reward（按端口）
  std::unordered_map<uint32_t, double> m_pendingDeltaByIf; // 尚未分发给“下一步”的增量（按端口）
};

} // namespace ns3
#endif // CONWEAVE_OBS_MANAGER_H