/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * MINIMALIST Conweave-specific RL Environment for connection validation.
 * Based on the structure of PRISMA's PacketRoutingEnv.
 */

#ifndef CONWEAVE_ROUTING_ENV_H
#define CONWEAVE_ROUTING_ENV_H

#include "ns3/opengym-module.h" // Essential for OpenGymEnv, spaces, containers
#include "ns3/core-module.h"     // Essential for Ptr, Object, TypeId, Simulator
#include "ns3/network-module.h"  // Essential for Node

namespace ns3 {

/**
 * @brief A MINIMALIST OpenAI Gym Environment for Conweave.
 *
 * Its sole purpose is to establish a stable ZMQ connection and verify
 * that data can be exchanged with the Python agent. It uses constant
 * values for observations and actions.
 */
class ConweaveObsManager;

class ConweaveRoutingEnv : public OpenGymEnv
{
public:
  // Standard NS3 object boilerplate, 100% reused from PacketRoutingEnv
  static TypeId GetTypeId (void);

  //9.9新增
  static void NotifyPktRcv(Ptr<ConweaveRoutingEnv> entity,
                         Ptr<NetDevice>          inDev,
                         Ptr<const Packet>       pkt);
  

  // Simplified constructor. The only essential part is binding to a Node.
  
  ConweaveRoutingEnv (); // Default constructor for NS3
  ConweaveRoutingEnv (Ptr<Node> node);
  virtual ~ConweaveRoutingEnv ();

  // --- OpenAI Gym API Implementation ---
  // The structure is 100% reused from PacketRoutingEnv.
  // The internal logic is replaced with placeholders.
  virtual Ptr<OpenGymSpace> GetActionSpace (void) override;
  virtual Ptr<OpenGymSpace> GetObservationSpace (void) override;
  virtual Ptr<OpenGymDataContainer> GetObservation (void) override;
  virtual float GetReward (void) override;
  virtual bool GetGameOver (void) override;
  virtual bool ExecuteActions (Ptr<OpenGymDataContainer> action) override;

  std::string GetExtraInfo() override;

  // --- Communication Engine ---
  // This is the core mechanism for solving the timing and logic issues,
  // 100% reused from PacketRoutingEnv's design pattern.
  void Initialize(void);

  // main 里已经在用的 3 个 setter（最小适配）
  void SetOverlayNeighbors (const std::vector<int>& v) { m_overlayNeighbors = v; }
  void SetIndexToSwitchMap (const std::map<int,uint32_t>& mp) { m_indexToSwitch = mp; }
  void SetNodeIdToOverlay  (const std::vector<int>& v) { m_nodeIdToOverlay = v; }

  // 给 switch-node 用的两个入口（二选一）
  void SetCurrentDstOverlay (int dstOverlay) { m_currentDstOverlay = dstOverlay; }
  void SetDstByNodeId (uint32_t dstNodeId)
  {
    if (dstNodeId < m_nodeIdToOverlay.size()) m_currentDstOverlay = m_nodeIdToOverlay[dstNodeId];
  }
  void OnDataPacketForDecision(uint64_t pktUid, uint32_t pktSizeBytes, int dstOverlay);
  void OnPacketDropped(uint64_t pktUid);
  Ptr<ConweaveObsManager> GetObsManager() const { return m_obsMgr; }

protected:
  // Standard NS3 object boilerplate
  virtual void DoDispose (void) override;

private:
  // The single most important member variable, binding logic to physics.
  Ptr<Node> m_node;

  // ---- 映射 & 目的缓存（从 main 注入）----
  std::vector<int> m_overlayNeighbors;         // 本 agent 的 overlay 邻居（overlay 下标）
  std::map<int,uint32_t> m_indexToSwitch;      // overlay 下标 -> 真实 switch node id
  std::vector<int> m_nodeIdToOverlay;          // 真实 node id -> overlay 下标
  int m_currentDstOverlay = -1;                 // 观测第0项

  // ---- 观测委派器 ----
  Ptr<ConweaveObsManager> m_obsMgr;
  // A simple timer-based trigger for communication, replacing PRISMA's
  // complex packet-based event system for now.
  //void ScheduleNextCommunication();
};

} // namespace ns3

#endif /* CONWEAVE_ROUTING_ENV_H */