/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "conweave-routing-env.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

// ======================= IMPORTANT =======================
// You MUST include the header for the specific Node type you are using.
// From your switch-node.cc, it seems you are using a custom SwitchNode.
// We include it here.
#include "ns3/switch-node.h" 
// =========================================================
#include "ns3/qbb-net-device.h"
#include "ns3/qbb-channel.h"
#include "conweave-obs-manager.h"
#include "src/network/utils/custom-header.h"

#include <cstdint>
#include <sstream>
#include <iomanip>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ConweaveRoutingEnv");

NS_OBJECT_ENSURE_REGISTERED (ConweaveRoutingEnv);

// --- Boilerplate and Constructor ---
// Reusing the structure from PacketRoutingEnv but heavily simplified.

TypeId
ConweaveRoutingEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ConweaveRoutingEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<ConweaveRoutingEnv> ()
  ;
  return tid;
}

ConweaveRoutingEnv::ConweaveRoutingEnv ()
{
  NS_LOG_FUNCTION (this);
}

ConweaveRoutingEnv::ConweaveRoutingEnv (Ptr<Node> node) :
  m_node(node)
{
  NS_LOG_FUNCTION (this << node);
  NS_ASSERT (m_node != 0); // Ensure we are always bound to a valid node
}

ConweaveRoutingEnv::~ConweaveRoutingEnv ()
{
  NS_LOG_FUNCTION (this);
}

void
ConweaveRoutingEnv::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_obsMgr = 0;
  m_node = 0;
  OpenGymEnv::DoDispose();
}

// --- OpenAI Gym API Implementation (with Placeholders) ---

Ptr<OpenGymSpace>
ConweaveRoutingEnv::GetActionSpace (void)
{
  // // For validation, we define a simple action space of 4 discrete actions.
  // uint32_t num_actions = 4; 
  // return CreateObject<OpenGymDiscreteSpace> (num_actions);
  // uint32_t n = static_cast<uint32_t>(m_overlayNeighbors.size());
  // if (n == 0) n = 1; // 防御，避免 0
  // return CreateObject<OpenGymDiscreteSpace> (n);
  if (!m_obsMgr) return CreateObject<OpenGymDiscreteSpace>(1);
  return m_obsMgr->GetActionSpace();
}
 
Ptr<OpenGymSpace>
ConweaveRoutingEnv::GetObservationSpace (void)
{
  // // For validation, we define a simple observation space of 2 floating point numbers.
  // uint32_t obs_dims = 2;
  // return CreateObject<OpenGymBoxSpace> (0.0, 1.0, std::vector<uint32_t>{obs_dims}, TypeNameGet<float>());
  // uint32_t dim = 1 + static_cast<uint32_t>(m_overlayNeighbors.size());
  // std::vector<uint32_t> shape { dim };

  // double high = 1e9; // 兜底
  // if (m_obsMgr) {
  //   high = std::max(1.0, m_obsMgr->GetObsUpperBound());
  // }
  // return CreateObject<OpenGymBoxSpace>(0.0, high, shape, TypeNameGet<float>());
  if (!m_obsMgr) return CreateObject<OpenGymBoxSpace>(0.0, 1.0,
                     std::vector<uint32_t>{1}, TypeNameGet<float>());
  return m_obsMgr->GetObservationSpace();
}

Ptr<OpenGymDataContainer>
ConweaveRoutingEnv::GetObservation (void)
{
  // // We send a constant observation vector to Python.
  // // The values are based on the node ID to verify each agent gets unique data.
  // auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{2});
  // box->AddValue( 1.0f ); // First value is the node ID
  // box->AddValue( 2.0f ); // Second value is current time
  
  // return box;
  if (!m_obsMgr) {
    auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{1});
    box->AddValue(0.0f);
    return box;
  }
  if (!m_obsMgr->HasPreparedObservation()) {
    // Emit a placeholder observation (same dimensionality as the observation space)
    auto space = DynamicCast<OpenGymBoxSpace>(m_obsMgr->GetObservationSpace());
    uint32_t dim = 1;
    if (space && !space->GetShape().empty()) dim = space->GetShape().at(0);
    auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{dim});
    for (uint32_t i = 0; i < dim; ++i) box->AddValue(0.0f);
    // NS_LOG_INFO("[RL] placeholder obs dim=" << dim);
    return box;
  }

  std::vector<float> obs = m_obsMgr->GetPreparedObservation();
  auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{(uint32_t)obs.size()});
  for (float v : obs) box->AddValue(v);
  return box;
}

float
ConweaveRoutingEnv::GetReward (void)
{
  // Return a constant reward.
  return m_obsMgr ? m_obsMgr->GetReward() : 0.0f;
}

bool
ConweaveRoutingEnv::GetGameOver (void)
{
  // The simulation will not end based on this for now.
  return false;
}

bool
ConweaveRoutingEnv::ExecuteActions (Ptr<OpenGymDataContainer> action)
{
  // // We receive the action and print it to the console to verify communication.
  // auto discrete_action = DynamicCast<OpenGymDiscreteContainer>(action);
  // uint32_t actionId = discrete_action->GetValue();
  
  // // This log is the ultimate proof that the round-trip communication works.
  // NS_LOG_UNCOND("VALIDATION SUCCESS: Node " << m_node->GetId() 
  //             << " received action " << actionId << " from Python agent.");
              
  // return true;
  // auto discrete_action = DynamicCast<OpenGymDiscreteContainer>(action);
  // uint32_t actionId = discrete_action ? discrete_action->GetValue() : 0;

  // NS_LOG_UNCOND("VALIDATION SUCCESS: Node " 
  //   << (m_node ? m_node->GetId() : 999999)
  //   << " received action " << actionId << " from Python agent.");

  // // 这里先只打日志，后面再把 action 应用到 switch-node 的选择上
  // return true;
  auto a = DynamicCast<OpenGymDiscreteContainer>(action);
  uint32_t actionId = a ? a->GetValue() : 0;
  NS_LOG_INFO("[Action] node=" << (m_node ? m_node->GetId() : 999999) << " a=" << actionId);

  bool ok = true;
  if (m_obsMgr) {
    ok = m_obsMgr->ApplyAction(actionId, m_currentDstOverlay);
    // Ensure the prepared one-shot snapshot is cleared immediately so the
    // subsequent GetObservation/GetExtraInfo produce placeholder state.
    m_obsMgr->ResetPreparedObservation();
    // NS_LOG_INFO("[RL] snapshot reset after action (ExecuteActions)");
  }
  return ok;
}

std::string
ConweaveRoutingEnv::GetExtraInfo() //9.5版本
{
  // 若当前没有逐跳预置的观测缓存，返回占位（不触发 KeyError）
  if (!m_obsMgr || !m_obsMgr->HasPreparedObservation()) {
    // 返回完整键列表的占位，避免 Python 端解析异常
    std::ostringstream oss; oss.setf(std::ios::fixed); oss<<std::setprecision(6);
    const double nowSec = Simulator::Now().GetSeconds();
    oss << "delay_time=" << 0.0 << ","
        << "pkt_size=" << 0.0 << ","
        << "curr_time=" << nowSec << ","
        << "pkt_id=" << 0   << ","
        << "pkt_type=" << 2 << ","  // 占位按控制包处理，避免 Python 当作数据包
        << "avg_e2e_delay=" << 0.0 << ","
        << "cost=" << 0.0 << ","
        << "global_avg_e2e_delay=" << 0.0 << ","
        << "global_cost=" << 0.0 << ","
        << "dropped=" << 0.0 << ","
        << "delivered=" << 0.0 << ","
        << "injected=" << 0.0 << ","
        << "buffered=" << 0.0 << ","
        << "global_dropped=" << 0.0 << ","
        << "global_delivered=" << 0.0 << ","
        << "global_injected=" << 0.0 << ","
        << "global_buffered=" << 0.0 << ","
        << "signaling_overhead=" << 0.0 << ","
        << "lost_packets_id=" << ";";
    return oss.str();
  }

  //读取消费一次交付事件
  double eventDelay = 0.0;
  const bool deliveredPulse = m_obsMgr->ConsumeDeliveryEvent(eventDelay);
  const double injectedPulse = deliveredPulse ? 1u : 0u; // 无损场景交付即视作曾注入
  const uint32_t droppedPulse = 0u; // 目前不考虑丢包事件

  // 使用 ObsManager 的逐跳缓存，保证 obs/info 对齐
  const uint32_t pktSize   = m_obsMgr->GetLastPktSize();
  const double   nowSec    = Simulator::Now().GetSeconds();    // 仿真时钟（秒）
  const uint64_t pktUid    = m_obsMgr->GetLastPktId();
  const int      pktType   = 0;                                // 仅数据事件
  const double   zero      = 0.0;

  std::string lostList = m_obsMgr->DrainLostPacketsSemicolon();
  

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(6);

  // ---- 索引 0..4：基础字段（顺序不可变）----
  //oss << "delay_time=" << delayTime << ","
  oss << "delay_time=" << eventDelay << ","
      << "pkt_size=" << pktSize << ","
      << "curr_time=" << nowSec << ","
      << "pkt_id=" << pktUid << ","
      << "pkt_type=" << pktType << ",";

  // ---- 索引 5..17：统计（先全部置 0，但必须占位）----
  // oss << "avg_e2e_delay=" << zero << ","
  //     << "cost=" << zero << ","
  //     << "global_avg_e2e_delay=" << zero << ","
  //     << "global_cost=" << zero << ","
  //     << "dropped=" << zero << ","
  //     << "delivered=" << zero << ","
  //     << "injected=" << zero << ","
  //     << "buffered=" << zero << ","
  //     << "global_dropped=" << zero << ","
  //     << "global_delivered=" << zero << ","
  //     << "global_injected=" << zero << ","
  //     << "global_buffered=" << zero << ","
  //     << "signaling_overhead=" << zero << ",";
    oss << "avg_e2e_delay=" << zero << ","
      << "cost=" << zero << ","
      << "global_avg_e2e_delay=" << zero << ","
      << "global_cost=" << zero << ","
      << "dropped=" << static_cast<double>(droppedPulse) << ","
      << "delivered=" << static_cast<double>(deliveredPulse) << ","
      << "injected=" << static_cast<double>(injectedPulse) << ","
      << "buffered=" << zero << ","
      << "global_dropped=" << static_cast<double>(m_obsMgr->GetGlobalDropped()) << ","
      << "global_delivered=" << static_cast<double>(m_obsMgr->GetGlobalDelivered()) << ","
      << "global_injected=" << static_cast<double>(m_obsMgr->GetGlobalInjected()) << ","
      << "global_buffered=" << static_cast<double>(m_obsMgr->GetGlobalBuffered()) << ","
      << "signaling_overhead=" << zero << ",";

  // ---- 索引 18：丢包 UID 列表（分号分隔 & 以分号结尾；空表用 ";"）----
  // Python 端用 tokens[18].split('=')[-1].split(';')[:-1] 解析
  oss << "lost_packets_id=" << lostList;

  return oss.str();
}


// --- Communication Engine ---

void
ConweaveRoutingEnv::Initialize(void)
{
  NS_LOG_FUNCTION (this);
  // 构建并配置 ObsManager（一次性）
  Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(m_node);
  NS_ASSERT_MSG(sw != nullptr, "ConweaveRoutingEnv must bind to a SwitchNode");

  m_obsMgr = CreateObject<ConweaveObsManager>();
  m_obsMgr->Configure(sw, m_overlayNeighbors, m_indexToSwitch, m_nodeIdToOverlay);
  m_obsMgr->SetNotifyCallback(MakeCallback(&OpenGymEnv::Notify, this));
  sw->AttachRlMgr(m_obsMgr);   // 关键：把 RL 管理器绑到交换机

  // 绑定 MacRx → Env::NotifyPktRcv（关键线头：收包事件进入统计/交付脉冲）
  {
    Ptr<ConweaveRoutingEnv> self = this;
    sw->TraceConnectWithoutContext(
      "MacRx",
      MakeBoundCallback(&ConweaveRoutingEnv::NotifyPktRcv, self)
    );
  }
  // This first call safely triggers the ZMQ handshake.
  // It reuses the exact same robust mechanism as PacketRoutingEnv.
  // 把 Notify 回调交给 ObsManager，由它决定何时触发
  //m_obsMgr->RegisterPacketHooks(sw, MakeCallback(&OpenGymEnv::Notify, this));

  // 选一种稳妥的周期节拍（示例：rp_timer 或已有 switch_mon_interval）
  // m_obsMgr->SchedulePeriodicNotify(MicroSeconds(rp_timer), MakeCallback(&OpenGymEnv::Notify, this));

  // 首次触发，完成 ZMQ 握手
  Notify();
}

  // Start the periodic communication loop.
  //ScheduleNextCommunication();


void
ConweaveRoutingEnv::OnDataPacketForDecision(uint64_t /*pktUid*/, uint32_t /*pktSizeBytes*/, int /*dstOverlay*/)
{
  // Intentionally no-op: do not Notify here.
  // Packet delivery events are used for statistics only.
  // Per-hop decisions (Notify) are triggered by ConweaveObsManager::OnPerHopPacket().
}

//9.9收包时发送观测逻辑
void
ConweaveRoutingEnv:: NotifyPktRcv(Ptr<ConweaveRoutingEnv> entity,
                                     Ptr<NetDevice> /*netDev*/,
                                     Ptr<const Packet> packet) 
{
  if (!entity || !entity->m_obsMgr) return;

  Ptr<Packet> p = packet->Copy();
  CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);
  ch.getInt = 1;
  p->PeekHeader(ch);

  // 0) 统一过滤控制帧：QCN(0xFF)/PFC(0xFE) 直接跳过
  if (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE) return;

  // 1) ACK/NACK(0xFC/0xFD)：只做奖励统计，不触发交付
  if (ch.l3Prot == 0xFC || ch.l3Prot == 0xFD) {
    entity->m_obsMgr->ReportAckOnIngress(p);
    return;
  }

  // 2) 防呆：仅统计 TCP(0x06)/UDP(0x11) 数据包为 delivered
  if (!(ch.l3Prot == 0x06 || ch.l3Prot == 0x11)) return;

  // 3) 数据包：仅当到达“目的 ToR”时记一次交付事件
  auto itSwitch = Settings::hostIp2SwitchId.find(ch.dip);
  if (itSwitch == Settings::hostIp2SwitchId.end()) return;
  uint32_t rxToSwitchId = itSwitch->second;

  if (!entity->m_node || entity->m_node->GetId() != rxToSwitchId) return; // 非目的 ToR

  // 到达目的 ToR：记交付脉冲（GetExtraInfo 会把 delivered/injected=1 上报给 Python）
  entity->m_obsMgr->OnDelivered(p->GetUid(), Simulator::Now().GetSeconds());

  // 低频进度打印（可选）
  static uint64_t s_delivered = 0;
  if ((++s_delivered % 500) == 0) {
    NS_LOG_UNCOND("[DEL] delivered=" << s_delivered
                   << " t=" << Simulator::Now().GetSeconds());
  }
}

void
ConweaveRoutingEnv::OnPacketDropped(uint64_t pktUid)
{
  if (m_obsMgr) m_obsMgr->ReportPacketDrop(pktUid);
}

// void
// ConweaveRoutingEnv::ScheduleNextCommunication()
// {
//   // To keep the connection alive and test data exchange, we will
//   // periodically trigger a state notification to Python.
//   // We stop after a certain time to allow the simulation to end.
//   if (Simulator::Now().GetSeconds() < 20.0) { // Run for 20 seconds
//       Notify();
//       // Schedule the next communication in 1 second.
//       Simulator::Schedule(Seconds(1.0), &ConweaveRoutingEnv::ScheduleNextCommunication, this);
//   }
// }

 

  
} // namespace ns3