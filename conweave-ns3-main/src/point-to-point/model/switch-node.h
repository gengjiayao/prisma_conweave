#ifndef SWITCH_NODE_H
#define SWITCH_NODE_H

#include <ns3/node.h>
#include <ns3/event-id.h>

#include <unordered_map>
#include <unordered_set>

#include "qbb-net-device.h"
#include "switch-mmu.h"

#include "ns3/traced-callback.h"
#include "ns3/callback.h"
#include "ns3/packet.h"

namespace ns3 {
class ConweaveObsManager;
class Packet;

class SwitchNode : public Node {
    static const unsigned qCnt = 8;    // Number of queues/priorities used
    static const unsigned pCnt = 128;  // port 0 is not used so + 1	// Number of ports used
    uint32_t m_ecmpSeed;
    std::unordered_map<uint32_t, std::vector<int> >
        m_rtTable;  // map from ip address (u32) to possible ECMP port (index of dev)

    // monitor uplinks
    uint64_t m_txBytes[pCnt];  // counter of tx bytes, for HPCC


   

    

   protected:
    bool m_ecnEnabled;
    uint32_t m_ccMode;
    uint32_t m_ackHighPrio;  // set high priority for ACK/NACK

   private:
   struct RlHeldPkt
    {
        Ptr<Packet>  p;
        CustomHeader ch;
        uint32_t     inDev = 0;   // 入口端口
        bool         valid = false;
    };

    bool                    m_rlBusy = false;   // 是否正等待 Python 动作
    RlHeldPkt               m_rlHeld;           // 被挂起的包
    Ptr<Object> m_rlMgr;            // 外部注入的观测管理器
    EventId     m_rlTimeoutEv;      // RL 挂起兜底超时事件
    double      m_rlTimeoutUs = 100.0; // 兜底超时：100 微秒

    /* 尝试挂起；返回 true 表示已挂起（后续由 RL 决策） */
    bool RlMaybeHold(Ptr<NetDevice> inDev, Ptr<Packet> p, CustomHeader &ch);

    int GetOutDev(Ptr<Packet>, CustomHeader &ch);
    void SendToDev(Ptr<Packet> p, CustomHeader &ch);
    void SendToDevContinue(Ptr<Packet> p, CustomHeader &ch);
    static uint32_t EcmpHash(const uint8_t *key, size_t len, uint32_t seed);
    void CheckAndSendPfc(uint32_t inDev, uint32_t qIndex);
    void CheckAndSendResume(uint32_t inDev, uint32_t qIndex);

    /* Sending packet to Egress port */
    void DoSwitchSend(Ptr<Packet> p, CustomHeader &ch, uint32_t outDev, uint32_t qIndex);

    /*----- Load balancer -----*/
    // Flow ECMP (lb_mode = 0)
    uint32_t DoLbFlowECMP(Ptr<const Packet> p, const CustomHeader &ch,
                          const std::vector<int> &nexthops);
    // DRILL (lb_mode = 2)
    uint32_t DoLbDrill(Ptr<const Packet> p, const CustomHeader &ch,
                       const std::vector<int> &nexthops);     // choose egress port
    uint32_t m_drill_candidate;                               // always 2 (power of two)
    std::map<uint32_t, uint32_t> m_previousBestInterfaceMap;  // <dip, previousBestInterface>
    uint32_t CalculateInterfaceLoad(uint32_t interface);      // Get the load of a interface
    // Conga (lb_mode = 3)
    uint32_t DoLbConga(Ptr<Packet> p, CustomHeader &ch, const std::vector<int> &nexthops);
    // Conga (lb_mode = 6)
    uint32_t DoLbLetflow(Ptr<Packet> p, CustomHeader &ch, const std::vector<int> &nexthops);
    // ConWeave (lb_mode = 9)
    uint32_t DoLbConWeave(Ptr<const Packet> p, const CustomHeader &ch,
                           const std::vector<int> &nexthops);  // dummy

    // RL override (lb_mode = 7)
    uint32_t DoLbRl(Ptr<Packet> p, CustomHeader &ch, const std::vector<int> &nexthops);

    // 7 == RL 覆写模式；0 表示“跟随全局 Settings::lb_mode”（默认）
    uint32_t m_lbMode = 0;

    // dstToR -> outIf（只消费一次）
    std::unordered_map<uint32_t, uint32_t> m_rlPreferOnce;


   public:
   /* 由 ConweaveRoutingEnv 在动作到达时调用，放行被挂起的包 */
    void RlRelease(uint32_t outIf);
    void RlTimeoutFallback();          // 超时回退（ECMP 放行）

    /* 仿真初始化阶段把 ObsManager 绑进来 */
    void AttachRlMgr(Ptr<Object> mgr) { m_rlMgr = mgr; }

    /* ----------------  你原来的其余 public 声明  ---------------- */
    
    // Ptr<BroadcomNode> m_broadcom;
    Ptr<SwitchMmu> m_mmu;
    bool m_isToR;                                 // true if ToR switch
    std::unordered_set<uint32_t> m_isToR_hostIP;  // host's IP connected to this ToR

    static TypeId GetTypeId(void);
    TracedCallback<Ptr<NetDevice>,Ptr<const Packet>>
        m_traceMacRx;  // receive trace

    SwitchNode();
    void SetEcmpSeed(uint32_t seed);
    void AddTableEntry(Ipv4Address &dstAddr, uint32_t intf_idx);
    void ClearTable();
    bool SwitchReceiveFromDevice(Ptr<NetDevice> device, Ptr<Packet> packet, CustomHeader &ch);
    void SwitchNotifyDequeue(uint32_t ifIndex, uint32_t qIndex, Ptr<Packet> p);
    uint64_t GetTxBytesOutDev(uint32_t outdev);

    //9.11 RL动作实际生效将 dstToR -> outIf 写入一次性表；下一次对该 dstToR 的转发将优先使用并消费掉
    void SetRlPreferredOutIf(uint32_t dstTorId, uint32_t outIf);
    // 如果存在一次性首选端口，返回 true 并通过 outIfOut 给出，同时从表中移除（只消费一次）
    bool TryConsumeRlPreferredOutIf(uint32_t dstTorId, uint32_t &outIfOut);
};

} /* namespace ns3 */

#endif /* SWITCH_NODE_H */
