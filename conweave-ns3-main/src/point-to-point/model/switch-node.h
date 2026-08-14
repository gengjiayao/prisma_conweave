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

    Ptr<Object> m_rlMgr;            // 外部注入的观测管理器
    EventId     m_rlTimeoutEv;      // RL 挂起兜底超时事件
    double      m_rlTimeoutUs = 100.0; // 兜底超时：100 微秒

    /* 【改造】现在只负责将包“喂”给RL管理器，不自己决定是否挂起 */
    void RlHandover(Ptr<NetDevice> inDev, Ptr<Packet> p, CustomHeader &ch);

    int GetOutDev(Ptr<Packet>, CustomHeader &ch);
    void SendToDev(Ptr<Packet> p, CustomHeader &ch);
    
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

    // dstToR -> outIf（只消费一次）— 旧接口，保留但 DoLbRl 不再调用
    std::unordered_map<uint32_t, uint32_t> m_rlPreferOnce;

    // [Design A 2026-05-14] Per-flowKey 路由表
    // Key = ConWeaveRouting::GetFlowKey(sip, dip, sport, dport) 的 5-tuple 哈希
    // Value = (RL 选的 outIf, 最后访问时间秒数)
    // 设计目标：同一 flowlet 内所有包都跟随首包走 RL 的 port，消除 intra-flowlet reorder
    struct FlowletRoutePref {
      uint32_t outIf;
      double   last_access_sec;
    };
    std::unordered_map<uint64_t, FlowletRoutePref> m_rlPrefByFlow;
    // Defensive cache only: the ObsManager owns normal v8 continuation
    // routing directly.  Match its 20 us flowlet boundary so this cache can
    // never preserve an action into a later flowlet.
    static constexpr double kFlowletPrefTTL = 20e-6;
    static constexpr size_t kFlowletPrefSizeCap = 50000; // 触发清理的上限


   public:
   void SendToDevContinue(Ptr<Packet> p, CustomHeader &ch);
   /* 由 ConweaveObsManager 在动作到达时调用，放行被挂起的包 */
    void RlRelease(Ptr<Packet> p, CustomHeader& ch, uint32_t outIf);
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

    // [Design A 2026-05-14] Per-flowKey 路由表接口
    // SetRlPreferredForFlow: RL 对某 flow 做出决策后调用（在 ApplyAction 里）
    void SetRlPreferredForFlow(uint64_t flowKey, uint32_t outIf);
    // LookupRlPreferredForFlow: DoLbRl 中按 flowKey 查找 RL 选的 port
    // 含 refresh-on-access 语义：访问时刷新 last_access_sec，长 flowlet 不会被截断
    bool LookupRlPreferredForFlow(uint64_t flowKey, uint32_t &outIfOut);
    // CleanupExpiredFlowPref: 清理过期 entry（在 Set 时若 size 超上限触发）
    void CleanupExpiredFlowPref();

    // [Design B 2026-05-14] Warm-start IL: 暴露 ECMP seed + EcmpHash 让 obs-manager 计算同步的 ECMP 期望动作
    uint32_t GetEcmpSeed() const { return m_ecmpSeed; }
    static uint32_t EcmpHash(const uint8_t *key, size_t len, uint32_t seed);

    // -- 用于跨模块更新 RL 统计的静态函数 --
    static void IncrementRlHeld();
    static void IncrementRlBusySkip();
};

} /* namespace ns3 */

#endif /* SWITCH_NODE_H */
