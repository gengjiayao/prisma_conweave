# SONiC 交换机硬件信息存档

> 采集日期：2026-04-09
> 用途：算法闭环后的部署参考，当前仅存档

---

## 基本信息

| 项目 | 值 |
|------|-----|
| 平台 | x86_64-centec_v782_32d_d-r0 |
| HwSKU | V782-32d-d |
| ASIC | Centec（盛科） |
| ASIC 数量 | 1 |
| SONiC 版本 | SONiC.202211-centec.0-d94fde9b8 |
| 内核 | 5.10.0-18-2-amd64 |
| Debian | 11.11 |
| 序列号 | E985GD247003 |
| 硬件版本 | 2.1 |

## 端口信息

- 32 口 400G（Ethernet0 ~ Ethernet124，步长 4）
- 每口 4 lane（如 Ethernet0: lane 136,137,138,139）
- FEC: RS
- MTU: 9100
- 当前状态：所有端口 Oper down / Admin up（未接线）
- Vlan 模式：trunk

## 已确认可用的能力

### 队列计数器

```
show queue counters Ethernet0
```

输出 8 个队列（ALL0-ALL7），每个有：
- Counter/pkts, Counter/bytes（发送统计）
- Drop/pkts, Drop/bytes（丢包统计）

当前全 0（端口未接线）。**这是累积计数器，不是瞬时队列深度**——部署时需确认是否有 SAI_QUEUE_STAT_CURR_OCCUPANCY_BYTES 实时深度计数器。

### 路由表

FRR (v8.2.2) 可通过 vtysh 访问，当前只有直连路由 10.18.0.0/16。ECMP 路由和权重修改需要配置后验证。

### Docker 环境

已有多个 SONiC 容器运行（orchagent、syncd-centec、fpm-frr 等），可部署自定义容器。

## 待确认事项（部署阶段再查）

| 事项 | 验证方法 | 重要性 |
|------|---------|--------|
| Centec SAI 是否支持 ECMP member weight 修改 | 配置 ECMP 路由后尝试改 weight | 高——方案 A 的前提 |
| 队列瞬时深度是否可读 | redis-cli 查 COUNTERS_DB 中 CURR_OCCUPANCY 类 key | 中——影响观测精度 |
| ECN 标记计数器是否可用 | redis-cli 查 SAI_PORT_STAT_ECN_MARKED_PACKETS | 中——额外观测维度 |
| ECMP 权重修改的生效延迟 | 发流量 + 改权重 + 抓端口计数器 | 中——影响决策频率 |
| 盛科 SAI 支持的完整计数器列表 | redis-cli -n 2 keys "COUNTERS:*" 枚举 | 低——进去自己查 |

## 与仿真环境的对照

| 维度 | NS-3 仿真 | 该交换机 | 备注 |
|------|----------|---------|------|
| 端口带宽 | 100Gbps (仿真) | 400Gbps (实际) | 4 倍差异，BDP 需重算 |
| 端口数 | 8 uplink (leaf) | 32 口 (角色待定) | 取决于组网方式 |
| 队列深度 | QbbNetDevice API | ASIC 计数器 | 实际精度可能更高 |
| 远端拥塞 | CONGA piggyback | 不可用（需 INT） | 方案 A 的主要损失 |
| ECN | 未建模 | 可能有硬件支持 | 部署的额外观测 |

## 注意事项

- 盛科的 SAI 实现可能不如博通完整，某些高级特性（如 per-member weight、INT）可能不支持
- SONiC 202211 是较旧版本，某些新 SAI API 可能需要升级
- 当前无 IP 路由配置，部署前需先搭建 Leaf-Spine 组网环境
