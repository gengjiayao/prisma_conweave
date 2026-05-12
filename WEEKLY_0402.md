# 周报 2026.3.27 - 2026.4.2

## 本周工作

- [x] 周三参加中期验收答辩（15 分钟问答），准备了状态空间设计、Reward 函数、Huber Loss 等技术细节的 QA 材料
- [x] 在消融实验（T 权重置零，reward 从 0.34 提升至 0.82）的基础上，进行了长训练验证（simul_time=10，约 4 小时），观察到 reward 稳定在 0.82 附近但未出现持续上升。经分析发现 BuildObservation 中 5 维 CONGA 端口特征有 4 维为硬编码 0，Agent 观测信息不足，已修复接入 CongaRouting::GetOneHopMetrics() 提供的真实拥塞指标

## 下周计划

- [ ] 在补全观测特征的基础上，继续进行高负载训练实验，验证 Agent 是否能学到有效的端口选择策略
- [ ] 采集 ECMP baseline 的 FCT 数据，为后续 RL vs ECMP 性能对比做准备
