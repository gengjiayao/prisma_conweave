# Project Context Sync Prompt

Please read the following project status and retain it as context for our subsequent discussions.

---

## Project Overview

We are building an RL-based adaptive routing system for datacenter networks by integrating two open-source projects:
- **PRISMA**: A Multi-Agent RL framework for packet routing (Python, TensorFlow, DQN)
- **ConWeave**: An NS3-based datacenter network simulator (C++, NS3)

**Architecture**: 8 Leaf (ToR) switches + 8 Spine switches in a 2-tier Clos topology with 128 hosts. Each Leaf has an independent DQN Agent. Agents make routing decisions at **flowlet granularity** (not per-packet): when a new flowlet arrives at a Leaf, the Agent chooses which of 8 uplink ports (to Spines) to forward it through. Spine switches do not need Agents (they have only one deterministic downlink to the destination Leaf).

**Communication**: Python RL Agents and NS3 simulator communicate via ZMQ + Protobuf, step-by-step synchronous interaction.

**Observation space**: 2 header dims (dstOverlay, lastAction) + 5 CONGA features per port x 8 ports = 42 dims. The 5 per-port features are: queue occupancy (normalized), local congestion (ce_local), remote min congestion (ce_remote_min), feedback freshness (age), path coverage (cov). These come from CONGA routing's GetOneHopMetrics().

**Action space**: Discrete(8) — choose one of 8 uplink ports.

**Reward**: `r = 0.4 * qSmooth + 0.6 * (1 - R_norm^2)` where qSmooth is per-port queue health score and R_norm is per-port reorder rate (dupAck EMA). A throughput term T was originally weighted at 0.6 but has been set to 0 due to a design flaw (see below).

---

## Key Findings (chronological)

### Finding 1: Throughput metric T structurally broken

T (port utilization) was computed from flow-to-port mapping entries with 20ms TTL. But RL decides at flowlet granularity (gap > 20us, inter-flowlet intervals >> 20ms). By the time ComputeReward() runs, the mapping has expired -> T is always 0. This 0.6-weighted component contributed nothing, locking reward at floor value 0.34.

**Fix**: Set T weight to 0, redistribute to qSmooth(0.4) + R_norm(0.6). Reward rose from 0.34 to 0.82.

### Finding 2: Observation features were mostly zeros

4 of 5 per-port CONGA features were hardcoded to 0.0f in BuildObservation(). Also lastAction was missing from obs header. Agent could only see raw queue bytes.

**Fix**: Wired real CONGA metrics from GetOneHopMetrics() and added lastAction.

### Finding 3: Low signal-to-noise ratio in reward

Even with correct reward and observations, after 1d9h training (netload=50), Agent did not learn. Per-port reward difference: 0.007; per-port reward std: 0.045. SNR = 0.15. All 8 Spine ports have nearly identical congestion/reorder levels in symmetric topology under moderate load. DQN cannot extract signal from such noise.

### Finding 4: Bandwidth was 1Gbps not 100Gbps

Topology file had been modified from 100Gbps to 1Gbps for faster simulation. This made training 100x slower and invalidated all timing-dependent parameters (BDP, RTT). Has been corrected back to 100Gbps.

---

## Current Status

- Reward signal pathway verified (reward rose from 0.34 to 0.82 after T ablation)
- CONGA observation features wired in
- 100Gbps bandwidth restored
- **Agent has NOT yet learned an effective policy** — need higher load (netload >= 70) to create port-level congestion asymmetry
- **No ECMP baseline FCT data yet** — this is the next immediate task
- T metric needs redesign using per-port real throughput (accTxBytes delta / time / bandwidth)

## Competitive Positioning

- vs ECMP: **Winnable** at netload >= 70 where hash collisions create load imbalance
- vs CONGA: **Difficult but possible** in asymmetric failure or periodic traffic scenarios
- vs ConWeave: **Not realistic in 2 months** — ConWeave is current SOTA with years of engineering

## Timeline

- Deadline: June 2026
- Critical milestone: Agent learns effective policy under high load (by mid-April)
- Risk: If DQN cannot learn from weak signals, may need to switch to PPO/SAC
