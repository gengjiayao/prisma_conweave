# PRISMA x ConWeave: Multi-Agent RL for Datacenter Adaptive Routing

## 1. Research Motivation

Traditional datacenter routing (ECMP, WCMP) is oblivious to real-time network state, leading to load imbalance and high tail latency. ConWeave and similar flowlet-based schemes use hardcoded heuristics for path switching but lack global optimization capability. RL can learn from online network feedback to adaptively select optimal uplink paths for each flowlet.

## 2. Core Idea

Deploy PRISMA (a Multi-Agent RL routing framework) onto ConWeave (an NS3 datacenter simulator):

- **Each Leaf (ToR) switch** = an independent DQN Agent
- **Observation space**: destination overlay node ID + last action + per-uplink queue/congestion features
- **Action space**: which uplink port to forward the current flowlet to
- **Reward**: weighted combination of port utilization + queue health + reorder rate
- **Communication**: Python (PRISMA) <-> C++ (NS3/ConWeave) via ZMQ + Protobuf step-by-step

## 3. System Architecture

```
+--------------------------------------------------+
|                  Python (PRISMA)                  |
|                                                   |
|  main.py                                          |
|    +-- parse_arguments()     parse args + topo    |
|    +-- Agent.init_static_vars()  init globals     |
|    +-- for each episode:                          |
|          +-- run_ns3() ----------------------+    |
|          |                                   |    |
|          +-- Forwarder(leaf_i)  x N_leaf     |    |
|          |   +-- Ns3Env(port=base+i)  ZMQ    |    |
|          |   +-- DQN_AGENT (q_net, target)   |    |
|          |   +-- run():                      |    |
|          |       while connected:            |    |
|          |         obs,r,done,info = step()   |    |
|          |         action = agent.step(obs)   |    |
|          |         env.step(action)  ---------+    |
|          |                                   |    |
|          +-- Trainer(leaf_i)  x N_leaf       |    |
|              +-- run(): sample->train->sync  |    |
|                                              |    |
+------------- ZMQ (per-agent port) -----------+    |
|                                              |    |
|                C++ (ConWeave NS3)            <--+ |
|                                                   |
|  ConweaveObsManager (per-switch)                  |
|    +-- OnPerHopPacket()    build obs + hold pkt   |
|    +-- ApplyAction()       apply RL routing       |
|    +-- ComputeReward()     multi-dim reward       |
|    |   +-- T: port utilization (per-flow EMA)     |
|    |   +-- qSmooth: queue health (ACC ladder)     |
|    |   +-- R_norm: reorder rate (dup ACK EMA)     |
|    |   +-- R_qcn: QCN penalty (weight=0 now)      |
|    +-- GetReward()         return smoothed r_level |
|                                                   |
|  SwitchNode                                       |
|    +-- packet forwarding + flowlet + VOQ reorder  |
+---------------------------------------------------+
```

## 4. Topology Adaptation

ConWeave uses Leaf-Spine topology (e.g., 128 hosts + 8 leaf + 8 spine), while PRISMA natively supports arbitrary overlay graphs. Adaptation:

1. `convert_conweave_topo.py`: extract leaf nodes from ConWeave topo, generate overlay adjacency matrix + overlay_index <-> switch_id mapping
2. Only create Agents for Leaf nodes (Spine does not participate in RL)
3. Action space = number of uplink ports per Leaf (= number of Spines)

## 5. Current Status

- Basic communication pipeline works: Python and NS3 interact step-by-step via ZMQ
- Model can train: Huber Loss drops to 0
- **Core problem**: Reward signal is decoupled from Agent actions, see KNOWN_ISSUES.md
- **Next steps**: Fix reward attribution + adjust reward formula + align decision granularity

## 6. Key Files

| File | Purpose |
|------|---------|
| `prisma/main.py` | Main entry, orchestrates episode loop |
| `prisma/source/forwarder.py` | Forwarder agent, main loop interacting with NS3 |
| `prisma/source/trainer.py` | Async DQN training |
| `prisma/source/agent.py` | Agent base class, global static vars |
| `prisma/source/learner.py` | DQN network definition + train/step |
| `prisma/source/run_ns3.py` | Launch ConWeave NS3 simulation |
| `prisma/ns3_model/ns3env.py` | ZMQ communication bridge (Python side) |
| `conweave-ns3-main/src/opengym/model/conweave-obs-manager.cc` | Obs/Reward/Action core (C++ side) |
| `prisma/scripts/convert_conweave_topo.py` | Topology conversion script |

## 7. How to Run

Inside Docker:
```bash
cd /workspace/PRISMA_conweave/prisma
python3 main.py \
  --train 1 \
  --lb rl \
  --topo leaf_spine_128_100G_OS2 \
  --simul_time 5 \
  --netload 20 \
  --basePort 6555 \
  ...
```

`main.py` launches `conweave-ns3-main/run.py` via `run_ns3()`, then both sides interact through per-agent ZMQ ports for training.
