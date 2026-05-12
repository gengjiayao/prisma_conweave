# PRISMA x ConWeave Known Issues

> Last updated: 2026-03-25
> Status: RED unfixed | YELLOW in progress | GREEN fixed

---

## P0: Reward signal decoupled from Agent actions (root cause)

### P0-1 GREEN T (port utilization) dominated reward but was invisible to RL

- **Symptom**: Reward converges to ~0.34 regardless of Agent actions
- **Data**: T mean=0.031, 95.4% samples T~0; weight was 0.6
- **Root cause**: T = RL-routed bps / port bandwidth(100Gbps). RL only controls 2.9% of packets (hold_ratio=0.028), so T is nearly always 0
- **Fix applied**: Changed weights from (T=0.6, qSmooth=0.2, R_norm=0.2) to (T=0.1, qSmooth=0.4, R_norm=0.5)
- **File**: `conweave-ns3-main/src/opengym/model/conweave-obs-manager.h` lines 255-257
- **Expected impact**: RL-controllable portion goes from 0.34/0.36=94% to ~100% of reward variance

### P0-2 GREEN Transition (s, a, r, s') bootstrap across different destinations

- **Symptom**: Huber Loss drops to 0 but policy does not improve
- **Root cause**: When consecutive steps observe different flowlets (different dst), s' has no relationship to s. Q(s',a') estimate is meaningless noise, corrupting the TD target
- **Fix applied**: Set `done_for_buffer = True` when `obs[0] != prev_obs[0]` (different destination), cutting off bootstrap. Agent learns immediate reward for cross-dst transitions
- **File**: `prisma/source/forwarder.py` lines 1339-1341

### P0-3 RED RL coverage too low (2.9%)

- **Data**: seen=2,919,698, held=84,244, ctrl_skip=6,219,304, dsttor_skip=3,299,606
- **Status**: Not yet addressed. May improve naturally with better exploration strategy

---

## P0-4 GREEN Exploration-congestion positive feedback loop (NEW)

- **Symptom**: reward starts at ~0.56 (warmup) but drops to ~0.34 within 0.5s of simulation and never recovers
- **Data breakdown by phase**:
  - warmup (t<2.1s): avg_T=0.34, avg_R_norm=0.48, avg_r_inst=0.56
  - early (2.1-2.5s): avg_T=0.05, avg_R_norm=0.53, avg_r_inst=0.37
  - late (>3.0s): avg_T=0.006, avg_R_norm=0.54, avg_r_inst=0.34
- **Root cause**: Uniform random exploration picks a different uplink port 87.5% of the time (7/8 ports) causing massive packet reordering. Reordering triggers congestion signals (NACK, retransmission). Congestion reduces throughput -> lower T -> lower reward -> no good experience to learn from -> policy stays random -> more reordering. A vicious cycle.
- **Fix applied**: Sticky exploration - when exploring randomly, 50% probability to repeat last action (same port), reducing unnecessary port switches
- **File**: `prisma/source/learner.py` step() method
- **Expected impact**: Halve the reorder rate during exploration. Allow network to stay healthier so reward signal is more informative

---

## P1: RL vs ConWeave mechanism conflicts

### P1-1 YELLOW RL port switching causes reordering (R_norm~0.53)

- **Partially addressed by P0-4 sticky exploration**
- **Further work**: Consider aligning RL decisions with ConWeave flowlet gap detection

### P1-2 RED Time scale mismatch

- **PRISMA original**: per-packet decision
- **ConWeave environment**: reward computed per 40us window (5 RTT) with EMA smoothing
- **Status**: Deferred. Flowlet-level gating in C++ already limits RL to flowlet boundaries

---

## P2: Engineering defects

### P2-1 RED Agent static variables lack thread synchronization

- Not yet addressed. CPython GIL provides basic safety but logical races remain

### P2-2 RED Ns3Env.reset() dead code

- Not yet addressed (reverted to avoid ZMQ init complications)

### P2-3 RED Trainer.run() has no exit condition

- Not yet addressed. Relies on daemon=True

---

## Fix Round 1 (2026-03-25) - Training run: fix_v1_reward_sticky

**Changes applied:**
1. P0-1: Reward weights (0.1/0.4/0.5) in conweave-obs-manager.h
2. P0-2: done_for_buffer same_dst check in forwarder.py
3. P0-4: Sticky exploration (50% repeat) in learner.py

**Parameters:**
```
--train=1 --netload=20 --buffer=50 --simul_time=2
--exploration_schedule_timesteps=150000 --exploration_final_eps=0.05
--replay_buffer_max_size=100000 --pfc=0 --irn=1 --cc=dctcp --gamma=0.9
```

**Baseline comparison (previous run 618051410):**
- avg_r_inst across all leaves: ~0.36
- T contribution: ~0.018 (wasted)
- R_norm: ~0.53

**Results: (pending - run in progress)**

---

## Fix Round 2 (2026-03-25) - Run 940694653

**Single change**: Override reward weights in  ComputeReward():
- Old:  -> floor at 0.34
- New:  -> floor at 0.82

**Why T=0 weight**: T (port utilization) is structurally broken in current architecture.
flow EMA entries expire after 20ms (m_flowMapTtlSec), and RL only controls flowlet
first-packets. Between flowlets, no active flow maps to the RL-chosen port -> T=0 always.
T accounted for 60% of reward weight but contributed 0 information.

**Results**:
| Phase    | Baseline r_inst | Fix r_inst | R_norm |
|----------|----------------|------------|--------|
| warmup   | 0.56           | 0.85       | 0.48   |
| early    | 0.37           | 0.83       | 0.53   |
| mid      | 0.35           | 0.82       | 0.54   |
| late     | 0.34           | 0.82       | 0.54   |

Reward no longer collapses. R_norm unchanged (expected - only fixed signal, not policy).
Next step: verify RL can now learn to reduce R_norm over training episodes.
