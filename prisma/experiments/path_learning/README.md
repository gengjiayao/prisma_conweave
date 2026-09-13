# Reward-only path learning

The learner starts with eight zero score coefficients and uses network completion
rewards to optimize initial path selection through paired ARS-style perturbations.
Continuation scales exploration from training observations and reduces perturbation
and update sizes over time. The deployed score contains no fixed-rule blend or
expert initialization. Telemetry, observation histories and per-flow path holding
are fixed environment mechanisms. The native DQN-named file format is a container;
the current optimizer does not perform DQN training.

The native implementation now serializes remote measurements into actual
feedback packet bytes. Destination prefixes, output queue bytes and link rates
are advertised by the spine from its own forwarding table and devices. Leaves
update private caches only after decoding a received packet; no packet-UID
lookup table or remote device/routing-table access supplies RL observations.
Sequence numbers reject old/duplicate reports, receiver-local time determines
freshness, and a missing/expired report uses a nominal local-link estimate.
Pressure reports use the same wire mechanism. Oracle diagnostics are disabled
during RL runs and protected by a fail-fast guard.

The [wire feedback evaluation](wire-feedback.json) retains the three selected
actors unchanged and repeats 240 paired runs (five workloads, twelve traffic
seeds, three actors plus CONGA). All 1,776,476 offered flows complete; all sixty
CONGA completion logs match the original baseline exactly. Relative to CONGA,
the repaired implementation gives:

| Metric | Change | 95% paired bootstrap interval |
|---|---:|---:|
| Mean FCT | -21.31% | [-21.62%, -20.90%] |
| P99 FCT | -17.57% | [-18.03%, -17.02%] |
| Batch completion time | -1.57% | [-3.55%, +0.40%] |

Each report is now 194 B, including the simulated link framing, versus 80 B
in the previous abstraction. With the same frozen models, the complete repair
increases mean FCT by 0.69% and P99 by 0.55% relative to the old implementation.
These changes include encoding cost, receive-based freshness and discovery of
destination prefixes; they do not isolate bandwidth cost alone. Batch completion
does not show a statistically established aggregate improvement, and some
workloads regress on that metric. No additional RL training was performed.

The historical local-change input remains zero for compatibility: its training
scale was derived from an all-zero signal. Enabling it requires new normalization
and training. The two new C++ tests in `conweave-ns3-main/tests/` verify wire
fields, malformed messages, delay/loss/expiry behavior, reordering, clock offset,
and decoding a fresh ns-3 packet with a different UID and no metadata tags.

This directory contains only the three selected native policies and a compact
[historical model/results manifest](selected.json), plus the repaired-wire
evaluation summary. The historical manifest includes normalization,
coefficients, model hashes, starting coefficients, test splits, all control
comparisons and confidence intervals. All three training lineages are retained;
none was selected using test performance. Intermediate checkpoints, raw traces,
logs, figures and detailed audits belong in separate experiment storage.

The historical continuation test covers 128 hosts, eight leaf and eight spine
switches with asymmetric 1/0.5 Gbps uplinks, five workload families and twelve
held-out traffic seeds. Validation selected additional round 18 after 24 rounds
of continuation from the three actors trained for twelve rounds from zero.
All 780 test runs completed every flow.

The following learning-attribution results were obtained with the former
feedback abstraction. Use the repaired-wire results above for the current
implementation's CONGA comparison.

| Comparison | Mean FCT change (95% CI) | P99 change (95% CI) |
|---|---:|---:|
| Current RL vs paired starting RL | -1.41% [-1.68%, -1.10%] | -2.00% [-2.67%, -1.39%] |
| Current RL vs shuffled-reward continuation | -1.41% [-1.68%, -1.10%] | -2.00% [-2.67%, -1.39%] |
| Current RL vs original-parameter CONGA | -21.85% [-22.15%, -21.51%] | -18.02% [-18.62%, -17.41%] |

The CONGA comparison includes the entire routing scheme. The matched starting
and shuffled controls isolate the incremental reward-learning improvement.
CONGA retains its original DRE/flowlet/aging parameters (50/100/500 microseconds)
and feedback, without added ACK/NACK feedback. The learner meets the predefined
noninferiority margins against three fixed-rule controls, but does not establish
superiority over all of them. Posthoc small-flow P99 increases by 2.13% relative
to the starting actors. These results do not establish improvements on 1024 hosts
or hardware; inference compute has zero modeled delay. Confidence intervals
resample traffic seeds and paired model lineages, not training datasets or the
whole validation-selection process.

## Running a new experiment

Use Python 3 with NumPy and a compatible native simulator built from this
repository. The native build needs the existing ns-3 dependencies, including
ZeroMQ, matching protobuf headers/libraries and `protoc`. Run the ns-3 configure
step before building: it generates `messages.pb.cc` and `messages.pb.h` from
`messages.proto`; these generated C++ bindings and build products are excluded
from Git.

From the repository root, set `NS3_BINARY` to the absolute path of the built
`scratch/network-load-balance` executable and `RUN_ROOT` to an output directory
outside the checkout. Provide any additional shared-library locations through
`LD_LIBRARY_PATH`; the runtime also adds the executable's parent build directory.

```bash
python3 prisma/scripts/learn_path_from_scratch.py \
  --experiment-dir "$RUN_ROOT/from_scratch" --ns3-binary "$NS3_BINARY" --stage all
python3 prisma/scripts/continue_path_learning.py \
  --experiment-dir "$RUN_ROOT/continued" --ns3-binary "$NS3_BINARY" \
  --previous-experiment "$RUN_ROOT/from_scratch" --stage all
```

Use `--stage calibrate` for the initial calibration check or `--stage setup` for
the continuation setup check. Continuation requires the completed first run,
including its training monitor trajectories for exploration scaling. The shared
runtime and base configuration are tracked in the repository; no copied helper
scripts or pre-existing simulation output directories are required.

For selected-policy evaluation, use `path_rollout.configure(output, binary)`,
`make_trial(..., model=policy_path)` and `execute_jobs(jobs)` from
`prisma/source/path_rollout.py`. `make_trial(..., conga=True)` applies the original
CONGA baseline settings. Keep the selected text models unchanged: normalization
is already folded into their native weights. Starting actors can be reconstructed
with `path_policy_search.export(theta, normalization, output)` using the manifest.

The full original archives remain separate and sealed. Their identifying hashes
are in the manifest. The published runtime consolidates their common helper
functions; historical source hashes describe the archived implementation, not
this packaging refactor. Reproducing the complete historical audit requires the
original raw archives and frozen binary dependencies. A new build and campaign
must record their own provenance rather than overwrite a completed experiment.
