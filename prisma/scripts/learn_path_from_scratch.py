#!/usr/bin/env python3
"""Frozen native 128-host experiment: learn a path score from zero coefficients.

Run with an isolated --experiment-dir and a compatible --ns3-binary.
The rollout runtime and base configuration are included in this repository.
The experiment never changes default routing or CONGA parameters.
"""
import argparse
import os
from pathlib import Path
import sys

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment-dir", type=Path, required=True)
parser.add_argument("--ns3-binary", type=Path, help="Native executable; defaults to experiment-dir/binaries/frozen/scratch/network-load-balance")
parser.add_argument("--stage", choices=["all", "calibrate", "train", "validate", "test", "ablate"], default="all")
args = parser.parse_args()
TASK = args.experiment_dir.resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import path_policy_search as policy
import path_rollout as env
import json
import time
from collections import defaultdict
import numpy as np

env.configure(TASK, args.ns3_binary)
BINARY = env.BINARY
FAMILIES = ["uniform", "storage", "sweep", "mixed"]
TEST_FAMILIES = FAMILIES + ["mixed_fast"]
FITS = [9101, 9102, 9103]
ARMS = ["real", "shuffled"]
METRICS = ["all_mean_us", "all_p99_us", "all_cvar99_us", "makespan_us", "all_max_us"]


read = env.read
write = env.write
make = env.make_trial
execute = env.execute_jobs


def init_protocol():
    if (TASK / "protocol.json").exists():
        return
    write("protocol.json", dict(
        created_epoch=time.time(), hosts=128, switches=16,
        question="Can reward-only policy search learn useful initial-path scoring without an expert score or coefficient warm start?",
        stage_scope="Initial selection only. Flow holding, telemetry, and 200-us history construction remain fixed environment mechanisms.",
        observations=list(policy.FEATURE_NAMES), feature_indices=list(policy.FEATURES),
        architecture="Eight independent signed linear coefficients, all zero initially. Native 24x32x2 ReLU container is an exact linear export. No fixed local score, expert actions, teacher rewards, positive sign constraints, or expert blending.",
        scale="RMS of each measurement on four neutral training-only calibration trajectories; frozen before optimization. No FCT or expert decisions used to compute scales.",
        optimizer="ARS-style paired Gaussian policy perturbations; top four of eight directions, return standard-deviation normalization, step .12, update norm cap .30, symmetric unit-ball projection of center weights. No neural representation training.",
        reference="https://proceedings.neurips.cc/paper_files/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf",
        fits=FITS, arms=ARMS, rounds=12, directions=8, top=4, step=.12, update_cap=.30,
        sigma=[.35]*4 + [.15]*4 + [.05]*4, calibration_traffic=81000,
        training_traffic=list(range(81001, 81013)), training_families=FAMILIES,
        reward="-100 * mean_over_four_families(log(mean_FCT_us/1000) + .25*log(P99_FCT_us/1000)). A family with unfinished flows at the unchanged horizon receives -10000; never average only completed flows. Native errors abort. Expert performance is never a target.",
        failure_return=-10000.,
        shuffled="Permute sixteen own measured returns among perturbation/sign labels each round and seed, then use the identical update. Same directions, initialization, rollouts, checkpoint selection budget.",
        training_native_per_arm=2304, center_monitor_native_per_arm=144,
        validation_traffic=[82001,82002,82003], validation_rounds=[-1,0,3,7,11],
        selection="One common round per arm across all three optimization seeds and four families, maximizing the same reward. Never select seeds or mix in fixed rules.",
        validation_native_per_arm=180, fixed_grid=dict(remote=[.125,.25,.5], work=[.05,.1,.2], trend=[-.5,0,.5]),
        fixed_budget="27 configurations x 12 traffic conditions = 324 validation runs; larger than either learned arm. Original fixed rule also tested separately.",
        test_traffic=list(range(83001,83009)), test_families=TEST_FAMILIES,
        test_freeze="Create test traces only after selection file is written; retain all seeds and outcomes.",
        ablations="On all primary test traffic, zero only the selected learned local-delay, remote-delay, or port-work coefficient separately, without retraining. Exploratory; not used for selection.",
        claim="Learning evidence requires mean-FCT 95% upper delta <0 vs neutral and matched shuffled. Recovering fixed-rule performance additionally requires mean upper delta <+1% and P99 upper delta <+2% vs BOTH original and validation-selected fixed rules. Exceeding rules requires mean upper delta <0 vs both. Report all magnitudes.",
        bootstrap="10000 paired traffic-seed block resamples across five fixed families, with matched resampling of three real/shuffled fit IDs. Does not resample training/validation datasets.",
        transport="Same previous 128-host asymmetric 1/.5-Gbps topology, DCTCP, IRN on, PFC off, RTO10/32ms, 50MiB buffers, 50ms injection plus 1s drain; all flows must finish.",
        feedback="24 feature slots, eight policy coefficients; local_change remains zero. Explicit wire reports carry destination prefixes, queues and link rates (194 B/200us in this 128-host topology). Freshness uses receiver-local time. Pressure off; report serialization/queues charged, inference computation delay zero.",
        conga="Original repository CONGA parameters DRE50/flowlet100/aging500us, no added ACK/NACK feedback. No tuning.",
        default_changed=False, native_source_changed=False))


def calibrate():
    if (TASK / "normalization.json").exists():
        return
    init_protocol()
    zero = policy.export(policy.neutral(), np.ones(8), TASK / "models/neutral.txt")
    fixed = np.zeros(8); fixed[[0,1,7]] = [1,.25,.1]
    manual = policy.export(fixed, np.ones(8), TASK / "models/original_fixed.txt")
    previous = TASK / "models/original_fixed_reference.txt"
    reference_weights = np.zeros(24); reference_weights[[0,1,8]] = [1,.25,.1]
    env.export_weights(reference_weights, previous)
    jobs, pairs, calibration = [], [], []
    for fam in FAMILIES:
        z = make(f"cal_zero_{fam}", fam, 81000, "neutral", "calibration", zero)
        hold = make(f"cal_hold_{fam}", fam, 81000, "neutral_hold", "calibration", zero, mode=9)
        m = make(f"cal_manual_{fam}", fam, 81000, "original_fixed", "calibration", manual)
        old = make(f"cal_previous_{fam}", fam, 81000, "previous_export", "calibration", str(previous))
        jobs += [z,hold,m,old]; pairs += [(z,hold),(m,old)]; calibration.append(z)
    write("calibration-jobs.json", list(map(str,jobs)))
    execute(jobs)
    for a,b in pairs:
        assert env.sha(a / "fct_output_file.txt") == env.sha(b / "fct_output_file.txt")
    total, n = np.zeros(8), 0
    for f in calibration:
        a = np.loadtxt(f / "transitions.csv.candidates.csv", delimiter=",", skiprows=1, ndmin=2)
        x = a[:, 5 + np.array(policy.FEATURES)]
        total += (x*x).sum(axis=0); n += len(x)
    scales = np.maximum(np.sqrt(total/n), 1e-6)
    write("normalization.json", dict(created_epoch=time.time(), scale=scales.tolist(), rows=n,
        sum_squares=total.tolist(), trajectories=list(map(str,calibration)),
        input_hashes={str(f):env.sha(f / "transitions.csv.candidates.csv") for f in calibration}))
    write("calibration-checks.json", dict(status="PASS", exact_pairs=[list(map(str,p)) for p in pairs]))
    print("CALIBRATED", scales.tolist(), flush=True)


def train():
    p, scale = read("protocol.json"), np.array(read("normalization.json")["scale"])
    if (TASK / "training-status.json").exists() and read("training-status.json")["stage"] == "complete":
        return
    theta = {(a,s):policy.neutral() for a in ARMS for s in FITS}
    rng = {s:np.random.default_rng(s) for s in FITS}
    shuffle = {s:np.random.default_rng(s+100000) for s in FITS}
    checkpoints, alljobs, monitors = [], [], []
    for rnd in range(p["rounds"]):
        directions = {s:rng[s].normal(size=(8,8)) for s in FITS}
        permutations = {(a,s):shuffle[s].permutation(16) if a=="shuffled" else np.arange(16) for a in ARMS for s in FITS}
        if (TASK / f"train-round{rnd}.json").exists():
            rec = read(f"train-round{rnd}.json")
            for u in rec["updates"]: theta[u["arm"],u["fit"]] = np.array(u["theta"])
            checkpoints += rec["updates"]; alljobs += rec["jobs"]; monitors += rec["monitor_jobs"]
            continue
        jobs, owners = [], {}
        for arm in ARMS:
            for fit in FITS:
                for d in range(8):
                    for sign in [-1,1]:
                        t = theta[arm,fit] + sign*p["sigma"][rnd]*directions[fit][d]
                        model = policy.export(t, scale, TASK / "perturbations" / f"{arm}_s{fit}_r{rnd}_d{d}_sign{sign}.txt")
                        for fam in FAMILIES:
                            f = make(f"train_{arm}_s{fit}_r{rnd}_d{d}_sign{sign}_{fam}", fam, p["training_traffic"][rnd], arm, "training", model, fit)
                            jobs.append(f); owners[f.name]=(arm,fit,d,sign)
        alljobs += list(map(str,jobs)); write("training-jobs.json", alljobs)
        grouped = defaultdict(list)
        for row in execute(jobs): grouped[owners[row["name"]]].append(row)
        updates, monitor_jobs = [], []
        for arm in ARMS:
            for fit in FITS:
                raw = np.array([policy.reward(grouped[arm,fit,d,sign], p['failure_return']) for d in range(8) for sign in [-1,1]])
                before = theta[arm,fit].copy()
                after, detail = policy.update(before, directions[fit], raw, permutations[arm,fit])
                theta[arm,fit] = after
                model = policy.export(after, scale, TASK / "checkpoints" / f"{arm}_s{fit}_r{rnd}.txt")
                updates.append(dict(arm=arm, fit=fit, round=rnd, before=before.tolist(),
                    directions=directions[fit].tolist(), raw_rewards=raw.tolist(), permutation=permutations[arm,fit].tolist(),
                    path=model, sha256=env.sha(model), **detail))
                for fam in FAMILIES:
                    monitor_jobs.append(make(f"center_{arm}_s{fit}_r{rnd}_{fam}", fam, p["training_traffic"][rnd], arm, "monitor", model, fit))
        monitor_rows = execute(monitor_jobs)
        checkpoints += updates; monitors += list(map(str,monitor_jobs))
        write(f"train-round{rnd}.json", dict(round=rnd, jobs=list(map(str,jobs)), updates=updates,
             monitor_jobs=list(map(str,monitor_jobs)), monitor_rows=monitor_rows))
        write("monitor-jobs.json", monitors)
        write("training-status.json", dict(stage="running", round=rnd, checkpoints=checkpoints))
        print("ROUND", rnd+1, "CENTER_REWARDS", {a:round(policy.reward([r for r in monitor_rows if f"center_{a}_" in r["name"]],p['failure_return']),3) for a in ARMS}, flush=True)
    write("training-status.json", dict(stage="complete", checkpoints=checkpoints))


def validate():
    if (TASK / "final-selection.json").exists():
        return
    p, status = read("protocol.json"), read("training-status.json")
    assert status["stage"] == "complete"
    scale = np.array(read("normalization.json")["scale"])
    configs = []
    for arm in ARMS:
        for rnd in p["validation_rounds"]:
            models = [u for u in status["checkpoints"] if u["arm"]==arm and u["round"]==rnd]
            if rnd == -1:
                models = [dict(arm=arm,fit=fit,round=-1,theta=policy.neutral().tolist(),
                    path=str(TASK/'models/neutral.txt'),sha256=env.sha(TASK/'models/neutral.txt')) for fit in FITS]
            configs.append(dict(id=f"{arm}_r{rnd}", arm=arm, round=rnd, models=models))
    for alpha in p["fixed_grid"]["remote"]:
        for work in p["fixed_grid"]["work"]:
            for trend in p["fixed_grid"]["trend"]:
                # Fixed rules keep the historical remote-change-times-age term.
                weights = np.zeros(24); weights[[0,1,8,11]] = [1,alpha,work,alpha*trend]
                cid=f"fixed_a{alpha}_w{work}_t{trend}"
                path=TASK / "fixed-grid" / f"{cid}.txt"
                env.export_weights(weights, path)
                configs.append(dict(id=cid, arm="fixed", models=[dict(path=str(path),fit=0,sha256=env.sha(path))]))
    jobs, owners = [], {}
    for c in configs:
        for m in c["models"]:
            for ts in p["validation_traffic"]:
                for fam in FAMILIES:
                    f=make(f"val_{c['id']}_s{m['fit']}_{fam}_{ts}",fam,ts,c["arm"],"validation",m["path"],m["fit"])
                    jobs.append(f); owners[f.name]=c["id"]
    write("validation-configs.json",configs); write("validation-jobs.json",list(map(str,jobs)))
    grouped=defaultdict(list)
    for r in execute(jobs): grouped[owners[r["name"]]].append(r)
    scores=[dict(**c, reward=policy.reward(grouped[c["id"]],p['failure_return']),
        all_complete=all(r['valid'] for r in grouped[c['id']])) for c in configs]
    selected={a:max((c for c in scores if c["arm"]==a),key=lambda c:(c["reward"],c["id"])) for a in ARMS+["fixed"]}
    write("validation-scores.json",scores)
    write("final-selection.json",dict(created_epoch=time.time(), selected=selected, protocol_sha256=env.sha(TASK/"protocol.json"),
        normalization_sha256=env.sha(TASK/"normalization.json"), training_status_sha256=env.sha(TASK/"training-status.json"), default_changed=False))
    print("FROZEN", {a:(v["id"],v["reward"]) for a,v in selected.items()},flush=True)


def test(ablate=False):
    phase="ablation" if ablate else "test"
    if (TASK / f"{phase}-status.json").exists():
        return
    p, frozen = read("protocol.json"), read("final-selection.json")
    specs=[]
    if ablate:
        scale=np.array(read("normalization.json")["scale"])
        for kind,col in [("no_local",0),("no_remote",1),("no_port_work",7)]:
            for m in frozen["selected"]["real"]["models"]:
                t=np.array(m["theta"]); t[col]=0
                path=policy.export(t,scale,TASK/"ablation-models"/f"{kind}_s{m['fit']}.txt")
                specs.append(dict(label=kind,fit=m["fit"],model=path))
    else:
        for arm in ARMS:
            specs += [dict(label=arm,fit=m["fit"],model=m["path"]) for m in frozen["selected"][arm]["models"]]
        specs += [dict(label=k,fit=0,model=str(TASK/"models"/f"{k}.txt")) for k in ["neutral","original_fixed"]]
        specs.append(dict(label="strong_fixed",fit=0,model=frozen["selected"]["fixed"]["models"][0]["path"]))
        specs.append(dict(label="conga",fit=0,model=None))
    jobs,index=[],[]
    for ts in p["test_traffic"]:
        for fam in TEST_FAMILIES:
            for s in specs:
                f=make(f"{phase}_{s['label']}_s{s['fit']}_{fam}_{ts}",fam,ts,s["label"],phase,s["model"],s["fit"],conga=s["label"]=="conga")
                j=json.loads((f/"job.json").read_text());j["selection_sha256"]=env.sha(TASK/"final-selection.json");env.dump(f/"job.json",j)
                assert Path(j["traffic"]).stat().st_mtime>=frozen["created_epoch"]
                jobs.append(f);index.append(dict(folder=str(f),label=s["label"],fit=s["fit"],family=fam,traffic=ts))
    write(f"{phase}-jobs.json",list(map(str,jobs))); write(f"{phase}-index.json",index)
    execute(jobs); write(f"{phase}-status.json",dict(stage="complete",runs=len(jobs)))
    print(phase.upper(),"COMPLETE",len(jobs),flush=True)


if __name__=="__main__":
    init_protocol()
    stages=dict(calibrate=calibrate,train=train,validate=validate,test=test,ablate=lambda:test(True))
    for name,fn in stages.items():
        if args.stage in ["all",name]: fn()
