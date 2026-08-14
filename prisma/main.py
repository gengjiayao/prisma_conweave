#!/usr/bin python3
# -*- coding: utf-8 -*-
""" -----Main file for the PRISMA project----- """


__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

### imports
from source.forwarder import Forwarder
from source.trainer import Trainer
from source.agent import Agent
from source.utils import save_model, save_all_models, convert_tb_data
from source.run_ns3 import run_ns3
from source.tb_logger import custom_plots, stats_writer_train, stats_writer_test
from source.argument_parser import parse_arguments
from source.utils import allocate_on_gpu, fix_seed
from source.rl_contract import RL_CORE_VERSION, checkpoint_manifest
from time import sleep, time
import numpy as np
import threading
import copy
import tensorflow as tf
import os
import datetime
import json
from tensorboard.plugins.hparams import api as hp
import subprocess, signal
import shlex
import pathlib


def _stop_owned_process(proc, timeout=5.0):
    """Reap a simulator launcher, terminating its process group if needed."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=timeout)


def main():
    ## Allocate GPU memory as needed
    allocate_on_gpu()
    
    ## Get the arguments from the parser
    params = parse_arguments()

    # 安全除法，避免 0 作分母导致崩溃
    def safe_div(num, den):
        try:
            num_f = float(num)
            den_f = float(den)
            return (num_f / den_f) if den_f else 0.0
        except Exception:
            return 0.0

    ## fix the seed
    fix_seed(params["seed"])

    # Baselines do not expose ns3-gym sockets.  Starting Forwarders for them
    # blocks forever in Ns3Env.reset(), so run and reap the simulator directly.
    if params["lb"] != "rl":
        for episode in range(params["numEpisodes"]):
            print(
                f"running non-RL baseline {params['lb']} "
                f"(episode {episode + 1}/{params['numEpisodes']})"
            )
            baseline_proc = run_ns3(params)
            try:
                return_code = baseline_proc.wait()
            except BaseException:
                _stop_owned_process(baseline_proc)
                raise
            if return_code != 0:
                raise RuntimeError(
                    f"Non-RL simulator exited with status {return_code}"
                )
        return None, None
    
    ## fill model version
    if "dqn_buffer" not in params["agent_type"] or params["train"] == 1:
        params["model_version"] = ""
    else:
        params["model_version"] = (params["load_path"] or "").split("/")[-1]
    
    ## check if the session already exists and the model is already trained
    if params["train"] == 1:
        pathlib.Path(params["logs_parent_folder"] + "/saved_models/").mkdir(parents=True, exist_ok=True)
        if os.path.exists(params["logs_parent_folder"] + "/saved_models/" + params["session_name"] + "/final"):
            if len(os.listdir(params["logs_parent_folder"] + "/saved_models/" + params["session_name"] + "/final")) > 0:
                print(f'The couple {params["seed"]} {params["traffic_matrix_index"]} already exists in : {params["logs_parent_folder"] + "/saved_models/" + params["session_name"]}')
                return None, None
    
    ## check if the test is already done   
    else:
        if os.path.exists(f"{params['logs_parent_folder']}/{params['session_name']}/test_results/{params['model_version']}"):
            if len(os.listdir(f"{params['logs_parent_folder']}/{params['session_name']}/test_results/{params['model_version']}")) > 0:
                ## check if the test load factor is already in the tensorboard file
                try: 
                    if int(100 * params["load_factor"]) in convert_tb_data(f"{params['logs_parent_folder']}/{params['session_name']}/test_results/{params['model_version']}")["step"].values:
                        print(f'The test session with load factor {params["load_factor"]} already exists in the {params["session_name"]}/test_results/{params["model_version"]} folder')
                        return None, None
                except:
                    pass
                            
    ## Setup writer for the global stats
    if params["train"] == 1:
        summary_writer_parent = tf.summary.create_file_writer(logdir=params["logs_folder"] )
        summary_writer_session = tf.summary.create_file_writer(logdir=params["global_stats_path"] )
        summary_writer_nb_arrived_pkts = tf.summary.create_file_writer(logdir=params["nb_arrived_pkts_path"] )
        summary_writer_nb_new_pkts = tf.summary.create_file_writer(logdir=params["nb_new_pkts_path"] )
        summary_writer_nb_lost_pkts = tf.summary.create_file_writer(logdir=params["nb_lost_pkts_path"] )

        ## write the session info (parameters)
        with tf.summary.create_file_writer(logdir=params["logs_folder"]).as_default():
            ## Adapt the dict to the hparams api
            dict_to_store = copy.deepcopy(params)
            dict_to_store["G"] = str(params["G"])
            dict_to_store["load_path"] = str(params["load_path"])
            dict_to_store["simArgs"] = str(params["simArgs"])
            # TensorBoard HParams only accepts bool/int/float/string values.
            # Keep the runtime None semantics while recording an explicit
            # sentinel for the optional independent traffic seed.
            dict_to_store["traffic_seed"] = (
                "legacy_default"
                if params["traffic_seed"] is None
                else str(params["traffic_seed"])
            )
            for key, value in tuple(dict_to_store.items()):
                if value is None:
                    dict_to_store[key] = "default"
            hp.hparams(dict_to_store)  # record the values used in this trial
    
        ## Define the custom categories in tensorboard
        with summary_writer_parent.as_default():
            tf.summary.experimental.write_raw_pb(
                    custom_plots().SerializeToString(), step=0
                )
    
    ## setup the agents (fix the static variables)
    Agent.init_static_vars(params)

    # 追加：全局 reward 参数与 writer（测试模式安全：writer=None，log_every=0）
    is_train = int(params.get("train", 1)) == 1

    # 统一的全局 reward 写入参数
    Agent.reward_ewma_alpha = 0.05
    Agent.reward_log_every  = 10 if is_train else 0
    Agent.global_reward_ewma = 0.0

    # 全局曲线 writer（训练时创建，测试为 None）
    Agent.reward_global_by_steps_writer = (
        tf.summary.create_file_writer(f"{params['logs_folder']}/reward_global/by_steps")
        if is_train else None
    )
    Agent.reward_global_by_time_writer = (
        tf.summary.create_file_writer(f"{params['logs_folder']}/reward_global/by_time")
        if is_train else None
    )
    
    ## start the profiler
    if params["profile_session"]:
        from viztracer import VizTracer
        tracer = VizTracer(tracer_entries=5000000, min_duration=100, max_stack_depth=20, output_file=f"{params['logs_parent_folder'].rstrip('/')}/{params['session_name']}/viztracer.json")
        tracer.start()
    
    ## Run tensorboard server
    tensorboard_process = None
    if params["start_tensorboard"]:
        args = shlex.split(f'python3 -m tensorboard.main --logdir={params["logs_folder"]} --port={params["tensorboard_port"]} --bind_all')
        tensorboard_process = subprocess.Popen(args).pid
        print(f"Tensorboard server started with pid {tensorboard_process}")

    # 仅为 leaf(ToR) 启动 Agent：解析 topo 和映射得到 leaf 的 overlay 索引集合
    def _parse_leaf_overlay_indices(topo_path, idx2swid_map_path):
        idx2swid = {}
        with open(idx2swid_map_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                i, sid = line.split()
                idx2swid[int(i)] = int(sid)

        with open(topo_path, 'r') as f:
            head = f.readline().strip().split()
            node_num = int(head[0]); switch_num = int(head[1]); link_num = int(head[2])
            host_num = node_num - switch_num
            _ = f.readline()  # switch id list
            leaf_switch_ids = set()
            for _ in range(link_num):
                parts = f.readline().strip().split()
                if len(parts) < 2:
                    continue
                src, dst = int(parts[0]), int(parts[1])
                if (src < host_num and dst >= host_num) or (dst < host_num and src >= host_num):
                    switch_id = dst if dst >= host_num else src
                    leaf_switch_ids.add(switch_id)

        swid2idx = {v: k for k, v in idx2swid.items()}
        enabled_nodes = sorted([swid2idx[sid] for sid in leaf_switch_ids if sid in swid2idx]) #只保留leaf过滤掉spine
        print("Total nodes in G:", len(params["G"].nodes()))
        print("Enabled leaf agent overlay indices:", enabled_nodes)
        return enabled_nodes

    # 加载 overlayIndex->switchId 映射（供打印与对齐使用）
    def _load_idx2swid(map_path):
        d = {}
        try:
            with open(map_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    i, sid = line.split()
                    d[int(i)] = int(sid)
        except Exception:
            d = {}
        return d

    forwarders = {}
    trainers = {}
    trainer_threads = []
    ns3_proc = None
    for episode in range(params["numEpisodes"]):
        ## run ns3 simulator
        print("running ns-3")
        ns3_proc = run_ns3(params) #此处调用run_ns3启动原PRISMA的ns3部分！！！
        Agent.reset()
        ## run the agents threads
        enabled_nodes = _parse_leaf_overlay_indices(
            params["conweave_topo_file_path"],
            params["index_to_switch_id_map_path"],
        )
        idx2swid = _load_idx2swid(params["index_to_switch_id_map_path"])  # 仅用于打印
        forwarder_threads = []
        for index in enabled_nodes:
            # 友好打印：映射 overlay_index -> switch_id
            swid = idx2swid.get(index, None)
            print(f"[Python] Starting Forwarder/Trainer for leaf idx={index}, swid={swid}")
            if episode == 0:
                ## create the agent class instance
                forwarders[index] = Forwarder(index, agent_type=params["agent_type"], train=params["train"])
            else:
                forwarders[index].reset(init=False)
            ## start the agent forwarder thread
            th1 = threading.Thread(target=forwarders[index].run, args=())
            th1.start()
            forwarder_threads.append(th1)
            if params["train"]:
                if episode == 0:
                    trainers[index] = Trainer(index, agent_type=params["agent_type"], train=params["train"])
                    ## start the agent trainer thread
                    th2 = threading.Thread(target=trainers[index].run, args=(), name=f"trainer-{index}", daemon=True)
                    th2.start()
                    trainer_threads.append(th2)
                else:
                    trainers[index].reset_episode()

            
        sleep(1)
        
        snapshot_index = 1
        ## wait until simulation complete and update info about the env at each timestep (基于线程存活)
        def _any_alive(lst):
            return any(t.is_alive() for t in lst)
        # Only wait for forwarder threads; trainer threads are daemonized and should not block shutdown
        while _any_alive(forwarder_threads):
            sleep(params["logging_timestep"])
            if params["train"] == 1:
                stats_writer_train(summary_writer_session, summary_writer_nb_arrived_pkts, summary_writer_nb_lost_pkts, summary_writer_nb_new_pkts, Agent)
                ## check if it is time to save a snapshot of the models
                if params["snapshot_interval"] > 0 and Agent.curr_time > 0:
                    if (Agent.base_curr_time + Agent.curr_time) > (snapshot_index * params["snapshot_interval"]):
                        print(f"Saving model at time {Agent.curr_time} with index {snapshot_index}")
                        save_all_models(Agent.agents, list(forwarders.keys()), params["session_name"], snapshot_index, 1, root=params["logs_parent_folder"] + "/saved_models/", snapshot=True)
                        snapshot_index += 1
                        
        print(f""" Summary of the Episode {episode}:
                Simulation time = {Agent.curr_time},
                Total Iterations = {Agent.total_nb_iterations},
                Total number of Transitions = {Agent.nb_transitions},
                Overlay Total injected packets = {Agent.sim_injected_packets}, 
                Global Total injected packets = {Agent.sim_global_injected_packets}, 
                Overlay arrived packets = {Agent.sim_delivered_packets},
                Global arrived packets = {Agent.sim_global_delivered_packets},
                Overlay lost packets = {Agent.sim_dropped_packets},
                Global lost packets = {Agent.sim_global_dropped_packets},
                Overlay buffered packets = {Agent.sim_buffered_packets},
                Global buffered packets = {Agent.sim_global_buffered_packets},
                Overlay lost ratio = {safe_div(Agent.sim_dropped_packets, Agent.sim_injected_packets)},
                Global lost ratio = {safe_div(Agent.sim_global_dropped_packets, Agent.sim_global_injected_packets)},
                Overlay e2e delay = {Agent.sim_avg_e2e_delay},
                Global e2e delay = {Agent.sim_global_avg_e2e_delay},
                Overlay Cost = {Agent.sim_cost},
                Global Cost = {Agent.sim_global_cost},
                Hops = {safe_div(Agent.total_hops, Agent.sim_delivered_packets)},
                OverheadRatio = {Agent.sim_signaling_overhead}
                """) 
        for idx in list(forwarders.keys()):
            forwarders[idx].env.close()
        try:
            ns3_return_code = ns3_proc.wait(timeout=120.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ns-3 launcher did not exit after all Forwarders stopped") from exc
        if ns3_return_code != 0:
            raise RuntimeError(f"ns-3 launcher exited with status {ns3_return_code}")
        Agent.base_curr_time += Agent.curr_time

    # Stop all model updates before writing test statistics or final checkpoints.
    if params["train"] and trainer_threads:
        for trainer in trainers.values():
            trainer.stop()
        join_deadline = time() + 30.0
        for trainer_thread in trainer_threads:
            trainer_thread.join(timeout=max(0.0, join_deadline - time()))
        alive_trainers = [thread.name for thread in trainer_threads if thread.is_alive()]
        if alive_trainers:
            raise RuntimeError(
                "Trainer threads did not stop; refusing to save a potentially inconsistent checkpoint: "
                + ", ".join(alive_trainers)
            )
        print("All trainer threads stopped; draining deterministic replay backlog")
        for index in sorted(trainers):
            drained = trainers[index].drain()
            print(
                f"[TRAIN-DRAIN] node={index} drained={drained} "
                f"total_gradient_steps={trainers[index].gradient_step_idx}"
            )

    # Persist the exact per-node action distribution so an output ID plus the
    # known session name is enough for post-run diagnosis; terminal output is
    # not required.
    policy_action_stats = {}
    for index, forwarder in forwarders.items():
        counts = np.asarray(forwarder.policy_action_counts, dtype=np.int64)
        total = int(np.sum(counts))
        max_action_ratio = float(np.max(counts) / total) if total > 0 else 0.0
        greedy_counts = np.asarray(forwarder.greedy_action_counts, dtype=np.int64)
        greedy_total = int(np.sum(greedy_counts))
        greedy_max_action_ratio = (
            float(np.max(greedy_counts) / greedy_total) if greedy_total > 0 else 0.0
        )
        policy_action_stats[str(index)] = {
            "total": total,
            "counts": counts.tolist(),
            "ratios": (
                (counts / total).tolist()
                if total > 0
                else np.zeros_like(counts, dtype=float).tolist()
            ),
            "max_action_ratio": max_action_ratio,
            "shadow_greedy": {
                "total": greedy_total,
                "counts": greedy_counts.tolist(),
                "ratios": (
                    (greedy_counts / greedy_total).tolist()
                    if greedy_total > 0
                    else np.zeros_like(greedy_counts, dtype=float).tolist()
                ),
                "max_action_ratio": greedy_max_action_ratio,
            },
            "exploration": {
                "random_decisions": int(forwarder.random_action_decisions),
                "greedy_decisions": int(forwarder.greedy_action_decisions),
                "behavior_matches_greedy": int(forwarder.behavior_matches_greedy),
                "epsilon_mean": (
                    float(forwarder.epsilon_sum / total) if total > 0 else 0.0
                ),
                "epsilon_min": forwarder.epsilon_min,
                "epsilon_max": forwarder.epsilon_max,
            },
            "transition_alignment": {
                "transitions_written": int(forwarder.transitions_written),
                "action_sequence_mismatches": int(forwarder.action_sequence_mismatches),
                "first_action_seq": forwarder.first_action_seq,
                "last_action_seq": forwarder.last_action_seq,
            },
            "observation": forwarder.observation_diagnostics(),
            "reward_by_action": forwarder.reward_diagnostics(),
            "q_values": forwarder.q_diagnostics(),
            "training": (
                trainers[index].training_stats() if index in trainers else None
            ),
        }
        if total >= 100 and max_action_ratio >= 0.9:
            print(
                f"WARNING: node {index} policy is highly concentrated "
                f"(max action ratio={max_action_ratio:.4f})"
            )
    os.makedirs(params["logs_folder"], exist_ok=True)
    policy_action_path = os.path.join(params["logs_folder"], "policy_actions.json")
    eval_prior_only = bool(params.get("eval_prior_only", 0))
    if params["train"]:
        fct_policy = "epsilon_greedy_training"
    elif eval_prior_only:
        fct_policy = "rl_native_prior_only"
    elif float(params.get("eval_epsilon", 0.0)) == 0.0:
        fct_policy = "greedy_frozen"
    else:
        fct_policy = "epsilon_greedy_frozen"
    with open(policy_action_path, "w") as policy_action_file:
        json.dump(
            {
                "rl_core_version": RL_CORE_VERSION,
                "seed": int(params["seed"]),
                "train": int(bool(params["train"])),
                "eval_epsilon": float(params.get("eval_epsilon", 0.0)),
                "eval_prior_only": int(eval_prior_only),
                "fct_policy": fct_policy,
                "fct_is_final_checkpoint_policy": bool(
                    not params["train"]
                    and float(params.get("eval_epsilon", 0.0)) == 0.0
                    and not eval_prior_only
                ),
                "nodes": policy_action_stats,
            },
            policy_action_file,
            indent=2,
            sort_keys=True,
        )
    print(f"Policy action statistics saved to {policy_action_path}")

    traffic_seed = params.get("traffic_seed")
    traffic_seed = None if traffic_seed is None else int(traffic_seed)
    run_manifest = checkpoint_manifest()
    run_manifest.update({
        "seed": int(params["seed"]),
        "traffic_seed": traffic_seed,
        "train": int(bool(params["train"])),
        "load_path": params.get("load_path"),
        "session_name": params["session_name"],
        "evaluation": {
            "eval_epsilon": float(params.get("eval_epsilon", 0.0)),
            "eval_prior_only": int(eval_prior_only),
            "fct_policy": fct_policy,
        },
        "optimization": {
            "batch_size": int(params["batch_size"]),
            "learning_starts": int(params["learning_starts"]),
            "train_every": int(params["train_every"]),
            "target_update_interval": int(params["target_update_interval"]),
            "exploration_initial_eps": float(params["exploration_initial_eps"]),
            "exploration_final_eps": float(params["exploration_final_eps"]),
            "exploration_schedule_timesteps": int(params["exploration_schedule_timesteps"]),
        },
        "simulation": {
            "lb": params["lb"],
            "cc": params["cc"],
            "pfc": int(params["pfc"]),
            "irn": int(params["irn"]),
            "simul_time": float(params["simul_time"]),
            "buffer": int(params["buffer"]),
            "netload": int(params["netload"]),
            "topo": params["topo"],
            "traffic_seed": traffic_seed,
            "conweave_timing_us": {
                "extra_reply_deadline": params.get("cwh_extra_reply_deadline"),
                "path_pause_time": params.get("cwh_path_pause_time"),
                "extra_voq_flush_time": params.get("cwh_extra_voq_flush_time"),
                "default_voq_waiting_time": params.get("cwh_default_voq_waiting_time"),
                "tx_expiry_time": params.get("cwh_tx_expiry_time"),
            },
        },
    })
    with open(os.path.join(params["logs_folder"], "run_manifest.json"), "w") as manifest_file:
        json.dump(run_manifest, manifest_file, indent=2, sort_keys=True)
    
    ## write the results for the test session
    if params["train"] == 0:
       stats_writer_test(params["logs_folder"] + "/test_results", Agent)

    ## save models        
    #if params["save_models"] and Agent.curr_time >= params["simTime"]-5:
    if params["save_models"] and (Agent.curr_time > 0 or Agent.nb_transitions > 0):
        save_all_models(Agent.agents, list(forwarders.keys()), params["session_name"], 1, 1, root=params["logs_parent_folder"] + "/saved_models/", snapshot=False)

    ## save the profiler results
    if params["profile_session"]:
        tracer.stop()
        tracer.save() 
        
    return (ns3_proc, tensorboard_process)

if __name__ == '__main__':
    ## create a process group
    import traceback
    ns3_proc, tb_process = None, None
    # os.setpgrp()
    try:
        print("starting process group")
        start_time = time()
        ns3_proc, tb_process = main()
        print("Elapsed time = ", str(datetime.timedelta(seconds= time() - start_time)))
    except Exception:
        traceback.print_exc()
        # write the error in the log file
        with open("examples/error.log", "a") as f:
            traceback.print_exc(file=f)
        raise
    finally:
        print("graceful shutdown ...")
        # 1) 先发送 ns-3 关闭指令（ZMQ Close），给析构打印留机会
        try:
            from source.agent import Agent
            for env in getattr(Agent, "envs", {}).values():
                if env is None: continue
                try:
                    env.ns3ZmqBridge.send_close_command()
                except Exception:
                    pass
        except Exception:
            pass

        # 2) 给 ns-3 一点时间把 stop 消息处理完并运行析构
        sleep(2.0)

        # 3) Reap the owned launcher; terminate its process group only if it
        # is still alive.  Keeping the Popen handle avoids zombie children.
        try:
            _stop_owned_process(ns3_proc)
        except Exception:
            pass

        try:
            if tb_process:
                try:
                    os.kill(int(tb_process), signal.SIGTERM)
                except Exception:
                    pass
        except Exception:
            pass
