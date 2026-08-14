### imports
import tensorflow as tf
import networkx as nx
import numpy as np
from ns3_model import ns3env
from source.learner import DQN_AGENT
from source.utils import load_model, LinearSchedule, optimal_routing_decision
from source.rl_contract import (
    EGRESS_FEATURE_NAMES,
    RL_CORE_VERSION,
    build_aligned_transition,
    is_valid_environment_step,
    native_action_prior,
    preprocess_observation,
    validate_observation_shape,
)
from source.models import *
import operator
import pandas as pd
import time
from source.agent import Agent

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

class Forwarder(Agent):
    """Forwarder agent class.
    """
    def __init__(self, index, agent_type="dqn", train=True):
        """Initialize the forwarder agent.
            Args:
                index (int): agent index.
                agent_type (str): agent type.
                train (bool): train or test mode.
        """
        # initialize the parent class
        super().__init__(index, agent_type, train)

        # define the communication port 端口basePort+index
        self.port = Agent.basePort + index
        print("Index: ", self.index, "Port: ", self.port)
        self.transition_number = 0

        # reset the env
        self.reset()

        
    def reset(self, init=True):
        """ Reset the ns3-gym env 
        """
        if Agent.G == None or Agent.numNodes == 0:
            raise("Please make sure you input the topology")

        ### define the ns3 env 用ns3env建一个ZMQ环境
        self.env = ns3env.Ns3Env(port=int(self.port), stepTime=Agent.stepTime, startSim=Agent.startSim, simSeed=Agent.seed, simArgs=Agent.simArgs, debug=Agent.debug)
        obs = self.env.reset()
        Agent.envs[self.index] = self.env
        self.obs_dim = int(self.env.observation_space.shape[0])
        validate_observation_shape(self.obs_dim, int(self.env.action_space.n))

        # 首拍默认无控制包，pkt_id 置为 -1，防止首拍未初始化访问
        self.signaling = False
        self.pkt_id = -1
        self.ecmp_action = -1
        if init:
            ## define the agent
            if self.agent_type == "dqn_buffer":
                ## declare the DQN buffer model
                model = DQN_buffer_model
            elif self.agent_type == "dqn_buffer_lite":
                ## declare the DQN buffer lite model
                model = DQN_buffer_lite_model
            elif self.agent_type == "dqn_buffer_lighter":
                ## declare the DQN buffer lighter model
                model = DQN_buffer_lighter_model    
            elif self.agent_type == "dqn_buffer_lighter_2":
                ## declare the DQN buffer lighter_2 model
                model = DQN_buffer_lighter_2_model
            elif self.agent_type == "dqn_buffer_lighter_3":
                ## declare the DQN buffer lighter_3 model
                model = DQN_buffer_lighter_3_model
            elif self.agent_type == "dqn_buffer_ff":
                ## declare the DQN buffer ff model
                model = DQN_buffer_ff_model
            elif self.agent_type == "dqn_routing":
                ## declare the DQN buffer model
                model = DQN_routing_model
                
            if "dqn" in self.agent_type:
                # 计算每邻居特征维度 K：(obs_dim - kObsHeaderDims) / num_actions
                # 新观测：[dstOverlay, lastAction] + N_neighbors * K
                try:
                    kObsHeaderDims = 2  # 与 C++ 端 kObsHeaderDims 对齐
                    K = int((self.env.observation_space.shape[0] - kObsHeaderDims) // self.env.action_space.n)
                except Exception:
                    K = self.env.action_space.n  # 兜底（不应触发）
                if not hasattr(Agent, 'K'):
                    Agent.K = K
                Agent.agents[self.index] = DQN_AGENT(
                    q_func=model,
                    observation_shape=self.env.observation_space.shape,
                    num_actions=self.env.action_space.n,
                    num_nodes=Agent.numNodes,
                    input_size_splits = [1,  # dstOverlay
                                        1,  # lastAction
                                        self.env.action_space.n * Agent.K,  # neighbor features
                                        ],
                    lr=Agent.lr,
                    gamma=Agent.gamma,
                    neighbors_degrees=[len(list(Agent.G.neighbors(x))) for x in self.neighbors],
                    grad_norm_clipping=10.0,  # 【修复】添加梯度裁剪，防止梯度爆炸
                )
            elif self.agent_type == "opt":
                Agent.agents[self.index] = optimal_routing_decision
            elif self.agent_type == "sp":
                Agent.agents[self.index] = nx.shortest_path
            else:
                raise ValueError("Unknown agent type")

            ## compute big signaling delay (only for non-ideal signaling)
            if "dqn" in self.agent_type and Agent.signaling_type != "ideal":
                self.nn_size = np.sum([np.prod(x.shape) for x in Agent.agents[self.index].q_network.trainable_weights])*32
                self.big_signaling_delay = (self.nn_size/ Agent.link_cap) + (Agent.link_delay*0.001)
                print("node:", self.index, "big signaling delay: ", self.big_signaling_delay, self.nn_size)
            elif "dqn" in self.agent_type and Agent.signaling_type == "ideal":
                # For ideal signaling, no actual signaling is used, so delay is 0
                self.nn_size = np.sum([np.prod(x.shape) for x in Agent.agents[self.index].q_network.trainable_weights])*32
                self.big_signaling_delay = 0.0
                print("node:", self.index, "ideal signaling mode - no actual signaling delay, nn_size:", self.nn_size)
        
            ### compute small signaling delay
            if Agent.signaling_type == "NN":
                self.small_signaling_pkt_size = 64 + 8 + (8 * (len(self.neighbors)+1)) # header + reward (float) + s' (double)
                self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + (Agent.link_delay*0.001)
                if self.sync_step < 0:
                    self.sync_step = self._compute_sync_step(ratio=Agent.sync_ratio)
            elif Agent.signaling_type == "target":
                self.small_signaling_pkt_size = 64 + 8  +8 # header + target (float)
                self.small_signaling_delay = (self.small_signaling_pkt_size / Agent.link_cap) + (Agent.link_delay*0.001)
            ## load the models
            if Agent.load_path is not None and "dqn" in self.agent_type:
                # load the model
                loaded_models = load_model(
                    Agent.load_path,
                    self.index,
                    expected_contract_version=RL_CORE_VERSION,
                )
                if loaded_models is not None:
                    print("Restoring from {} for node {}".format(Agent.load_path, self.index))
                    if self.index >= len(loaded_models) or loaded_models[self.index] is None:
                        raise ValueError(f"Checkpoint is missing node {self.index}")
                    Agent.agents[self.index].q_network.set_weights(loaded_models[self.index].get_weights())
                    Agent.agents[self.index].update_target()
        
            ## define the log file for exploration value
            self.tb_writer_dict = {"exploration":  tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/exploration/node_{self.index}')}

            # 每个 agent 的 reward 写入器（两条横轴）
            self.tb_writer_dict["reward_steps"] = tf.summary.create_file_writer(
                f"{Agent.logs_folder}/reward/node_{self.index}/by_steps"
            )
            self.tb_writer_dict["reward_time"]  = tf.summary.create_file_writer(
                f"{Agent.logs_folder}/reward/node_{self.index}/by_time"
            )

            # 本地 EWMA 状态
            self.r_alpha = getattr(Agent, "reward_ewma_alpha", 0.05)
            self.r_ewma  = 0.0
        
        # 默认的调试/写频参数（已存在则不覆盖）
        if not hasattr(Agent, "reward_log_every"):
            Agent.reward_log_every = 1000
        if not hasattr(Agent, "reward_ewma_alpha"):
            Agent.reward_ewma_alpha = 0.05
        if not hasattr(Agent, "reward_debug_every"):
            Agent.reward_debug_every = 5000
        if not hasattr(Agent, "debug_reward"):
            Agent.debug_reward = False
        ## env trackers definition
        self.count_arrived_packets = 0
        self.count_new_pkts = 0
        self.update_eps = 0
        Agent.sync_counters[self.index] = -1
        
        ## define action history and nb_actions
        self.action_history = np.ones(self.env.action_space.n)
        self.nb_actions = np.sum(self.action_history)
        # Exact counts for real routing decisions (no pseudocounts).
        self.policy_action_counts = np.zeros(self.env.action_space.n, dtype=np.int64)
        self.greedy_action_counts = np.zeros(self.env.action_space.n, dtype=np.int64)
        self.policy_decision_count = 0
        self.random_action_decisions = 0
        self.greedy_action_decisions = 0
        self.behavior_matches_greedy = 0
        self.epsilon_sum = 0.0
        self.epsilon_min = None
        self.epsilon_max = None
        self.transitions_written = 0
        self.action_sequence_mismatches = 0
        self.first_action_seq = None
        self.last_action_seq = None
        self.observation_stat_names = [
            "destination_overlay",
            "last_action",
            *EGRESS_FEATURE_NAMES,
        ]
        stat_dim = len(self.observation_stat_names)
        self.raw_observation_min = np.full(stat_dim, np.inf, dtype=np.float64)
        self.raw_observation_max = np.full(stat_dim, -np.inf, dtype=np.float64)
        self.raw_observation_sum = np.zeros(stat_dim, dtype=np.float64)
        self.raw_observation_count = np.zeros(stat_dim, dtype=np.int64)
        self.processed_observation_min = np.full(stat_dim, np.inf, dtype=np.float64)
        self.processed_observation_max = np.full(stat_dim, -np.inf, dtype=np.float64)
        self.processed_observation_sum = np.zeros(stat_dim, dtype=np.float64)
        self.processed_observation_count = np.zeros(stat_dim, dtype=np.int64)
        self.reward_action_count = np.zeros(self.env.action_space.n, dtype=np.int64)
        self.reward_action_sum = np.zeros(self.env.action_space.n, dtype=np.float64)
        self.reward_action_sum_sq = np.zeros(self.env.action_space.n, dtype=np.float64)
        self.q_value_count = 0
        self.q_value_min = np.inf
        self.q_value_max = -np.inf
        self.q_value_sum = 0.0
        self.q_value_sum_sq = 0.0
        self.q_margin_min = np.inf
        self.q_margin_max = -np.inf
        self.q_margin_sum = 0.0
        # Create the schedule for exploration (allow explicit decay length).
        schedule_steps = int(Agent.iterationNum)
        if getattr(Agent, "exploration_schedule_timesteps", 0) > 0:
            schedule_steps = int(Agent.exploration_schedule_timesteps)

        self.exploration = LinearSchedule(schedule_timesteps=schedule_steps,
                                    initial_p=Agent.exploration_initial_eps,
                                    final_p=Agent.exploration_final_eps)

    def _observation_groups(self, observation):
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        num_actions = int(self.env.action_space.n)
        feature_count = (values.size - 2) // num_actions
        ports = values[2:].reshape(num_actions, feature_count)
        return [values[0:1], values[1:2]] + [ports[:, i] for i in range(feature_count)]

    def _record_observation(self, raw_observation, processed_observation):
        for groups, mins, maxes, sums, counts in (
            (
                self._observation_groups(raw_observation),
                self.raw_observation_min,
                self.raw_observation_max,
                self.raw_observation_sum,
                self.raw_observation_count,
            ),
            (
                self._observation_groups(processed_observation),
                self.processed_observation_min,
                self.processed_observation_max,
                self.processed_observation_sum,
                self.processed_observation_count,
            ),
        ):
            for index, group in enumerate(groups):
                mins[index] = min(mins[index], float(np.min(group)))
                maxes[index] = max(maxes[index], float(np.max(group)))
                sums[index] += float(np.sum(group))
                counts[index] += int(group.size)

    def _record_q_values(self, q_values):
        values = np.asarray(q_values, dtype=np.float64).reshape(-1)
        self.q_value_count += int(values.size)
        self.q_value_min = min(self.q_value_min, float(np.min(values)))
        self.q_value_max = max(self.q_value_max, float(np.max(values)))
        self.q_value_sum += float(np.sum(values))
        self.q_value_sum_sq += float(np.sum(values * values))
        if values.size >= 2:
            top_two = np.partition(values, -2)[-2:]
            margin = float(np.max(top_two) - np.min(top_two))
            self.q_margin_min = min(self.q_margin_min, margin)
            self.q_margin_max = max(self.q_margin_max, margin)
            self.q_margin_sum += margin

    def _record_action_reward(self, action, reward):
        action = int(action)
        reward = float(reward)
        self.reward_action_count[action] += 1
        self.reward_action_sum[action] += reward
        self.reward_action_sum_sq[action] += reward * reward

    @staticmethod
    def _moments(count, total, total_sq):
        if int(count) <= 0:
            return {"count": 0, "mean": None, "std": None}
        mean = float(total / count)
        variance = max(0.0, float(total_sq / count) - mean * mean)
        return {"count": int(count), "mean": mean, "std": float(np.sqrt(variance))}

    def observation_diagnostics(self):
        result = {}
        for index, name in enumerate(self.observation_stat_names):
            result[name] = {
                "raw": {
                    "count": int(self.raw_observation_count[index]),
                    "min": (
                        float(self.raw_observation_min[index])
                        if self.raw_observation_count[index] else None
                    ),
                    "max": (
                        float(self.raw_observation_max[index])
                        if self.raw_observation_count[index] else None
                    ),
                    "mean": (
                        float(self.raw_observation_sum[index] / self.raw_observation_count[index])
                        if self.raw_observation_count[index] else None
                    ),
                },
                "processed": {
                    "count": int(self.processed_observation_count[index]),
                    "min": (
                        float(self.processed_observation_min[index])
                        if self.processed_observation_count[index] else None
                    ),
                    "max": (
                        float(self.processed_observation_max[index])
                        if self.processed_observation_count[index] else None
                    ),
                    "mean": (
                        float(self.processed_observation_sum[index] / self.processed_observation_count[index])
                        if self.processed_observation_count[index] else None
                    ),
                },
            }
        return result

    def reward_diagnostics(self):
        return {
            str(action): self._moments(
                self.reward_action_count[action],
                self.reward_action_sum[action],
                self.reward_action_sum_sq[action],
            )
            for action in range(int(self.env.action_space.n))
        }

    def q_diagnostics(self):
        if self.q_value_count <= 0:
            return {"count": 0}
        moments = self._moments(
            self.q_value_count,
            self.q_value_sum,
            self.q_value_sum_sq,
        )
        decision_count = int(self.policy_decision_count)
        moments.update({
            "min": float(self.q_value_min),
            "max": float(self.q_value_max),
            "top2_margin_mean": (
                float(self.q_margin_sum / decision_count) if decision_count else None
            ),
            "top2_margin_min": (
                float(self.q_margin_min) if decision_count else None
            ),
            "top2_margin_max": (
                float(self.q_margin_max) if decision_count else None
            ),
        })
        return moments

  

    def step(self, obs):
        """
        Do an env step
        
        """
        ## take the action
        if obs[0] == self.index or self.transition_number < 1 or obs[0] in(-1, 1000): # pkt arrived to dst or it is a train step, ignore the action
            self.action = 0
        else:
            self.action = self.take_action(obs)
            # 防止 temp_obs 内存爆炸：定期清理最旧的条目
            if len(Agent.temp_obs) > 5000:
                # 转换为列表切片，直接取最旧的 2000 个 key
                keys_to_remove = list(Agent.temp_obs.keys())[:2000]
                for key in keys_to_remove:
                    Agent.temp_obs.pop(key, None)
                    # 【关键】同时清理 tracking 字典，防止僵尸条目残留
                    Agent.pkt_tracking_dict.pop(key, None)

            # 硬 TTL 兜底：定期剔除超时样本，并按“丢包样本”写入回放
            # 为了避免误杀在途包，先关闭这个功能
            # try:
            #     if Agent.total_nb_iterations % 300 == 0:  # 周期更紧：每 300 步做一次 TTL 清理
            #         now = Agent.curr_time
            #         # 与 C++ reward 窗口对齐：ttl = max(2*W, 5*RTT)
            #         W = 1e-3  # 与 m_rewardWinSec=1ms 对齐
            #         rtt_guess = getattr(Agent, "rtt_guess", 8.5e-6)
            #         ttl = max(2.0 * W, 5.0 * rtt_guess)
            #         stale_keys = [k for k, v in list(Agent.temp_obs.items()) if now - v.get("time", now) > ttl]
            #         for stale_id in stale_keys:
            #             rec = Agent.temp_obs.pop(stale_id, None)
            #             if rec is None:
            #                 continue
            #             try:
            #                 if Agent.agents[rec["node"]]:
            #                     next_hop_degree = len(list(Agent.G.neighbors(self.neighbors[rec["action"]])))
            #                     Agent.agents[rec["node"]].update_num_taken_actions(rec["action"], next_hop_degree)
            #             except:
            #                 next_hop_degree = len(list(Agent.G.neighbors(self.index))) if Agent.G is not None else 1
            #             rew = self._get_reward_lost_pkt()
            #             obs_shape = next_hop_degree * getattr(Agent, 'K', 1)
            #             if Agent.loss_penalty_type == "fixed" and self.train:
            #                 Agent.replay_buffer[self.index].add(
            #                     np.array(rec["obs"], dtype=float).squeeze(),
            #                     rec["action"],
            #                     rew,
            #                     np.array([rec["obs"][0]] + [0]*(obs_shape), dtype=float).squeeze(),
            #                     True
            #                 )
            #             Agent.node_lost_pkts += 1
            #             Agent.pkt_tracking_dict.pop(int(stale_id), None)
            # except Exception:
            #     pass
            # =========== TTL惩罚已关闭 ===========
            
        ### Apply the action
        return self.env.step(self.action)
    
    def _get_reward_lost_pkt(self):
        """ Compute the reward when the packet is lost
        """
        if Agent.loss_penalty_type == "fixed":
            return -abs(float(Agent.loss_penalty))
        elif Agent.loss_penalty_type == "constrained":
            pass
    
    def take_action(self, obs):
        """ Take an action given obs

        Args :
            obs (list): observation list
        """
        if "dqn" in self.agent_type:
            actions_probs = None
            if Agent.smart_exploration:
                actions_probs = 1-(self.action_history/self.nb_actions)
                actions_probs /=sum(actions_probs)
            ### Take action using the NN
            obs_in = preprocess_observation(
                np.array([obs], dtype=np.float32),
                num_actions=int(self.env.action_space.n),
            )
            self._record_observation(obs, obs_in[0])
            eval_epsilon = float(getattr(Agent, "eval_epsilon", 0.0))
            stochastic = bool(self.train or eval_epsilon > 0.0)
            update_eps = (
                float(self.exploration.value(self.policy_decision_count))
                if self.train
                else eval_epsilon
            )
            self.update_eps = update_eps
            if self.train:
                with self.tb_writer_dict["exploration"].as_default():
                    tf.summary.scalar(
                        'exploration_value_over_decisions',
                        update_eps,
                        step=self.policy_decision_count,
                    )
                    tf.summary.scalar(
                        'exploration_value_over_time',
                        update_eps,
                        step=int((Agent.base_curr_time + Agent.curr_time) * 1e6),
                    )
            if bool(getattr(Agent, "eval_prior_only", False)):
                if self.train or update_eps != 0.0:
                    raise RuntimeError(
                        "Prior-only policy is valid only for deterministic frozen evaluation"
                    )
                q_values_array = native_action_prior(
                    obs_in,
                    num_actions=int(self.env.action_space.n),
                )
                action = int(np.argmax(q_values_array[0]))
                greedy_action = action
                used_random = False
            else:
                action_tensor, greedy_tensor, random_tensor, q_values = Agent.agents[self.index].step(
                    obs_in,
                    stochastic,
                    update_eps,
                    actions_probs=actions_probs,
                    return_diagnostics=True,
                )
                q_values_array = q_values.numpy()
                action = int(action_tensor.numpy().item())
                greedy_action = int(greedy_tensor.numpy().item())
                used_random = bool(random_tensor.numpy().item())
            self._record_q_values(q_values_array)
            self.policy_action_counts[action] += 1
            self.greedy_action_counts[greedy_action] += 1
            self.policy_decision_count += 1
            self.epsilon_sum += update_eps
            self.epsilon_min = update_eps if self.epsilon_min is None else min(self.epsilon_min, update_eps)
            self.epsilon_max = update_eps if self.epsilon_max is None else max(self.epsilon_max, update_eps)
            if used_random:
                self.random_action_decisions += 1
            else:
                self.greedy_action_decisions += 1
            if action == greedy_action:
                self.behavior_matches_greedy += 1
            if Agent.smart_exploration:
                self.action_history[action] += 1
                self.nb_actions += 1
        elif self.agent_type == "sp":
            action = self.neighbors.index(Agent.agents[self.index](Agent.G, self.index, obs[0])[1])
        elif self.agent_type == "opt":
            track =  Agent.pkt_tracking_dict[int(self.pkt_id)]
            action, track["tag"] = optimal_routing_decision(Agent.G, Agent.optimal_routing_mat, Agent.optimal_rejected_mat, self.index, track["src"], track["dst"], track["tag"])
        return action
    
    def treat_info(self, info):
        """ Treat the info received from the ns3 simulator
        Args:
            info (dict): info received from the ns3 simulator
        Returns:
            bool: True if it is a control packet, False otherwise
        """
        # 兼容空/非字符串的 info（如初始握手返回 {}）：直接视为控制信息，跳过处理
        if not isinstance(info, str) or len(info) == 0:
            self.action_applied = False
            self.action_seq = None
            self.pkt_type = -1
            return True
        tokens = info.split(",")
        kv = {}
        for token in tokens:
            if '=' in token:
                k, v = token.split('=', 1)
                kv[k.strip()] = v.strip()
        self.action_applied = True
        self.action_seq = None
        if "action_applied" in kv:
            try:
                self.action_applied = bool(int(kv["action_applied"]))
            except Exception:
                self.action_applied = True
        if "action_seq" in kv:
            try:
                self.action_seq = int(kv["action_seq"])
            except Exception:
                self.action_seq = None
        def _kv_float(keys, default="0"):
            for key in keys:
                val = kv.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except Exception:
                        pass
            return float(default)

        def _kv_int(keys, default="-1"):
            for key in keys:
                val = kv.get(key)
                if val is not None:
                    try:
                        return int(float(val))
                    except Exception:
                        pass
            return int(float(default))

        self.delay_time = _kv_float(["delay_time", "End to End Delay"])
        ## retrieve packet info
        self.pkt_size = _kv_float(["pkt_size", "Packet Size"])
        Agent.curr_time = _kv_float(["curr_time", "Current sim time"])
        self.pkt_id = _kv_int(["pkt_id", "Pkt ID"])
        pkt_type_str = kv.get("pkt_type", kv.get("packetType"))
        if pkt_type_str is None and len(tokens) > 4:
            pkt_type_str = tokens[4].split('=')[-1]
        try:
            pkt_type = int(pkt_type_str)
        except Exception:
            pkt_type = 2
        self.pkt_type = pkt_type
        self.signaling = pkt_type != 0
        # [Design B 2026-05-14] Warm-start IL: 从 info 解析 ECMP target action
        try:
            self.ecmp_action = int(kv.get("ecmp_action", -1))
        except Exception:
            self.ecmp_action = -1
        if(pkt_type==0): # data packet
            # treat lost packets
            lost_raw = kv.get("lost_packets_id", kv.get("Packet Lost"))
            if lost_raw is None and len(tokens) > 18:
                lost_raw = tokens[18].split('=')[-1]
            lost_packets_id = [v for v in (lost_raw or "").split(';') if v]
            for lost_packet_id in lost_packets_id: 
                # lost_packet_info = Agent.temp_obs.get(int(lost_packet_id)) 
                # if(lost_packet_info==None): 
                #     print("error") 
                #     continue
                rec = Agent.temp_obs.get(int(lost_packet_id))
                if rec is None:          # 包已过期或重复通知，直接跳过
                    continue
                # 用完即删，幂等；即使多线程重复到达也不会二次处理
                Agent.temp_obs.pop(int(lost_packet_id), None) 
                #if(int(lost_packet_time)!= int(lost_packet_info["time"]*1000)): 
                #    continue 
                next_hop_degree = int(rec.get("next_hop_degree", 1))
                try:
                    if Agent.agents[rec["node"]]:
                        Agent.agents[rec["node"]].update_num_taken_actions(rec["action"], next_hop_degree)
                except Exception:
                    pass
                rew = self._get_reward_lost_pkt()
                #obs_shape = next_hop_degree * getattr(Agent, 'K', 1)
                obs_dim = getattr(self, "obs_dim", int(self.env.observation_space.shape[0]))
                next_obs = np.zeros(obs_dim, dtype=float)
                try:
                    next_obs[0] = float(rec["obs"][0])
                except Exception:
                    pass
                buf_idx = int(rec.get("node", self.index))
                ## Add the lost packet to the replay buffer
                if Agent.loss_penalty_type == "fixed":
                    if(Agent.prioritizedReplayBuffer):
                        # Agent.replay_buffer[self.index].add(np.array(rec["obs"], dtype=float).squeeze(),
                        #             rec["action"], 
                        #             rew,
                        #             np.array([rec["obs"][0]] + [0]*(obs_shape), dtype=float).squeeze(),
                        Agent.replay_buffer[buf_idx].add(np.array(rec["obs"], dtype=float).squeeze(),
                                    int(rec["action"]),
                                    float(rew),
                                    next_obs.squeeze(), 
                                    True,
                                    #Agent.replay_buffer[self.index].latest_gradient_step[rec["action"]])
                                    Agent.replay_buffer[buf_idx].latest_gradient_step[int(rec["action"])])
                    else:
                        if(self.train):
                            # Agent.replay_buffer[self.index].add(np.array(rec["obs"], dtype=float).squeeze(),
                            #             rec["action"], 
                            #             rew,
                            #             np.array([rec["obs"][0]] + [0]*(obs_shape), dtype=float).squeeze(),
                            Agent.replay_buffer[buf_idx].add(np.array(rec["obs"], dtype=float).squeeze(),
                                        int(rec["action"]),
                                        float(rew),
                                        next_obs.squeeze(), 
                                        True)
                ## Increment the loss counter
                Agent.node_lost_pkts += 1
                ## Remove the lost packet from the pkt tracking dict
                Agent.pkt_tracking_dict.pop(int(lost_packet_id), None)
        else:
            self.action_applied = False
            return True
        
        ## update stats in static variables
        Agent.sim_avg_e2e_delay = _kv_float(["avg_e2e_delay", "Avg End to End Delay"])
        Agent.sim_cost = _kv_float(["cost", "Avg Cost"])
        Agent.sim_global_avg_e2e_delay = _kv_float(["global_avg_e2e_delay", "Avg Underlay End to End Delay"])
        Agent.sim_global_cost = _kv_float(["global_cost", "Avg Underlay Cost"])
        Agent.sim_dropped_packets = _kv_float(["dropped", "Packets dropped"])
        Agent.sim_delivered_packets = _kv_float(["delivered", "Packets delivered"])
        Agent.sim_injected_packets = _kv_float(["injected", "Packets injected"])
        Agent.sim_buffered_packets = _kv_float(["buffered", "Packets Buffered"])
        prev_dropped = _kv_float(["global_dropped", "Packets dropped Underlay"])
        prev_delivered = _kv_float(["global_delivered", "Packets delivered Underlay"])
        prev_injected = _kv_float(["global_injected", "Packets injected Underlay"])
        prev_buffered = _kv_float(["global_buffered", "Packets Buffered Underlay"])
        Agent.sim_global_dropped_packets = prev_dropped + Agent.sim_dropped_packets
        Agent.sim_global_delivered_packets = prev_delivered + Agent.sim_delivered_packets
        Agent.sim_global_injected_packets = prev_injected + Agent.sim_injected_packets
        Agent.sim_global_buffered_packets = prev_buffered + Agent.sim_buffered_packets
        Agent.sim_signaling_overhead = _kv_float(["signaling_overhead", "Signaling overhead"])
        if Agent.sim_global_delivered_packets > 0:
            Agent.sim_global_avg_e2e_delay = (
                (Agent.sim_global_avg_e2e_delay * prev_delivered) + (Agent.sim_avg_e2e_delay * Agent.sim_delivered_packets)
            ) / (Agent.sim_global_delivered_packets)
        if Agent.sim_global_delivered_packets + Agent.sim_global_dropped_packets > 0:
            prev_total = prev_dropped + prev_delivered
            curr_total = Agent.sim_dropped_packets + Agent.sim_delivered_packets
            Agent.sim_global_cost = (
                (Agent.sim_global_cost * prev_total) + (Agent.sim_cost * curr_total)
            ) / (Agent.sim_global_dropped_packets + Agent.sim_global_delivered_packets)
        return False

    def run(self):
        """ 
        Run an episode simulation
        """
        while True:
            obs = self.env.reset()
            self.transition_number = 0
                
            while True:
                if(not self.env.connected):
                    break
                prev_obs = obs
                # The ECMP label belongs to prev_obs. treat_info() below parses
                # the label for the newly returned observation, so capture it
                # before env.step() advances the protocol.
                ecmp_action_for_step = int(getattr(self, "ecmp_action", -1))
                pkt_id_for_step = int(getattr(self, "pkt_id", -1))
                signaling_for_step = bool(getattr(self, "signaling", False))
                obs, r_env, done_flag, info = self.step(prev_obs)
                action_for_step = int(self.action)
                is_ctrl = self.treat_info(info)
                action_applied = getattr(self, "action_applied", True)
                valid_step = is_valid_environment_step(
                    self.env.connected,
                    is_ctrl,
                    action_applied,
                )
                pkt_done = bool(done_flag)
                will_reach_max = (
                    Agent.max_nb_arrived_pkts > 0
                    and (Agent.total_arrived_pkts + (1 if pkt_done else 0)) >= Agent.max_nb_arrived_pkts
                )
                episode_done = (not self.env.connected) or will_reach_max
                # debug print suppressed; keep logs minimal to speed up

                ## check if episode is done_flag
                # 提前 break 会丢最后一条 transition，统一在后续 done 分支处理

                ## Increment the simulation and episode counters
                self.transition_number += 1
                Agent.total_nb_iterations += 1

                # —— 记录 reward（steps/time），不依赖 pkt_type ——
                if valid_step and getattr(Agent, "reward_log_every", 0) > 0 and (Agent.total_nb_iterations % Agent.reward_log_every == 0):
                    try:
                        r_val = float(r_env)
                        # 更新本地/全局 EWMA
                        self.r_ewma = (1.0 - self.r_alpha) * self.r_ewma + self.r_alpha * r_val
                        Agent.global_reward_ewma = (
                            (1.0 - Agent.reward_ewma_alpha) * getattr(Agent, "global_reward_ewma", 0.0)
                            + Agent.reward_ewma_alpha * r_val
                        )
                        step_idx = Agent.total_nb_iterations
                        t_us = int((Agent.base_curr_time + Agent.curr_time) * 1e6)
                        with self.tb_writer_dict["reward_steps"].as_default():
                            tf.summary.scalar("reward_over_steps", r_val, step=step_idx)
                            tf.summary.scalar("reward_ewma_over_steps", self.r_ewma, step=step_idx)
                        with self.tb_writer_dict["reward_time"].as_default():
                            tf.summary.scalar("reward_over_time", r_val, step=t_us)
                            tf.summary.scalar("reward_ewma_over_time", self.r_ewma, step=t_us)
                        if getattr(Agent, "reward_global_by_steps_writer", None) is not None:
                            with Agent.reward_global_by_steps_writer.as_default():
                                tf.summary.scalar("global_reward_ewma_over_steps", Agent.global_reward_ewma, step=step_idx)
                        if getattr(Agent, "reward_global_by_time_writer", None) is not None:
                            with Agent.reward_global_by_time_writer.as_default():
                                tf.summary.scalar("global_reward_ewma_over_time", Agent.global_reward_ewma, step=t_us)
                    except Exception:
                        pass

                if Agent.signaling_type in ("NN", "target") and Agent.signalingSim == 0:
                    self._get_upcoming_events()

                # env.step(a_t) sends a_t and then receives (s_{t+1}, r_{t+1}).
                # Store that transition immediately.  The former pending path
                # delayed it by one extra env.step and paired rewards/states
                # with the wrong action.
                wrote_transition = False
                decision_state = not (
                    prev_obs[0] == self.index or prev_obs[0] in (-1, 1000)
                )
                if valid_step:
                    action_seq = getattr(self, "action_seq", None)
                    if action_seq is None:
                        self.action_sequence_mismatches += 1
                    else:
                        action_seq = int(action_seq)
                        if self.first_action_seq is None:
                            self.first_action_seq = action_seq
                        if self.last_action_seq is not None and action_seq != self.last_action_seq + 1:
                            self.action_sequence_mismatches += 1
                        self.last_action_seq = action_seq

                if valid_step and decision_state:
                    transition = build_aligned_transition(
                        prev_obs,
                        action_for_step,
                        r_env,
                        obs,
                        done_flag,
                        ecmp_action=ecmp_action_for_step,
                        num_actions=int(self.env.action_space.n),
                    )
                    self._record_action_reward(
                        transition.action,
                        transition.reward,
                    )
                    if Agent.signaling_type == "ideal":
                        if Agent.prioritizedReplayBuffer:
                            Agent.replay_buffer[self.index].add(
                                transition.observation,
                                transition.action,
                                transition.reward,
                                transition.next_observation,
                                transition.done,
                                Agent.replay_buffer[self.index].latest_gradient_step[transition.action],
                            )
                        else:
                            Agent.replay_buffer[self.index].add(*transition)
                        wrote_transition = True
                    elif Agent.signaling_type == "NN":
                        self._push_upcoming_event(self.index, {
                            "time": Agent.curr_time + self.small_signaling_delay,
                            "obs": transition.observation,
                            "action": transition.action,
                            "reward": transition.reward,
                            "new_obs": transition.next_observation,
                            "flag": transition.done,
                            "pkt_id": getattr(self, "pkt_id", -1),
                        })
                        if Agent.signalingSim == 0 and self.train:
                            Agent.small_signaling_overhead_counter += self.small_signaling_pkt_size
                            Agent.small_signaling_pkt_counter += 1
                        wrote_transition = True
                    elif Agent.signaling_type == "target":
                        pass
                    if wrote_transition:
                        self.transitions_written += 1

                # —— 调试打印：收到/写入的 reward 与 temp_obs 规模 ——
                if getattr(Agent, "debug_reward", False) and (
                    (Agent.total_nb_iterations % getattr(Agent, "reward_debug_every", 5000) == 0) or (float(r_env) != 0.0)
                ):
                    try:
                        buffer_size = Agent.replay_buffer[self.index].__len__() if hasattr(Agent.replay_buffer[self.index], '__len__') else -1
                        print(f"[DBG][node {self.index}] step={Agent.total_nb_iterations} r_env={float(r_env):.6f} wrote_transition={wrote_transition} temp_obs_size={len(Agent.temp_obs)} eps={self.update_eps:.4f} buf_size={buffer_size}")
                    except Exception:
                        pass
                if getattr(Agent, "action_seq_log_every", 10000) > 0 and (
                    Agent.total_nb_iterations % getattr(Agent, "action_seq_log_every", 10000) == 0
                ):
                    try:
                        pkt_type = getattr(self, "pkt_type", -1)
                        action_seq = getattr(self, "action_seq", None)
                        print(
                            f"[SEQ][node {self.index}] step={Agent.total_nb_iterations} "
                            f"action_seq={action_seq} action_applied={action_applied} "
                            f"pkt_id={getattr(self, 'pkt_id', -1)} pkt_type={pkt_type} "
                            f"valid_step={valid_step} wrote_transition={wrote_transition}"
                        )
                    except Exception:
                        pass

                if not valid_step:
                    if episode_done:
                        if will_reach_max:
                            print("Done by max number of arrived pkts")
                        break
                    continue # if it is a control/no-op step, continue

                Agent.nb_transitions += 1
                is_new_pkt = self.pkt_id not in Agent.pkt_tracking_dict
                if is_new_pkt: ## check if the packet is a new arrival
                    self.handle_new_packet(obs)
                else: ## if the packet is not new in the network
                    self.handle_transit_packet(obs, pkt_done, r_env)

                if pkt_done and (self.pkt_id in Agent.pkt_tracking_dict): ## if the packet arrived to destination
                    self.handle_done()

                if episode_done:
                    if will_reach_max:
                        print("Done by max number of arrived pkts")
                    break

                if valid_step and (not pkt_done) and not (
                    prev_obs[0] == self.index or prev_obs[0] in (-1, 1000) or signaling_for_step
                ):
                    try:
                        nh_deg = len(list(Agent.G.neighbors(self.neighbors[self.action]))) if Agent.G is not None else 1
                    except Exception:
                        nh_deg = 1
                    track = Agent.pkt_tracking_dict.get(pkt_id_for_step, {"src": self.index, "dst": int(prev_obs[0])})
                    Agent.temp_obs[pkt_id_for_step] = {
                        "node": self.index,
                        "obs": np.array(prev_obs, dtype=float).squeeze(),
                        "action": int(self.action),
                        "time": Agent.curr_time,
                        "src": track.get("src", self.index),
                        "dst": track.get("dst", int(prev_obs[0])),
                        "next_hop_degree": int(nh_deg),
                    }
            break
        action_total = int(np.sum(self.policy_action_counts))
        action_ratios = (
            (self.policy_action_counts / action_total).tolist()
            if action_total > 0
            else np.zeros_like(self.policy_action_counts, dtype=float).tolist()
        )
        print(
            f"[POLICY-ACTIONS] node={self.index} total={action_total} "
            f"counts={self.policy_action_counts.tolist()} ratios={action_ratios} "
            f"shadow_greedy_counts={self.greedy_action_counts.tolist()} "
            f"random={self.random_action_decisions} greedy={self.greedy_action_decisions} "
            f"eval_epsilon={float(getattr(Agent, 'eval_epsilon', 0.0))} train={int(bool(self.train))}"
        )
        ## close the zmq bridge
        self.env.ns3ZmqBridge.send_close_command()
        return True
    
    def handle_new_packet(self, obs):
        """ Handle a new packet arrival and add it to the tracking dict
        Args:
            obs (list): observation from the environment
        """
        self.count_new_pkts += 1
        Agent.total_new_rcv_pkts += 1
        Agent.total_data_size += self.pkt_size
        ## add to tracked pkts
        Agent.pkt_tracking_dict[int(self.pkt_id)]= {"src": self.index,
                                                    "node": self.index,
                                                    "dst": int(obs[0]),
                                                    "hops": [self.index],
                                                    "delays_ideal": [],
                                                    "delays_real": [],
                                                    "start_time": Agent.curr_time,
                                                    "tag": None}

    def handle_transit_packet(self, obs, pkt_done, r_env):
        """ Handle a transit packet (not new). 
        Args:
            obs (list): observation from the environment
            pkt_done (bool): if the packet arrived to destination
        """
        states_info = Agent.temp_obs.pop(self.pkt_id, None)
        if states_info is None:
            # print(f"[WARN] transit without matching push, pkt_id={self.pkt_id}")
            return

        hop_time_real =  Agent.curr_time - states_info["time"]
        # 使用扩展观测的 cost_bytes（邻居块的第一个特征）估计理想跳时延
        try:
            K = getattr(Agent, 'K', 1)
            base = 1 + states_info["action"] * K
            cost_bytes = states_info["obs"][base]
        except Exception:
            cost_bytes = states_info["obs"][states_info["action"] + 1]
        hop_time_ideal = ((cost_bytes + 512 ) * 8 / Agent.link_cap) + (Agent.link_delay*0.001)
        Agent.total_rewards_with_loss += hop_time_real
        ## add to tracked pkts
        Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"].append(self.index)                        
        Agent.pkt_tracking_dict[int(self.pkt_id)]["node"] = self.index
        
        ## saving state info
        states_info["hop_time_real"] = hop_time_real
        states_info["hop_time_ideal"] = hop_time_ideal
        Agent.rewards.append(hop_time_real)
        # [OOM Fix] 限制全局 rewards 列表长度，只保留最近 20000 条
        if len(Agent.rewards) > 20000:
            Agent.rewards = Agent.rewards[-20000:]
        Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_ideal"].append(hop_time_ideal)
        Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"].append(hop_time_real)                  
        # 样本入库改由本地 pending 闭环在 run() 中完成；此处不再跨节点写样本 这里的reward计算已经被废弃
        # 保留上面统计/追踪信息更新，用于 hops/delays 统计和丢包逻辑

        # 不在此处写 TensorBoard；统一在 run() 中写，避免重复/漏写

    def handle_done(self):
        """ Handle the case when a packet arrives at the destination
        """
        self.count_arrived_packets += 1
        Agent.total_arrived_pkts += 1
        # 直接用逐跳累计的真实延迟作为 e2e，避免依赖 C++ info 里的占位值
        hops =  len(Agent.pkt_tracking_dict[int(self.pkt_id)]["hops"]) - 1
        Agent.total_hops += hops
        e2e_real = sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"])
        self.delay_time = e2e_real
        Agent.total_e2e_delay += e2e_real
        Agent.delays_ideal.append(sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_ideal"]))
        Agent.delays_real.append(sum(Agent.pkt_tracking_dict[int(self.pkt_id)]["delays_real"]))
        Agent.delays.append(self.delay_time)
        Agent.nb_hops.append(hops)
        # [OOM Fix] 限制全局统计列表长度，防止长时仿真撑爆内存
        limit_len = 20000
        if len(Agent.delays) > limit_len:
            Agent.delays = Agent.delays[-limit_len:]
            Agent.nb_hops = Agent.nb_hops[-limit_len:]
            Agent.delays_ideal = Agent.delays_ideal[-limit_len:]
            Agent.delays_real = Agent.delays_real[-limit_len:]
        if(len(Agent.nb_hops)>50):
            Agent.nb_hops = Agent.nb_hops[-50:]
        if(len(Agent.delays)>50):
            Agent.delays = Agent.delays[-50:]
        Agent.pkt_tracking_dict.pop(int(self.pkt_id))
        

    def _push_upcoming_event(self, node, info):
        """Put the info in the upcoming event queue

        Args:
            node (int): node index
            info (dict): info to store as event
        """
        Agent.upcoming_events[node].append(info)
        Agent.upcoming_events[node].sort(key=operator.itemgetter("time"))
        
    def _get_upcoming_events(self):
        """
        Go through the upcoming event list and check whether of signaling pkt arrived, then add it to replay buffer if small signaling or update target if big signaling
        """
        if Agent.signaling_type == "NN":
            while len(Agent.upcoming_events[self.index]) > 0:
                if Agent.upcoming_events[self.index][0]["time"]> Agent.curr_time:
                    break
                element = Agent.upcoming_events[self.index].pop(0)
                if "obs" in element.keys():
                    ## treat small signaling 
                    Agent.replay_buffer[self.index].add(element["obs"],
                                                    element["action"], 
                                                    element["reward"],
                                                    element["new_obs"], 
                                                    element["flag"])
                    #print(self.index, Agent.replay_buffer[self.index].sample(1))
                else:
                    ## treat big signaling
                    # print("receive sync event to %s to %s at %s" % (self.index, element["neighbor_idx"], element["time"]))
                    self._sync_current(element["neighbor_idx"])
                    
        elif Agent.signaling_type == "target":
            while len(Agent.upcoming_events[self.index]) > 0:
                if Agent.upcoming_events[self.index][0]["time"]> Agent.curr_time:
                    break
                element = Agent.upcoming_events[self.index].pop(0)
                Agent.replay_buffer[self.index].add(element["obs"],
                                                element["action"], 
                                                element["target"],
                                                element["new_obs"], 
                                                element["flag"])

    def _get_upcoming_events_real(self, signaling_pkt_id=None, ):
        """
        Go through the upcoming event list and check whether of signaling pkt arrived, then add it to replay buffer if small signaling or update target if big signaling
        """
        for idx, element in enumerate(Agent.upcoming_events[self.index]):
            if Agent.signaling_type == "NN":
                if element["pkt_id"] == signaling_pkt_id:
                    Agent.replay_buffer[self.index].add(element["obs"],
                                                        element["action"], 
                                                        element["reward"],
                                                        element["new_obs"], 
                                                        element["flag"])
                    Agent.upcoming_events[self.index].pop(idx)
                    break
                        
            elif Agent.signaling_type == "target":
                if element["pkt_id"] == signaling_pkt_id:
                    
                    if Agent.prioritizedReplayBuffer:
                        Agent.replay_buffer[self.index].add(element["obs"],
                                                            element["action"], 
                                                            element["target"],
                                                            element["new_obs"], 
                                                            element["flag"],
                                                            element["gradient_step"])
                        if element["gradient_step"] > Agent.replay_buffer[self.index].latest_gradient_step[element["action"]]:
                            Agent.replay_buffer[self.index].latest_gradient_step[element["action"]] = element["gradient_step"] 
                            Agent.replay_buffer[self.index].update_priorities(Agent.replay_buffer[self.index].neighbors_idx[element["action"]],
                                                                            element["action"])
                    else:
                        Agent.replay_buffer[self.index].add(element["obs"],
                                                            element["action"],
                                                            element["target"],
                                                            element["new_obs"],
                                                            element["flag"])
                    Agent.upcoming_events[self.index].pop(idx)
                    break
