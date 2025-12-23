### imports
import tensorflow as tf
import networkx as nx
import numpy as np
from ns3_model import ns3env
from source.learner import DQN_AGENT
from source.utils import load_model, LinearSchedule, optimal_routing_decision
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

        # 首拍默认无控制包，pkt_id 置为 -1，防止首拍未初始化访问
        self.signaling = False
        self.pkt_id = -1
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
                # 计算每邻居特征维度 K（(obs_dim - 1) / num_actions）
                try:
                    K = int((self.env.observation_space.shape[0] - 1) // self.env.action_space.n)
                except Exception:
                    K = self.env.action_space.n  # 兜底（不应触发）
                if not hasattr(Agent, 'K'):
                    Agent.K = K
                Agent.agents[self.index] = DQN_AGENT(
                    q_func=model,
                    observation_shape=self.env.observation_space.shape,
                    num_actions=self.env.action_space.n,
                    num_nodes=Agent.numNodes,
                    input_size_splits = [1,
                                        self.env.action_space.n * Agent.K,
                                        ],
                    lr=Agent.lr,
                    gamma=Agent.gamma,
                    neighbors_degrees=[len(list(Agent.G.neighbors(x))) for x in self.neighbors],
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
                loaded_models = load_model(Agent.load_path, self.index)
                if loaded_models is not None:
                    print("Restoring from {} for node {}".format(Agent.load_path, self.index))
                    Agent.agents[self.index].q_network.set_weights(loaded_models[self.index].get_weights())
        
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
        
        # 初始化每节点的 pending 容器（用于本地闭环）
        if not hasattr(Agent, "pending"):
            Agent.pending = {}

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
        # Create the schedule for exploration.
        self.exploration = LinearSchedule(schedule_timesteps=int(Agent.iterationNum),
                                    initial_p=Agent.exploration_initial_eps,
                                    final_p=Agent.exploration_final_eps)

  

    def step(self, obs):
        """
        Do an env step
        
        """
        ## schedule the exploration
        if self.train:
            self.update_eps = tf.constant(self.exploration.value(self.transition_number))
            ## log the exploration value
            with self.tb_writer_dict["exploration"].as_default():
                tf.summary.scalar('exploaration_value_over_steps', self.update_eps, step=self.transition_number)
                tf.summary.scalar('exploaration_value_over_time', self.update_eps, step=int((Agent.base_curr_time + Agent.curr_time)*1e6))

        ## take the action
        if obs[0] == self.index or self.transition_number < 1 or self.signaling == True or obs[0] in(-1, 1000): # pkt arrived to dst or it is a train step, ignore the action
            self.action = 0
        else:
            self.action = self.take_action(obs)
            # 防止 temp_obs 内存爆炸：定期清理最旧的条目
            if len(Agent.temp_obs) > 10000:  # 超过10000个条目时清理
                # 清理最旧的50%条目
                old_keys = sorted(Agent.temp_obs.keys())[:5000]
                for key in old_keys:
                    Agent.temp_obs.pop(key, None)

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
            
            Agent.temp_obs[int(self.pkt_id)]= {"node": self.index,
                                               "obs": obs,
                                               "action": self.action,
                                               "time": Agent.curr_time,
                                               "src" :Agent.pkt_tracking_dict[int(self.pkt_id)]["src"],
                                               "dst" :Agent.pkt_tracking_dict[int(self.pkt_id)]["dst"],
                                               }
        ### Apply the action
        return self.env.step(self.action)
    
    def _get_reward_lost_pkt(self):
        """ Compute the reward when the packet is lost
        """
        if Agent.loss_penalty_type == "fixed":
            return Agent.loss_penalty
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
            action = Agent.agents[self.index].step(np.array([obs]),
                                                   self.train,
                                                   self.update_eps,
                                                   actions_probs=actions_probs).numpy().item()
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
            return True
        tokens = info.split(",")
        self.delay_time = float(tokens[0].split('=')[-1])
        ## retrieve packet info
        self.pkt_size = float(tokens[1].split('=')[-1])
        Agent.curr_time = float(tokens[2].split('=')[-1])
        self.pkt_id = int(tokens[3].split('=')[-1])
        pkt_type = int(tokens[4].split('=')[-1]) 
        self.signaling = pkt_type != 0 
        if(pkt_type==0): # data packet
            # treat lost packets
            lost_packets_id = tokens[18].split('=')[-1].split(';')[:-1] 
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
                try:
                    if Agent.agents[rec["node"]]:
                        next_hop_degree = len(list(Agent.G.neighbors(self.neighbors[rec["action"]])))
                        Agent.agents[rec["node"]].update_num_taken_actions(rec["action"], next_hop_degree)
                except Exception:
                    next_hop_degree = len(list(Agent.G.neighbors(self.index))) if Agent.G is not None else 1
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
            if(pkt_type==2): # small signaling packet
                id_signaled = int(tokens[18].split('=')[-1]) 
                self._get_upcoming_events_real(id_signaled) 
                Agent.small_signaling_overhead_counter += self.pkt_size 
                Agent.small_signaling_pkt_counter += 1 
            if(pkt_type==1): # big signaling packet
                NNIndex = int(tokens[18].split('=')[-1]) 
                segIndex= int(tokens[19].split('=')[-1]) 
                NodeIdSignaled = int(tokens[20].split('=')[-1]) 
                if segIndex > Agent.nn_max_seg_index:
                    raise("segIndex > {}".format(Agent.nn_max_seg_index))
                if segIndex == Agent.nn_max_seg_index: ## NN signaling complete
                    # print(f"sync {self.index} with neighbor {self.neighbors.index(NodeIdSignaled)} at time {Agent.curr_time} {self.sync_counter} {NNIndex}")
                    if NNIndex ==Agent.sync_counters[self.index] - 1:
                        self._sync_current(self.neighbors.index(NodeIdSignaled), with_temp=True)
                    else:
                        #print(self.index, NodeIdSignaled)
                        self._sync_current(self.neighbors.index(NodeIdSignaled))
                #print("here") 
                Agent.big_signaling_overhead_counter += self.pkt_size 
                Agent.big_signaling_pkt_counter += 1

            return True
            #continue             
        
        ## update stats in static variables
        Agent.sim_avg_e2e_delay =  float(tokens[5].split('=')[-1])  
        Agent.sim_cost = float(tokens[6].split('=')[-1]) 
        Agent.sim_global_avg_e2e_delay = float(tokens[7].split('=')[-1])  
        Agent.sim_global_cost = float(tokens[8].split('=')[-1])  
        Agent.sim_dropped_packets = float(tokens[9].split('=')[-1]) 
        Agent.sim_delivered_packets = float(tokens[10].split('=')[-1]) 
        Agent.sim_injected_packets = float(tokens[11].split('=')[-1]) 
        Agent.sim_buffered_packets = float(tokens[12].split('=')[-1]) 
        Agent.sim_global_dropped_packets = float(tokens[13].split("=")[-1]) + Agent.sim_dropped_packets 
        Agent.sim_global_delivered_packets = float(tokens[14].split("=")[-1]) + Agent.sim_delivered_packets 
        Agent.sim_global_injected_packets = float(tokens[15].split("=")[-1]) + Agent.sim_injected_packets 
        Agent.sim_global_buffered_packets = float(tokens[16].split('=')[-1]) + Agent.sim_buffered_packets 
        Agent.sim_signaling_overhead = float(tokens[17].split('=')[-1])
        if Agent.sim_global_delivered_packets > 0:
            prev_delivered = float(tokens[13].split("=")[-1])
            Agent.sim_global_avg_e2e_delay = (
                (Agent.sim_global_avg_e2e_delay * prev_delivered) + (Agent.sim_avg_e2e_delay * Agent.sim_delivered_packets)
            ) / (Agent.sim_global_delivered_packets)
        if Agent.sim_global_delivered_packets + Agent.sim_global_dropped_packets > 0:
            prev_dropped = float(tokens[13].split("=")[-1])
            prev_delivered = float(tokens[14].split("=")[-1])
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
                obs, r_env, done_flag, info = self.step(prev_obs)
                is_ctrl = self.treat_info(info)
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
                if getattr(Agent, "reward_log_every", 0) > 0 and (Agent.total_nb_iterations % Agent.reward_log_every == 0):
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

                # —— 调试打印：收到/写入的 reward 与 temp_obs 规模 ——
                # —— 本地闭环：先用当前 (r_env, obs) 关闭上一拍 (s_{t-1}, a_{t-1}) ——
                had_pending = (self.index in Agent.pending)
                pend = Agent.pending.pop(self.index, None)
                wrote_transition = (pend is not None)
                if pend is not None:
                    if Agent.signaling_type == "ideal":
                        Agent.replay_buffer[self.index].add(
                            pend["obs"],
                            pend["action"],
                            float(r_env),
                            np.array(obs, dtype=float).squeeze(),
                            #done_flag,
                            episode_done,
                        )
                    elif Agent.signaling_type == "NN":
                        self._push_upcoming_event(self.index, {
                            "time": Agent.curr_time + self.small_signaling_delay,
                            "obs": pend["obs"],
                            "action": pend["action"],
                            "reward": float(r_env),
                            "new_obs": np.array(obs, dtype=float).squeeze(),
                            #"flag": done_flag,
                            "flag": episode_done,
                            "pkt_id": getattr(self, "pkt_id", -1),
                        })
                        if Agent.signalingSim == 0 and self.train:
                            Agent.small_signaling_overhead_counter += self.small_signaling_pkt_size
                            Agent.small_signaling_pkt_counter += 1
                    elif Agent.signaling_type == "target":
                        # 若仍使用 target 模式，可在此按需计算 target；保持最小入侵暂不变
                        pass

                    try:
                        Agent.agents[self.index].update_num_taken_actions(
                            pend["action"], pend.get("next_hop_degree", 1)
                        )
                    except Exception:
                        pass

                # —— 调试打印：收到/写入的 reward 与 temp_obs 规模 ——
                if getattr(Agent, "debug_reward", False) and (
                    (Agent.total_nb_iterations % getattr(Agent, "reward_debug_every", 5000) == 0) or (float(r_env) != 0.0)
                ):
                    try:
                        buffer_size = Agent.replay_buffer[self.index].__len__() if hasattr(Agent.replay_buffer[self.index], '__len__') else -1
                        print(f"[DBG][node {self.index}] step={Agent.total_nb_iterations} r_env={float(r_env):.6f} had_pending={had_pending} wrote_transition={wrote_transition} temp_obs_size={len(Agent.temp_obs)} eps={self.update_eps:.4f} buf_size={buffer_size}")
                    except Exception:
                        pass

                # —— 登记本拍 (s_t=prev_obs, a_t=self.action)，待下一拍 r_{t+1} 来闭环 ——
                if (not is_ctrl) and not (getattr(self, "signaling", False) or prev_obs[0] == self.index or prev_obs[0] in (-1, 1000)):
                    try:
                        next_hop_degree = len(list(Agent.G.neighbors(self.neighbors[self.action]))) if Agent.G is not None else 1
                    except Exception:
                        next_hop_degree = len(list(Agent.G.neighbors(self.index))) if Agent.G is not None else 1
                    Agent.pending[self.index] = {
                        "obs": np.array(prev_obs, dtype=float).squeeze(),
                        "action": int(self.action),
                        "next_hop_degree": next_hop_degree,
                    }

                ## Treat the info from the env（控制包也完成了闭环与登记）
                #if self.treat_info(info):
                if Agent.signaling_type in ("NN", "target") and Agent.signalingSim == 0:
                    self._get_upcoming_events()
                if is_ctrl:
                    if episode_done:
                        # —— 兜底：episode 结束，把最后一个 pending 关掉（terminal）——
                        final_pend = Agent.pending.pop(self.index, None)
                        if final_pend is not None:
                            try:
                                Agent.replay_buffer[self.index].add(
                                    final_pend["obs"],
                                    final_pend["action"],
                                    0.0,
                                    np.array(obs, dtype=float).squeeze(),
                                    True
                                )
                                if getattr(Agent, "debug_reward", False):
                                    print(f"[DBG][node {self.index}] terminal-close pending: action={final_pend['action']} r=0.0")
                            except Exception:
                                pass
                        if will_reach_max:
                            print("Done by max number of arrived pkts")
                        break
                    continue # if it is a control packet, continue
                
                Agent.nb_transitions += 1
                if self.pkt_id not in Agent.pkt_tracking_dict.keys(): ## check if the packet is a new arrival
                    self.handle_new_packet(obs)
                    
                else: ## if the packet is not new in the network
                    self.handle_transit_packet(obs, pkt_done, r_env)

                    if pkt_done: ## if the packet arrived to destination
                        self.handle_done()

                    if episode_done:
                        # —— 兜底：episode 结束，把最后一个 pending 关掉（terminal）——
                        final_pend = Agent.pending.pop(self.index, None)
                        if final_pend is not None:
                            try:
                                Agent.replay_buffer[self.index].add(
                                    final_pend["obs"],
                                    final_pend["action"],
                                    0.0,
                                    np.array(obs, dtype=float).squeeze(),
                                    True
                                )
                                if getattr(Agent, "debug_reward", False):
                                    print(f"[DBG][node {self.index}] terminal-close pending: action={final_pend['action']} r=0.0")
                            except Exception:
                                pass

                        ## check if the episode is done by max number of arrived pkts
                        #if Agent.max_nb_arrived_pkts > 0 and Agent.max_nb_arrived_pkts <= Agent.total_arrived_pkts:
                        if will_reach_max:
                            print("Done by max number of arrived pkts")
                        break
            break
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
