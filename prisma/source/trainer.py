### imports
from source.agent import Agent
import tensorflow as tf
import numpy as np
from source.utils import convert_bps_to_data_rate
import copy 
import time

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

class Trainer(Agent):
    """ Trainer class for the RL agent
    """
    
    def __init__(self, index, agent_type="dqn", train=True):
        Agent.__init__(self, index, agent_type, train)
        self.reset()
        ## define the log file for td error 
        self.tb_writer_dict = {"td_error": tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/td_error/node_{self.index}'),
                               "replay_buffer_length": tf.summary.create_file_writer(logdir=f'{Agent.logs_folder}/replay_buffer_length/node_{self.index}')}
    def reset(self):
        self.last_training_time = 0
        self.last_sync_time = 0
        self.gradient_step_idx = 1
    def run(self):
        """
            Start the trainer deamon
        """
        import time
        while True :
            time.sleep(np.random.uniform(0.1, 1.5))
            ## check if there are signaling pkts arrived if signaling type NN
            if Agent.signaling_type in ("NN", "target") and Agent.signalingSim == 0:
                self._get_upcoming_events()
            ## check if it is time to syncronize nn
            self._check_sync()
                
            ## check if it is time to train
            if Agent.curr_time > (self.last_training_time + Agent.training_step) and Agent.replay_buffer[self.index].total_samples>= Agent.batch_size:
                self.step()

    def step(self):
        """
        Do a training step
        
        【修复说明】
        针对MSE Loss爆炸问题，本版本实现了以下关键修复：
        1. 使用 Huber Loss 替代 MSE Loss（对异常值更鲁棒）
        2. 更激进的 Q 值裁剪（上界从 20 降到 5）
        3. Reward Clipping（限制在 [-1, 1]）
        4. 梯度裁剪（防止梯度爆炸）
        5. TD Error 监控（记录 Huber Loss 和 MSE 双版本）
        """
        self.last_training_time = Agent.curr_time
        ## sample from the replay buffer
        obses_t, actions_t, rewards_t, next_obses_t, dones_t, weights = Agent.replay_buffer[self.index].sample(Agent.batch_size)
        if Agent.signaling_type == "target":
            targets_t = tf.constant(rewards_t, dtype=float)
            obses_t = tf.constant(obses_t)
            actions_t = tf.constant(actions_t)
        else:
            obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
            actions_t = tf.convert_to_tensor(actions_t, dtype=tf.int32)
            next_obses_t = tf.convert_to_tensor(next_obses_t, dtype=tf.float32)
            
            # 【修复1】Reward Clipping：限制 reward 在 [-1, 1] 范围
            # 这能有效防止极端 reward 值导致的 Q 值爆炸
            rewards_clipped = np.clip(rewards_t, -1.0, 1.0)
            
            # Compute TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            next_q_values = Agent.agents[self.index].target_q_network(next_obses_t)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            
            # 【修复2】更激进的 Q 值裁剪
            # 原始：q_upper_bound = 2.0 / (1.0 - gamma) ≈ 20 (gamma=0.9)
            # 修复：假设 reward ∈ [-1, 1]，Q 上界 ≈ 1/(1-gamma) ≈ 10
            # 实际使用更保守的值 5.0，因为跨节点状态污染会导致 Q 值估计不准
            q_upper_bound = 5.0  # 更保守的上界
            max_next_q = tf.clip_by_value(max_next_q, -q_upper_bound, q_upper_bound)
            
            # TD target（使用 clipped reward）
            targets_t = rewards_clipped + Agent.gamma * max_next_q * (1.0 - dones_t)
            
            # 【修复3】对 target 做最终裁剪
            targets_t = tf.clip_by_value(targets_t, -q_upper_bound, q_upper_bound)
            targets_t = tf.convert_to_tensor(targets_t, dtype=tf.float32)
        
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        ### Make a gradient step (使用 Huber Loss)
        td_errors = Agent.agents[self.index].train(obses_t, actions_t, targets_t, weights)
        
        ## log the td error and replay buffer length
        if len(td_errors):
            with self.tb_writer_dict["td_error"].as_default():
                # 记录 Huber Loss（实际训练使用的 loss）
                huber_loss_val = np.mean(np.abs(td_errors))  # Huber loss 的近似
                tf.summary.scalar('Huber_loss_over_steps', huber_loss_val, step=self.gradient_step_idx)
                tf.summary.scalar('Huber_loss_over_time', huber_loss_val, step=int((Agent.base_curr_time + Agent.curr_time)*1e6))
                # 同时记录 MSE Loss 用于对比诊断
                mse_loss_val = np.mean(td_errors**2)
                tf.summary.scalar('MSE_loss_over_steps', mse_loss_val, step=self.gradient_step_idx)
                tf.summary.scalar('MSE_loss_over_time', mse_loss_val, step=int((Agent.base_curr_time + Agent.curr_time)*1e6))
                # 记录 Q 值范围用于诊断
                tf.summary.scalar('target_mean', float(tf.reduce_mean(targets_t).numpy()), step=self.gradient_step_idx)
                tf.summary.scalar('target_max', float(tf.reduce_max(targets_t).numpy()), step=self.gradient_step_idx)
                tf.summary.scalar('target_min', float(tf.reduce_min(targets_t).numpy()), step=self.gradient_step_idx)
        with self.tb_writer_dict["replay_buffer_length"].as_default():
            tf.summary.scalar('replay_buffer_length_over_steps', len(Agent.replay_buffer[self.index]), step=self.gradient_step_idx)
            tf.summary.scalar('replay_buffer_length_over_time', len(Agent.replay_buffer[self.index]), step=int((Agent.base_curr_time + Agent.curr_time)*1e6))
        
        self.gradient_step_idx += 1
    
    def _check_sync(self):
        """
        Check the time to sync the NN depending on the signaling mode
        """
        ### Sync target NN
        if Agent.curr_time > ((Agent.sync_counters[self.index]+1)*Agent.sync_step):
                if "dqn" in self.agent_type and Agent.signaling_type != "target":
                    Agent.agents[self.index].update_target()
                self._sync_all(update_upcoming=True)
                Agent.sync_counters[self.index] += 1
                # print("sync all at %s" % Agent.curr_time, "for node:", self.index, "sync counter:", self.sync_counter)
                if Agent.signaling_type in ("ideal"):
                    self._sync_all(update_upcoming=False)
                self.last_sync_time = Agent.curr_time

    def _compute_sync_step(self, ratio=0.1):
        """
        Compute sync step to have control over data of ratio.
        """
        ## load traffic matrix and convert it to bps
        traff_mat = np.loadtxt(Agent.traffic_matrix_path, dtype=object)
        traff_mat = np.vectorize(convert_bps_to_data_rate)(traff_mat)
        
        ## data load per second
        data_load_per_s = np.sum(traff_mat)

        ## number of pkts per second
        nb_pkts_per_s = data_load_per_s/ (Agent.packet_size*8)

        ## small signaling load per s
        control_load_per_s = (nb_pkts_per_s * self.small_signaling_pkt_size)

        ## compute sync step
        sync_step = (self.nn_size) /((data_load_per_s * ratio)- control_load_per_s)
        print(f"Sync step computed automatically : {sync_step} seconds")
        return sync_step



    def _sync_upcoming(self, neighbor_num, neighbor_idx):
        """
        Sync this node neighbor upcoming target neural network with the neighbor nn

        Args:
            neighbor_num (int): neighbor number
            neighbor_idx (int): neighbor index for this node
        """
        # 仅当邻居是启用的、且已初始化的 agent 时才同步
        if neighbor_num in Agent.agents and Agent.agents[neighbor_num] is not None:
            Agent.agents[self.index].sync_neighbor_upcoming_target_q_network(Agent.agents[neighbor_num], neighbor_idx)

    def _sync_all(self, update_upcoming=False):
        """
        Sync this node all neighbors neural networks.
        If signaling type is target, sync the node target NN.
        
        Args:
            upcoming (bool): if True, update the upcoming nn with neighbor nn, else, update target with upcoming
        """
        if self.signaling_type == "target":
            Agent.agents[self.index].update_target()
        else:
            if update_upcoming:
                for indx, neighbor in enumerate(self.neighbors): 
                    # 仅当邻居是启用的、且已初始化的 agent 时才同步
                    if neighbor in Agent.agents and Agent.agents[neighbor] is not None:
                        self._sync_upcoming(neighbor, indx)
                        if Agent.signaling_type == "NN" and Agent.signalingSim == 0: ## programm the big signaling pkts
                            # print("program a sync from %s to %s  idx %s at %s, arrived at %s" % (self.index, neighbor, indx,  Agent.curr_time, Agent.curr_time + self.big_signaling_delay))
                            self._push_upcoming_event(self.index, {"time": Agent.curr_time + self.big_signaling_delay,
                                                             "neighbor_idx": indx})
                            Agent.big_signaling_overhead_counter += self.nn_size
                            Agent.big_signaling_pkt_counter += 1
                        
            else:
                for indx, neighbor in enumerate(self.neighbors): 
                    # 仅当邻居是启用的、且已初始化的 agent 时才同步
                    if neighbor in Agent.agents and Agent.agents[neighbor] is not None:
                        self._sync_current(indx)
