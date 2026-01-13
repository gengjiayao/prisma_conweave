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
        """
        self.last_training_time = Agent.curr_time
        ## sample from the replay buffer
        obses_t, actions_t, rewards_t, next_obses_t, dones_t, weights = Agent.replay_buffer[self.index].sample(Agent.batch_size)   
        if Agent.signaling_type == "target":
            targets_t = tf.constant(rewards_t, dtype=float)
            obses_t = tf.constant(obses_t)
            actions_t = tf.constant(actions_t)
        else:
            # ### Construct the target values
            # targets_t = []
            # action_indices_all = []
            # for indx, neighbor in enumerate(self.neighbors): #这里修改前，仅对leaf建立Agent，但是neighbor会包含spine
            #     filtered_indices = np.where(np.array(list(Agent.G.neighbors(neighbor)))!=self.index)[0] # filter the net interface from where the pkt comes
            #     # filtered_indices = np.where(np.array(list(Agent.G.neighbors(neighbor)))!=1000)[0] # filter the net interface from where the pkt comes
            #     action_indices = np.where(actions_t == indx)[0]
            #     action_indices_all.append(action_indices)
            #     if len(action_indices):
            #         if neighbor not in Agent.agents or Agent.agents[neighbor] is None: #消除spine的干扰
            #             targets_t.append(tf.convert_to_tensor(rewards_t[action_indices], dtype = tf.float32))
            #             continue
            #         if Agent.signaling_type in ("NN", "ideal"):
            #             targets_t.append(Agent.agents[self.index].get_neighbor_target_value(indx, 
            #                                                                                 rewards_t[action_indices], 
            #                                                                                 tf.constant(np.array(np.vstack(next_obses_t[action_indices]),
            #                                                                                                     dtype=float)), 
            #                                                                                 dones_t[action_indices],
            #                                                                                 filtered_indices))
            # action_indices_all = np.concatenate(action_indices_all)
            # ### prepare tf variables
            # try:
            #     obses_t = tf.constant(obses_t[action_indices_all,])
            # except:
            #     print("ERROR")
            #     print("Node: ", self.index)
            #     print(obses_t[0], obses_t.shape, type(obses_t[0]))
            #     raise(1)
            # actions_t = tf.constant(actions_t[action_indices_all], shape=(Agent.batch_size))
            # targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(Agent.batch_size))
            # Use immediate reward as target (disable local bootstrap).
            obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
            actions_t = tf.convert_to_tensor(actions_t, dtype=tf.int32)
            targets_t = tf.convert_to_tensor(rewards_t, dtype=tf.float32)
            targets_t = tf.reshape(targets_t, shape=(Agent.batch_size,))
        
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        ### Make a gradient step
        td_errors = Agent.agents[self.index].train(obses_t, actions_t, targets_t, weights)
        
        ## log the td error and replay buffer length
        if len(td_errors):
            with self.tb_writer_dict["td_error"].as_default():
                tf.summary.scalar('MSE_loss_over_steps', np.mean(td_errors**2), step=self.gradient_step_idx)
                tf.summary.scalar('MSE_loss_over_time', np.mean(td_errors**2), step=int((Agent.base_curr_time  + Agent.curr_time)*1e6))
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
