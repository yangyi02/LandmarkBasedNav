import os
import numpy as np
import logging
import time
import sys
from collections import deque
from xworld import xworld_navi_goal_obs_crop, xworld_navi_goal_gt, xworld_args
from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer_drqn_ini import ReplayBuffer
from utils.schedule import LinearExploration, LinearSchedule
import tensorflow as tf
import tensorflow.contrib.layers as layers
from configs.drqn_xworld_agent import config as config_agent
from configs.drqn_xworld_instructor import config as config_instructor
from drqn_xworld_agent import DRQN
import shutil
import copy
import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

class DRQN_instructor(object):
    """
    Implement DRQN with Tensorflow
    """
    def __init__(self, env, config, current_graph, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            
        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.current_graph = current_graph

        # build model
        self.build()

    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, nchannels)
                    of type tf.uint8.
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state


    def get_best_action(self, state, h_state_fw, h_state_bw, slen):
        """
        Return best action

        Args:
            state: observation
            h_state: hidden state for rnn
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values, h_state_out_fw, h_state_out_bw = self.sess.run([self.q, self.hs_out_fw, self.hs_out_bw], feed_dict={self.s: state, self.hs_fw: h_state_fw, self.hs_bw: h_state_bw, self.slen: slen})
        return np.argmax(action_values[0]), action_values[0], h_state_out_fw, h_state_out_bw

    def get_best_action_batch(self, state, h_state_fw, h_state_bw, slen):
        """
        Return best action

        Args:
            state: observation
            h_state: hidden state for rnn
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values, h_state_out_fw, h_state_out_bw = self.sess.run([self.q, self.hs_out_fw, self.hs_out_bw], feed_dict={self.s: state, self.hs_fw: h_state_fw, self.hs_bw: h_state_bw, self.slen: slen})
        return np.argmax(action_values, axis=1), action_values, h_state_out_fw, h_state_out_bw

    def get_action(self, state, h_state_fw, h_state_bw, slen):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
            h_state: hidden state for rnn
        """
        best_action, q_values, h_state_out_fw, h_state_out_bw = self.get_best_action(state, h_state_fw, h_state_bw, slen)
        if np.random.random() < self.config.soft_epsilon:
            return np.random.randint(0, 2), q_values, h_state_out_fw, h_state_out_bw
        else:
            return best_action, q_values, h_state_out_fw, h_state_out_bw

    def get_action_batch(self, state, h_state_fw, h_state_bw, slen):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
            h_state: hidden state for rnn
        """
        best_action, q_values, h_state_out_fw, h_state_out_bw = self.get_best_action_batch(state, h_state_fw, h_state_bw, slen)
        action = np.zeros(best_action.shape[0])
        for i in range(action.shape[0]):
            if np.random.random() < self.config.soft_epsilon:
                action[i] = np.random.randint(0, 2)
            else:
                action[i] = best_action[i]
        return action, q_values, h_state_out_fw, h_state_out_bw
            
    @property
    def policy(self):
        """
        model.policy(state, h_state) = action
        """
        return lambda state, h_state_fw, h_state_bw, slen: self.get_action(state, h_state_fw, h_state_bw, slen)

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = 0
        self.max_reward = 0
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
        self.eval_reward = 0

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.
        """
        # here, typically, a state shape is (5,3,22)
        state_shape = list([4*3, 3, len(self.env.state.xmap.item_class_id)+2]) # four orientations
        self.s = tf.placeholder(tf.bool, shape=(None, None, state_shape[0], state_shape[1], state_shape[2]))
        self.hs_fw = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.hs_bw = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.slen = tf.placeholder(tf.int32, shape=(None))
        self.sp = tf.placeholder(tf.bool, shape=(None, None, state_shape[0], state_shape[1], state_shape[2]))
        self.hsp_fw = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.hsp_bw = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.splen = tf.placeholder(tf.int32, shape=(None))

        self.a = tf.placeholder(tf.int32, shape=(None)) # (nb*state_history,)
        self.r = tf.placeholder(tf.float32, shape=(None)) # (nb*state_history,)
        self.done_mask = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.seq_mask = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.lr = tf.placeholder(tf.float32, shape=(None))

    def get_q_values_op(self, state, seq_len, h_state_fw, h_state_bw, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, seq_len, img_w, img_h, nchannel)
            goal_state: (tf tensor)
                shape = (batch_size, 1, img_w, img_h, nchannel, 4)
            past_a: (tf tensor)
                shape = (batch_size*seq_len,)
            seq_len: (tf tensor)
                shape = (batch_size,)
            h_state: (tf tensor) 
                shape = (batch_size, h_size)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size * seq_len, num_actions)
            h_state_out: (tf tensor) of shape = (batch_size, h_size)
        """
        num_actions = 2
        h_size = self.config.h_size
        max_seq_len = tf.shape(state)[1]
        state_shape = list([4*3, 3, len(self.env.state.xmap.item_class_id)+2])
        #past_a = tf.reshape(tf.one_hot(past_a, num_actions), shape=(-1, max_seq_len, 1, num_actions))
        #past_a = tf.tile(past_a, multiples=[1,1,4,1])
        out = tf.reshape(state, shape=(-1, max_seq_len, 4, np.int32(state_shape[0]*state_shape[1]*state_shape[2]/4)))
        with tf.variable_scope(scope, reuse = False):
            #### bidirectional recurrent
            #out = tf.concat([out, past_a], axis=3)
            out = layers.fully_connected(layers.fully_connected(out, 200), 100)
            out = tf.reduce_max(out, axis=2)
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
            out, h_state_out = tf.nn.bidirectional_dynamic_rnn(inputs=out, cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, sequence_length=seq_len, dtype=tf.float32, initial_state_fw=h_state_fw, initial_state_bw=h_state_bw)
            out = tf.reshape(tf.concat(out, 2), shape=[-1,2*h_size])
            h_state_out_fw, h_state_out_bw = h_state_out

            #### recurrent
            '''
            out = tf.concat([out, past_a], axis=3)
            out = layers.fully_connected(layers.fully_connected(out, 200), 100)
            out = tf.reduce_max(out, axis=2)
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
            out, h_state_out = tf.nn.dynamic_rnn(inputs=out, cell=lstm_cell, sequence_length=seq_len, dtype=tf.float32, initial_state=h_state)
            out = tf.reshape(out, shape=[-1,h_size])
            '''
            #### feed forward
            '''
            out = layers.fully_connected(layers.fully_connected(out, 200), 100)
            out = tf.reduce_max(out, axis=2)
            out = tf.reshape(out, shape=[-1,100])
            h_state_out = h_state
            '''

            streamA, streamV = tf.split(out, 2, axis=1)
            advantage = layers.fully_connected(streamA, num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            value = layers.fully_connected(streamV, 1, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = value+tf.subtract(advantage,tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return out, h_state_out_fw, h_state_out_bw


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,q_scope)
        tgt_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,target_q_scope)
        self.update_target_op = tf.no_op()
        for i in range(len(params)):
            self.update_target_op = tf.group(self.update_target_op, tf.assign(tgt_params[i],params[i]))


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = 2#self.env.agent.num_actions
        q_sample = self.r+tf.reduce_max(target_q, axis=1)*(1-tf.cast(self.done_mask,tf.float32))*self.config.gamma
        q_pred = q*tf.one_hot(self.a, num_actions)
        q_pred = tf.reduce_sum(q_pred, axis=1)
        self.loss = tf.reduce_sum(tf.square(q_pred-q_sample)*tf.cast(self.seq_mask,tf.float32), axis=0)/tf.reduce_sum(tf.cast(self.seq_mask,tf.float32), axis=0)


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """
        with tf.variable_scope(scope):
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            grads_and_vars = opt.compute_gradients(self.loss, params)
            if self.config.grad_clip:
                grads_and_vars = [(tf.clip_by_norm(grad, self.config.clip_val), var) for grad, var in grads_and_vars]
            self.train_op = opt.apply_gradients(grads_and_vars)
            grads = [grad for grad, var in grads_and_vars]
            self.grad_norm = tf.global_norm(grads)  

    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg_Q", self.avg_q_placeholder)
        tf.summary.scalar("Max_Q", self.max_q_placeholder)
        tf.summary.scalar("Std_Q", self.std_q_placeholder)

        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.sess.graph)

    def save(self, t):
        """
        Saves session
        """
        with self.current_graph.as_default():
            if not os.path.exists(os.path.dirname(self.config.model_output)):
                os.makedirs(os.path.dirname(self.config.model_output))

            self.saver.save(self.sess, self.config.model_output, global_step = t)


    def restore(self, t):
        """
        Restore session
        """
        with self.current_graph.as_default():
            #self.saver = tf.train.import_meta_graph(self.config.model_output+'-'+str(t)+'.meta')
            self.saver.restore(self.sess, self.config.model_output+'-'+str(t))

    def build(self):
        """
        Build model by adding all necessary variables
        """
        with self.current_graph.as_default():
            # add placeholders
            self.add_placeholders_op()

            # compute Q values of state
            s = self.process_state(self.s)
            self.q, self.hs_out_fw, self.hs_out_bw = self.get_q_values_op(s, self.slen, self.hs_fw, self.hs_bw, scope="q", reuse=False)

            # compute Q values of next state
            sp = self.process_state(self.sp)
            self.target_q, self.hsp_out_fw, self.hsp_out_bw = self.get_q_values_op(sp, self.splen, self.hsp_fw, self.hsp_bw, scope="target_q", reuse=False)

            # add update operator for target network
            self.add_update_target_op("q", "target_q")

            # add square loss
            self.add_loss_op(self.q, self.target_q)

            # add optmizer for the main networks
            self.add_optimizer_op("q")

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        with self.current_graph.as_default():
            self.sess = tf.Session()
            # tensorboard stuff
            self.add_summary()

            # initiliaze all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # synchronise q and target_q networks
            self.sess.run(self.update_target_op)

            # for saving networks weights
            self.saver = tf.train.Saver(max_to_keep=50)
            if self.config.restore_param:
                self.restore(self.config.restore_t)

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)

    def convert_state_to_goal_state(self, state):
        #### goal state is used for filtering and raw_goal_state is used to predict Q
        #### crop to get 3x3 goal ####
        side_radius = self.config.visible_radius_unit_side
        raw_goal_state = copy.deepcopy(state[:3,side_radius-1:side_radius+2,:])
        goal_state = copy.deepcopy(raw_goal_state)
        #### missing could be everything ####
        num_classes = len(self.env.state.xmap.item_class_id)
        for i in range(3):
            for j in range(3):
                if goal_state[i,j,num_classes] == True:
                    goal_state[i,j,:] = True
        #### treat missing observation as correct observation ####
        goal_state[:,:,num_classes] = True
        #### rotate ####
        goal_state = np.expand_dims(goal_state, 3)
        goal_state = np.concatenate([np.rot90(goal_state, 0), np.rot90(goal_state, 1),
            np.rot90(goal_state, 2), np.rot90(goal_state, 3)], 3)
        #### rotate raw goal state ####
        raw_goal_state = np.concatenate([np.rot90(raw_goal_state, 0), np.rot90(raw_goal_state, 1),
            np.rot90(raw_goal_state, 2), np.rot90(raw_goal_state, 3)], 0)
        return raw_goal_state, goal_state

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save(t)

        return loss_eval, grad_eval

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, slen_batch, a_batch, past_a_batch, r_batch, done_mask_batch, seq_mask_batch, sp_batch, splen_batch = replay_buffer.sample_batch(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.slen: slen_batch,
            self.hs_fw: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
            self.hs_bw: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.splen: splen_batch,
            self.hsp_fw: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
            self.hsp_bw: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
            self.done_mask: done_mask_batch,
            self.seq_mask: seq_mask_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: self.avg_reward, 
            self.max_reward_placeholder: self.max_reward, 
            self.std_reward_placeholder: self.std_reward, 
            self.avg_q_placeholder: self.avg_q, 
            self.max_q_placeholder: self.max_q, 
            self.std_q_placeholder: self.std_q, 
            self.eval_reward_placeholder: self.eval_reward, 
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval

    def train(self, model_a, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """
        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        #scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        self.env.state.is_render_image = self.config.render_train
        model_a.env.state.is_render_image = model_a.config.render_train
        orientation_map = [np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0])]

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            flag = True
            while flag:
                state = self.env.reset() # h x w x c
                agent_location = self.env.state.agent_location
                if self.env.teacher.dist_map[agent_location[1],agent_location[0]]!=np.inf:
                    flag = False
            model_a.env.reset()
            model_a.env.state.copy_state(model_a.env.agent, self.env.state)
            h_state_fw = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))
            h_state_bw = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))
            state_batch = list()
            goal_state_batch = list()
            goal_obs_image_batch = list()
            path_loc = list()
            path_ori = list()
            done_batch = list()
            width, height = self.env.state.xmap.dim['width'], self.env.state.xmap.dim['height']
            side_radius = min(self.config.visible_radius_unit_side, max(width - 1, height - 1))
            block_size = self.env.state.image_block_size
            for i in range(200):
                #### teacher rotate ####
                agent_location = self.env.state.agent_location
                agent_orientation = self.env.state.agent_orientation
                goal_location = agent_location+agent_orientation
                gt_action = self.env.teacher.action_map[agent_location[1], agent_location[0]]
                if np.dot(agent_orientation, orientation_map[gt_action])!=1:
                    tmp = np.cross(agent_orientation, orientation_map[gt_action])
                    if tmp==1:
                        state, reward_i, done = self.env.step(3)
                    else:
                        state, reward_i, done = self.env.step(2)
                    continue
                path_loc.append(copy.deepcopy(goal_location))
                path_ori.append(copy.deepcopy(agent_orientation))
                raw_goal_state, goal_state = self.convert_state_to_goal_state(state)
                state_batch.append(raw_goal_state[None][None])
                goal_state_batch.append(goal_state)
                if self.config.render_train:
                    goal_obs_image_batch.append(self.env.state.image[:3*block_size, (side_radius-1)*block_size:(side_radius+2)*block_size, :])
                state, reward_i, done = self.env.step(0)
                done_batch.append(done)
                if done:
                    break

            slen = np.array([len(state_batch)]).astype('int32')
            state_batch = np.concatenate(state_batch, axis=1)
            best_action_batch, q_values_batch, h_state_fw, h_state_bw = self.get_best_action_batch(state_batch, h_state_fw, h_state_bw, slen)
            action_batch = exp_schedule.get_action_batch(best_action_batch)
            for i in range(q_values_batch.shape[0]):
                max_q_values.append(max(q_values_batch[i]))
                q_values += list(q_values_batch[i])

            reward_batch = list()
            for i, action in enumerate(action_batch):
                if action==0:
                    reward_batch.append(0)
                else:
                    if self.config.render_train:
                        model_a.env.teacher.goal_obs_image = goal_obs_image_batch[i]
                    h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                    model_a.env.teacher.set_goal(goal_state_batch[i], path_loc[i])
                    reward_a = model_a.navi_goal(h_state_a, goal_state_batch[i])
                    if model_a.env.teacher.goal_finish:
                        reward_batch.append(-0.05)
                    else:
                        reward_batch.append(-0.1)
                    #model_a.env.state.teleport(model_a.env.agent, path_loc[i], path_ori[i])
            if action_batch[-1]==1 and model_a.env.teacher.goal_finish:
                reward_batch[-1] += 1
            else:
                if self.config.render_train:
                        model_a.env.teacher.goal_obs_image = goal_obs_image_batch[-1]
                h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                model_a.env.teacher.set_goal(goal_state_batch[-1], path_loc[-1])
                reward_a = model_a.navi_goal(h_state_a, goal_state_batch[-1])
                if model_a.env.teacher.goal_finish:
                    reward_batch[-1] += 1

            for i in range(action_batch.shape[0]):
                idx = replay_buffer.store_frame(state_batch[0][i])
                replay_buffer.store_effect(idx, action_batch[i], reward_batch[i], done_batch[i])

            for i in range(action_batch.shape[0]):
                t += 1
                last_eval += 1
                last_record += 1
                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                   (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward), 
                                        ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                        ("Grads", grad_eval), ("Max Q", self.max_q), 
                                        ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
                                                        self.config.learning_start))
                    sys.stdout.flush()

            # count reward
            total_reward = sum(reward_batch)
            # updates to perform at the end of an episode
            rewards.append(total_reward)          

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                self.logger.info("Global step: %d"%(t))
                scores_eval += [self.evaluate(model_a)]

        # last words
        self.logger.info("- Training done.")
        self.save(t)
        scores_eval += [self.evaluate(model_a)]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def evaluate(self, model_a, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env
        env.state.is_render_image = self.config.render_test
        model_a.env.state.is_render_image = model_a.config.render_test
        rewards = []
        orientation_map = [np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0])]
        num_success = 0
        num_ldms = []
        #### visualize landmark in a fixed scene ####
        if self.config.deploy_only and self.config.vis_heat_map:
            width, height = self.env.state.xmap.dim['width'], self.env.state.xmap.dim['height']
            goal_heatmap = np.zeros([height, width])
        #############################################
        for i in range(num_episodes):
            total_reward = 0
            flag = True
            while flag:
                state = env.reset() # h x w x c
                agent_location = env.state.agent_location
                if env.teacher.dist_map[agent_location[1],agent_location[0]]!=np.inf:
                    flag = False
            model_a.env.reset()
            model_a.env.state.copy_state(model_a.env.agent, env.state)
            h_state_fw = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))
            h_state_bw = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))
            h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
            state_batch = list()
            goal_state_batch = list()
            goal_obs_image_batch = list()
            path_loc = list()
            path_ori = list()
            done_batch = list()
            width, height = self.env.state.xmap.dim['width'], self.env.state.xmap.dim['height']
            side_radius = min(self.config.visible_radius_unit_side, max(width - 1, height - 1))
            block_size = self.env.state.image_block_size
            for i in range(50):
                #### teacher rotate ####
                agent_location = self.env.state.agent_location
                agent_orientation = self.env.state.agent_orientation
                goal_location = agent_location+agent_orientation
                gt_action = self.env.teacher.action_map[agent_location[1], agent_location[0]]
                if np.dot(agent_orientation, orientation_map[gt_action])!=1:
                    tmp = np.cross(agent_orientation, orientation_map[gt_action])
                    if tmp==1:
                        state, reward_i, done = self.env.step(3)
                    else:
                        state, reward_i, done = self.env.step(2)
                    continue
                path_loc.append(copy.deepcopy(goal_location))
                path_ori.append(copy.deepcopy(agent_orientation))
                raw_goal_state, goal_state = self.convert_state_to_goal_state(state)
                state_batch.append(raw_goal_state[None][None])
                goal_state_batch.append(goal_state)
                if self.config.render_test:
                    goal_obs_image_batch.append(self.env.state.image[:3*block_size, (side_radius-1)*block_size:(side_radius+2)*block_size, :])
                state, reward_i, done = self.env.step(0)
                done_batch.append(done)
                if done:
                    break

            slen = np.array([len(state_batch)]).astype('int32')
            state_batch = np.concatenate(state_batch, axis=1)
            action_batch, q_values_batch, h_state_fw, h_state_bw = self.get_action_batch(state_batch, h_state_fw, h_state_bw, slen)

            succeed_flag = 1
            reward_batch = list()
            for i, action in enumerate(action_batch):
                if action==1:
                    if self.config.render_test:
                        model_a.env.teacher.goal_obs_image = goal_obs_image_batch[i]
                    h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                    model_a.env.teacher.set_goal(goal_state_batch[i], path_loc[i])
                    reward_a = model_a.navi_goal(h_state_a, goal_state_batch[i])
                    reward_batch.append(reward_a)
                    #### visualize landmark in a fixed scene ####
                    if self.config.deploy_only and self.config.vis_heat_map:
                        if (path_loc[i]>=0).sum()==2 and (path_loc[i]<[width, height]).sum()==2:
                            goal_heatmap[path_loc[i][1], path_loc[i][0]] += 1
                    #############################################
            if action_batch[-1]==1 and model_a.env.teacher.goal_finish:
                reward_batch[-1] += 1
            else:
                if self.config.render_test:
                        model_a.env.teacher.goal_obs_image = goal_obs_image_batch[-1]
                h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                model_a.env.teacher.set_goal(goal_state_batch[-1], path_loc[-1])
                reward_a = model_a.navi_goal(h_state_a, goal_state_batch[-1])
                reward_batch.append(reward_a)
                if model_a.env.teacher.goal_finish:
                    reward_batch[-1] += 1
                else:
                    succeed_flag = 0

            num_ldms.append(sum(action_batch))
            # count reward
            total_reward += sum(reward_batch)
            num_success += succeed_flag
            # updates to perform at the end of an episode
            rewards.append(total_reward)    

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        #### visualize landmark in a fixed scene ####
        if self.config.deploy_only and self.config.vis_heat_map:
            #goal_heatmap /= (goal_heatmap_norm+1e-6)
            plt.imshow(goal_heatmap, cmap='hot', interpolation='nearest')
            plt.show()
            plt.pause(100)
        #############################################

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)
            msg = "Success Rate: {:04.2f}".format(num_success/num_episodes)
            self.logger.info(msg)
            msg = "Number Landmark: {:04.2f}".format(sum(num_ldms)/len(num_ldms))
            self.logger.info(msg)

        return avg_reward

    def run(self, model_a, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()
        # model
        self.train(model_a, exp_schedule, lr_schedule)

    def deploy(self, model_a):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()
        # model
        self.evaluate(model_a)  


if __name__ == '__main__':
    config_a = config_agent()
    args = xworld_args.parser().parse_args()
    args.visible_radius_unit_side = config_a.visible_radius_unit_side
    args.visible_radius_unit_front = config_a.visible_radius_unit_front
    args.ego_centric = config_a.ego_centric
    args.map_config = config_a.map_config_file
    args.goal_id = 0
    args.israndom_goal = False
    env_a = xworld_navi_goal_obs_crop.XWorldNaviGoal(args)

    config_i = config_instructor()
    args = xworld_args.parser().parse_args()
    args.visible_radius_unit_side = config_i.visible_radius_unit_side
    args.visible_radius_unit_front = config_i.visible_radius_unit_front
    args.ego_centric = config_i.ego_centric
    args.map_config = config_i.map_config_file
    args.goal_id = 0
    args.israndom_goal = False
    env_i = xworld_navi_goal_gt.XWorldNaviGoal(args)

    # load model
    g_a = tf.Graph()
    model_a = DRQN(env_a, config_a, g_a)
    model_a.initialize()

    # exploration strategy
    exp_schedule = LinearExploration(2, config_i.eps_begin, 
            config_i.eps_end, config_i.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config_i.lr_begin, config_i.lr_end,
            config_i.lr_nsteps)

    # train model
    g_i = tf.Graph()
    model_i = DRQN_instructor(env_i, config_i, g_i)

    if config_i.deploy_only:
        model_i.deploy(model_a)
    else:
        shutil.copyfile('./configs/drqn_xworld_instructor.py', config_i.output_path+'config.py')
        shutil.copy(os.path.realpath(__file__), config_i.output_path)
        shutil.copy(config_i.map_config_file, config_i.output_path)
        model_i.run(model_a, exp_schedule, lr_schedule)
