import os
import numpy as np
import logging
import time
import sys
from collections import deque
from xworld import xworld_navi_goal_obs_crop, xworld_navi_goal_gt_dnc, xworld_args
from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer_drqn_ini import ReplayBuffer
from utils.schedule import LinearExploration, LinearSchedule
from utils.dnc_simp2 import DNC
import tensorflow as tf
import tensorflow.contrib.layers as layers
from configs.drqn_xworld_agent import config as config_agent
from configs.drqn_xworld_planner import config as config_planner
from drqn_xworld_agent import DRQN
import shutil
import copy
import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

class DRQN_planner(object):
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


    def get_best_action(self, state, h_state, slen, past_a, agate):
        """
        Return best action

        Args:
            state: observation
            h_state: hidden state for rnn
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values, h_state_out = self.sess.run([self.q, self.hs_out], feed_dict={self.s: state, self.hs: h_state, self.slen: slen, self.past_a: past_a, self.agate: agate})
        return np.argmax(action_values[0]), action_values[0], h_state_out

    def get_action(self, state, h_state, slen, past_a, agate):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
            h_state: hidden state for rnn
        """
        best_action, q_values, h_state_out = self.get_best_action(state, h_state, slen, past_a, agate)
        if np.random.random() < self.config.soft_epsilon:
            return np.random.randint(0, 2), q_values, h_state_out
        else:
            return best_action, q_values, h_state_out
            
    @property
    def policy(self):
        """
        model.policy(state, h_state) = action
        """
        return lambda state, h_state, slen, past_a, agate: self.get_action(state, h_state, slen, past_a, agate)

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
        self.hs = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.slen = tf.placeholder(tf.int32, shape=(None))
        self.sp = tf.placeholder(tf.bool, shape=(None, None, state_shape[0], state_shape[1], state_shape[2]))
        self.hsp = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.splen = tf.placeholder(tf.int32, shape=(None))

        self.agate = tf.placeholder(tf.float32, shape=(None)) # (nb*state_history,)
        self.agatep = tf.placeholder(tf.float32, shape=(None)) # (nb*state_history,)

        self.a = tf.placeholder(tf.int32, shape=(None)) # (nb*state_history,)
        self.past_a = tf.placeholder(tf.int32, shape=(None)) # (nb*state_history,)
        self.r = tf.placeholder(tf.float32, shape=(None)) # (nb*state_history,)
        self.done_mask = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.seq_mask = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.lr = tf.placeholder(tf.float32, shape=(None))

        self.s_dnc = tf.placeholder(tf.float32, shape=(None, 4*self.config.word_size+self.config.mask_size+2))
        self.hs_dnc = DNC.state_placeholder(self.config)
        self.src_inputs_dnc = tf.placeholder(tf.float32, shape=(None, 4*4, 5, 5, self.config.ndigits*self.config.nway))
        self.tgt_inputs_dnc = tf.placeholder(tf.float32, shape=(None, 4*4, 5, 5, self.config.ndigits*self.config.nway))
        self.max_len_dnc = tf.placeholder(tf.int32, shape=(None))

    def clip_if_enabled(self, x):
        if self.config.clip_val > 0:
          return tf.clip_by_value(x, -self.config.clip_val, self.config.clip_val)
        else:
            return x

    def get_q_values_op(self, state, past_a, agate, seq_len, h_state, scope, reuse=False):
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
        past_a = tf.reshape(tf.one_hot(past_a, num_actions), shape=(-1, max_seq_len, 1, num_actions))
        past_a = tf.tile(past_a, multiples=[1,1,4,1])

        out = tf.reshape(state, shape=(-1, max_seq_len, 4, np.int32(state_shape[0]*state_shape[1]*state_shape[2]/4)))

        out = tf.reduce_sum(state[:,:,:,:,1:17], axis=4, keep_dims=True)
        out = tf.concat([out, tf.expand_dims(state[:,:,:,:,0],4), tf.expand_dims(state[:,:,:,:,17],4), state[:,:,:,:,22:24]], axis=4)
        out = tf.reshape(out, shape=(-1, max_seq_len, 4, 3*3*5))

        agate = tf.reshape(agate, shape=(-1, max_seq_len, 1, 1))
        agate = tf.tile(agate, multiples=[1,1,4,1])
        with tf.variable_scope(scope, reuse = False):
            #### recurrent
            out = tf.concat([out, past_a, agate], axis=3)
            #out = tf.concat([out, past_a], axis=3)
            out = layers.fully_connected(layers.fully_connected(out, 200), 100)
            out = tf.reduce_max(out, axis=2)
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
            out, h_state_out = tf.nn.dynamic_rnn(inputs=out, cell=lstm_cell, sequence_length=seq_len, dtype=tf.float32, initial_state=h_state)
            out = tf.reshape(out, shape=[-1,h_size])
            out = self.clip_if_enabled(out)
            h_state_out = tf.nn.rnn_cell.LSTMStateTuple(self.clip_if_enabled(h_state_out[0]),self.clip_if_enabled(h_state_out[1]))

            streamA, streamV = tf.split(out, 2, axis=1)
            advantage = layers.fully_connected(streamA, num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            value = layers.fully_connected(streamV, 1, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = value+tf.subtract(advantage,tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return out, h_state_out


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
                grads_and_vars = [(tf.clip_by_norm(grad, self.config.grad_clip_val), var) for grad, var in grads_and_vars]
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
            self.q, self.hs_out = self.get_q_values_op(s, self.past_a, self.agate, self.slen, self.hs, scope="q", reuse=False)

            # compute Q values of next state
            sp = self.process_state(self.sp)
            self.target_q, self.hsp_out = self.get_q_values_op(sp, self.a, self.agatep, self.splen, self.hsp, scope="target_q", reuse=False)

            self.dnc = DNC(self.config, self.config.dnc_h_size)
            self.out_dnc, self.hs_out_dnc = self.dnc(self.s_dnc, self.hs_dnc)
            self.agate_dnc = self.dnc.get_agate(self.s_dnc, self.hs_dnc)
            self.path_dnc, self.target_ldm_dnc = self.dnc.value_iteration(self.hs_dnc.access_state, self.src_inputs_dnc, self.tgt_inputs_dnc, self.max_len_dnc)

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

            # for saving networks weights
            #self.saver = tf.train.Saver(max_to_keep=50, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q'))
            self.saver = tf.train.Saver(max_to_keep=50)
            if self.config.restore_param:
                self.restore(self.config.restore_t)

            # synchronise q and target_q networks
            self.sess.run(self.update_target_op)

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

        s_batch, slen_batch, a_batch, past_a_batch, agate_batch, agatep_batch, r_batch, done_mask_batch, seq_mask_batch, sp_batch, splen_batch = replay_buffer.sample_batch(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.slen: slen_batch,
            self.hs: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
            self.a: a_batch,
            self.past_a: past_a_batch,
            self.agate: agate_batch,
            self.agatep: agatep_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.splen: splen_batch,
            self.hsp: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
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

        npath = self.config.npath # paths to generate in each environment
        nsubgoal = self.config.nsubgoal
        nquery = self.config.nquery # query to generate in each environment
        max_plan_len = self.config.max_plan_len
        ndigits = self.config.ndigits
        nway = self.config.nway

        num_classes = len(self.env.state.xmap.item_class_id)

        # three steps:
        #   1. sample paths from the teacher environment and pass to dnc
        #   2. get immediate reward from whether agent could reach the subgoal
        #   3. sample query paths and ask agent to follow the plan, get the final big reward
        #   -- train one step after each teacher's move
        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            self.env.reset()
            model_a.env.reset()
            model_a.env.state.copy_state(model_a.env.agent, self.env.state)
            dnc_state = DNC.zero_state(self.config, batch_size=1)
            slen = np.ones(1).astype('int32')
            action = 0

            # sample paths
            for i in range(npath):
                #state_seq, path_loc, path_ori = self.env.teacher.gen_sample_seq(self.env.state)
                state_seq, path_loc, path_ori, flag_seq = self.env.teacher.gen_sample_long_seq(self.env.state, nsubgoal)
                state_seq_encoding = DRQN_planner.encode_state(state_seq, ndigits, nway)
                goal_state_seq = copy.deepcopy(np.reshape(state_seq, [state_seq.shape[0], 4, 3, 3, num_classes+2]).astype('bool'))
                #### missing could be everything ####
                goal_state_seq = np.tile(goal_state_seq[:,:,:,:,[num_classes]], [1,1,1,1,num_classes+2])+goal_state_seq
                #### treat missing observation as correct observation ####
                goal_state_seq[:,:,:,:,num_classes] = True
                #### transpose
                goal_state_seq = np.transpose(goal_state_seq, [0,2,3,4,1])
                path_len = state_seq.shape[0]
                mask_seq = np.logical_not(state_seq[:,:3,:,num_classes])
                #flag_seq = np.zeros([path_len])
                #flag_seq[-1] = 1
                model_a.env.state.teleport(model_a.env.agent, path_loc[0], orientation_map[path_ori[0]])
                h_state = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))

                for j in range(path_len):
                    # get agate from dnc
                    cur_dnc_in = np.concatenate([state_seq_encoding[j].reshape(-1),mask_seq[j].reshape(-1), np.array([0, flag_seq[j]])], axis=0)
                    agate_dnc_val = self.sess.run(self.agate_dnc, feed_dict={self.s_dnc: cur_dnc_in[None], self.hs_dnc: dnc_state})
                    agate_dnc_val = agate_dnc_val[0,0]
                    # get q value and sample action
                    idx = replay_buffer.store_frame(state_seq[j])
                    q_input = replay_buffer.encode_recent_observation()
                    best_action, q_values, h_state = self.get_best_action([q_input], h_state, slen, [action], [agate_dnc_val])
                    action = exp_schedule.get_action(best_action)
                    # store q values
                    max_q_values.append(max(q_values))
                    q_values += list(q_values)
                    # take action and update dnc
                    cur_dnc_in[-2] = action
                    dnc_state = self.sess.run(self.hs_out_dnc, feed_dict={self.s_dnc: cur_dnc_in[None], self.hs_dnc: dnc_state})
                    # acquire reward
                    reward = 0
                    done = False
                    if action==1:
                        h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                        model_a.env.teacher.set_goal(goal_state_seq[j], path_loc[j])
                        reward_a = model_a.navi_goal(h_state_a, goal_state_seq[j])
                        # strong penalize all zero
                        #if goal_state_seq[j][:,:,0,:].all():
                        #    reward += -1
                        if not model_a.env.teacher.goal_finish:
                            reward += -0.05
                        reward += -0.05
                        model_a.env.state.teleport(model_a.env.agent, path_loc[j], orientation_map[path_ori[j]])
                    # acquire final reward
                    if i==npath-1 and j==path_len-1:
                        done = True
                        reward_list = list()
                        for k in range(nquery):
                            reward_list.append(0)
                            src_inputs, tgt_inputs, src_loc, tgt_loc, goal_obs_onehot_state = self.env.teacher.gen_sample_query(self.env.state, path_loc)
                            src_inputs = DRQN_planner.encode_state(src_inputs, ndigits, nway)
                            tgt_inputs = DRQN_planner.encode_state(tgt_inputs, ndigits, nway)
                            path_dnc_val, target_ldm_dnc_val = self.sess.run([self.path_dnc, self.target_ldm_dnc], feed_dict={self.hs_dnc: dnc_state, self.src_inputs_dnc: src_inputs[None], 
                                self.tgt_inputs_dnc: tgt_inputs[None], self.max_len_dnc: max_plan_len})
                            path_dnc_val = DRQN_planner.decode_state(np.reshape(path_dnc_val[0], [max_plan_len, 3, 3, -1]), ndigits, nway, num_classes+2)
                            target_ldm_dnc_val = DRQN_planner.decode_state(np.reshape(target_ldm_dnc_val[0], [3, 3, -1]), ndigits, nway, num_classes+2)
                            path_dnc_val_inner = np.argmax(path_dnc_val, axis=3)
                            target_ldm_dnc_val_inner = np.argmax(target_ldm_dnc_val, axis=2)
                            cur_len = max_plan_len
                            for l in range(max_plan_len):
                                if (path_dnc_val_inner[l]==target_ldm_dnc_val_inner).all():
                                    cur_len = l+1
                                    break
                            path_dnc_val = path_dnc_val[:cur_len]
                            path_dnc_val = np.concatenate([path_dnc_val, goal_obs_onehot_state[None]], 0)
                            #### modify goal state ####
                            #### missing could be everything ####
                            path_dnc_val = np.tile(path_dnc_val[:,:,:,[num_classes]], [1,1,1,num_classes+2])+path_dnc_val
                            #### treat missing observation as correct observation ####
                            path_dnc_val[:,:,:,num_classes] = True
                            model_a.env.state.teleport(model_a.env.agent, src_loc, np.array([0,1]))
                            h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                            for l in range(path_dnc_val.shape[0]):
                                cur_goal_state = path_dnc_val[l]
                                cur_goal_state = np.expand_dims(cur_goal_state, 3)
                                cur_goal_state = np.concatenate([np.rot90(cur_goal_state, 0), np.rot90(cur_goal_state, 1),
                                    np.rot90(cur_goal_state, 2), np.rot90(cur_goal_state, 3)], 3)
                                model_a.env.teacher.set_goal(cur_goal_state, tgt_loc)
                                reward_list[-1] += model_a.navi_goal(h_state_a, cur_goal_state)
                            if model_a.env.teacher.goal_finish:
                                reward_list[-1] += 10
                        reward += sum(reward_list)/len(reward_list)
                    # store everything into replay buffer
                    replay_buffer.store_effect(idx, action, agate_dnc_val, reward, done)

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
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

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
        #### visualize landmark in a fixed scene ####
        if self.config.deploy_only and self.config.vis_heat_map:
            width, height = env.state.xmap.dim['width'], env.state.xmap.dim['height']
            goal_heatmap = np.zeros([height, width])
            goal_heatmap_norm = np.zeros([height, width])
        #############################################
        npath = self.config.npath # paths to generate in each environment
        nsubgoal = self.config.nsubgoal
        nquery = self.config.nquery # query to generate in each environment
        max_plan_len = self.config.max_plan_len
        ndigits = self.config.ndigits
        nway = self.config.nway
        num_classes = len(self.env.state.xmap.item_class_id)
        for ii in range(num_episodes):
            total_reward = 0
            env.reset()
            model_a.env.reset()
            model_a.env.state.copy_state(model_a.env.agent, env.state)
            dnc_state = DNC.zero_state(self.config, batch_size=1)
            slen = np.ones(1).astype('int32')
            action = 0

            # sample paths
            for i in range(npath):
                #state_seq, path_loc, path_ori = env.teacher.gen_sample_seq(env.state)
                state_seq, path_loc, path_ori, flag_seq = self.env.teacher.gen_sample_long_seq(self.env.state, nsubgoal)
                state_seq_encoding = DRQN_planner.encode_state(state_seq, ndigits, nway)
                goal_state_seq = copy.deepcopy(np.reshape(state_seq, [state_seq.shape[0], 4, 3, 3, num_classes+2]).astype('bool'))
                #### missing could be everything ####
                goal_state_seq = np.tile(goal_state_seq[:,:,:,:,[num_classes]], [1,1,1,1,num_classes+2])+goal_state_seq
                goal_state_seq[:,:,:,:,num_classes] = True
                #### transpose
                goal_state_seq = np.transpose(goal_state_seq, [0,2,3,4,1])
                path_len = state_seq.shape[0]
                mask_seq = np.logical_not(state_seq[:,:3,:,num_classes])
                #flag_seq = np.zeros([path_len])
                #flag_seq[-1] = 1
                model_a.env.state.teleport(model_a.env.agent, path_loc[0], orientation_map[path_ori[0]])
                h_state = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))

                for j in range(path_len):
                    # get agate from dnc
                    cur_dnc_in = np.concatenate([state_seq_encoding[j].reshape(-1),mask_seq[j].reshape(-1), np.array([0, flag_seq[j]])], axis=0)
                    agate_dnc_val = self.sess.run(self.agate_dnc, feed_dict={self.s_dnc: cur_dnc_in[None], self.hs_dnc: dnc_state})
                    agate_dnc_val = agate_dnc_val[0,0]
                    # get q value and sample action
                    action, q_values, h_state = self.get_action(state_seq[j][None][None], h_state, slen, [action], [agate_dnc_val])
                    # take action and update dnc
                    # eric
                    #if state_seq[j][:,:,0].all():
                    #    action = 0
                    cur_dnc_in[-2] = action
                    dnc_state = self.sess.run(self.hs_out_dnc, feed_dict={self.s_dnc: cur_dnc_in[None], self.hs_dnc: dnc_state})                    
                    #### visualize landmark in a fixed scene ####
                    if self.config.deploy_only and self.config.vis_heat_map:
                        if (goal_location>=0).sum()==2 and (goal_location<[width, height]).sum()==2:
                            goal_heatmap_norm[path_loc[j][1], path_loc[j][0]] += 1
                            if action==1:
                                goal_heatmap[path_loc[j][1], path_loc[j][0]] += 1
                    
            reward_list = list()
            for k in range(nquery):
                reward_list.append(0)
                src_inputs, tgt_inputs, src_loc, tgt_loc, goal_obs_onehot_state = self.env.teacher.gen_sample_query(self.env.state, path_loc)
                src_inputs = DRQN_planner.encode_state(src_inputs, ndigits, nway)
                tgt_inputs = DRQN_planner.encode_state(tgt_inputs, ndigits, nway)
                path_dnc_val, target_ldm_dnc_val = self.sess.run([self.path_dnc, self.target_ldm_dnc], feed_dict={self.hs_dnc: dnc_state, self.src_inputs_dnc: src_inputs[None], 
                    self.tgt_inputs_dnc: tgt_inputs[None], self.max_len_dnc: max_plan_len})
                path_dnc_val = DRQN_planner.decode_state(np.reshape(path_dnc_val[0], [max_plan_len, 3, 3, -1]), ndigits, nway, num_classes+2)
                target_ldm_dnc_val = DRQN_planner.decode_state(np.reshape(target_ldm_dnc_val[0], [3, 3, -1]), ndigits, nway, num_classes+2)
                path_dnc_val_inner = np.argmax(path_dnc_val, axis=3)
                target_ldm_dnc_val_inner = np.argmax(target_ldm_dnc_val, axis=2)
                cur_len = max_plan_len
                for l in range(max_plan_len):
                    if (path_dnc_val_inner[l]==target_ldm_dnc_val_inner).all():
                        cur_len = l+1
                        break
                path_dnc_val = path_dnc_val[:cur_len]
                path_dnc_val = np.concatenate([path_dnc_val, goal_obs_onehot_state[None]], 0)
                #### modify goal state ####
                #### missing could be everything ####
                path_dnc_val = np.tile(path_dnc_val[:,:,:,[num_classes]], [1,1,1,num_classes+2])+path_dnc_val
                #### treat missing observation as correct observation ####
                path_dnc_val[:,:,:,num_classes] = True
                model_a.env.state.teleport(model_a.env.agent, src_loc, np.array([0,1]))
                h_state_a = (np.zeros([1,model_a.config.h_size]),np.zeros([1,model_a.config.h_size]))
                for l in range(path_dnc_val.shape[0]):
                    cur_goal_state = path_dnc_val[l]
                    cur_goal_state = np.expand_dims(cur_goal_state, 3)
                    cur_goal_state = np.concatenate([np.rot90(cur_goal_state, 0), np.rot90(cur_goal_state, 1),
                        np.rot90(cur_goal_state, 2), np.rot90(cur_goal_state, 3)], 3)
                    model_a.env.teacher.set_goal(cur_goal_state, tgt_loc)
                    reward_list[-1] += model_a.navi_goal(h_state_a, cur_goal_state)
                if model_a.env.teacher.goal_finish:
                    reward_list[-1] += 10
            total_reward = sum(reward_list)/len(reward_list)
                    
            # updates to perform at the end of an episode
            rewards.append(total_reward)
            print(total_reward)

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

        return avg_reward

    @staticmethod
    def encode_state(state, ndigits, nway):
        # state of shape (~, one_hot_len)
        # encoding of shape (~, ndigits*nway)
        state_shape = list(np.shape(state))
        state_idx = np.argmax(np.reshape(state, [-1, state_shape[-1]]),1)
        encoding = np.zeros([len(state_idx), nway*ndigits])
        for i in range(ndigits):
            encoding[np.arange(encoding.shape[0]), state_idx%nway+i*nway] = 1
            state_idx = np.floor(state_idx/nway).astype('int32')
        state_shape[-1] = ndigits*nway
        encoding = encoding.reshape(state_shape)
        return encoding

    @staticmethod
    def decode_state(encoding, ndigits, nway, num_classes):
        state_shape = list(np.shape(encoding))
        state_idx = np.argmax(np.reshape(encoding, [-1, ndigits, nway]), 2)
        for i in range(ndigits-1):
            state_idx[:,i+1:] *= nway
        state_idx = np.sum(state_idx, axis=1)
        state_idx[state_idx>=num_classes] = num_classes-1
        state = np.zeros([len(state_idx), num_classes])
        state[np.arange(len(state_idx)),state_idx] = 1
        state_shape[-1] = num_classes
        state = state.reshape(state_shape)
        return state

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

    config_i = config_planner()
    args = xworld_args.parser().parse_args()
    args.visible_radius_unit_side = config_i.visible_radius_unit_side
    args.visible_radius_unit_front = config_i.visible_radius_unit_front
    args.ego_centric = config_i.ego_centric
    args.map_config = config_i.map_config_file
    args.goal_id = 0
    args.israndom_goal = False
    env_i = xworld_navi_goal_gt_dnc.XWorldNaviGoal(args)

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
    model_i = DRQN_planner(env_i, config_i, g_i)

    '''
    model_i.initialize()
    nldms = []
    nins = 2
    nway = 5
    ndigits = 2
    for i in range(1):
        encoding, target, pred_flag = model_i.gen_sample_seq(nway, ndigits, nins)
        j = 1
        print('Map')
        print(model_i.env.state.origin_inner_state)
        print('Encoding')
        nldms.append(0)
        print(encoding[1])
        while 1:
            print(DRQN_planner.decode_state(encoding[j][:nins*nway*ndigits], nway, ndigits, nins),
                DRQN_planner.decode_state(encoding[j][nins*nway*ndigits:-2], nway, ndigits, nins))
            j += 1
            nldms[-1] += 1
            if encoding[j][-2]==1:
                j += 1
                break
        print('Query')
        print(DRQN_planner.decode_state(encoding[j][:nins*nway*ndigits], nway, ndigits, nins),
                DRQN_planner.decode_state(encoding[j][nins*nway*ndigits:-2], nway, ndigits, nins))
        print('Anwser')
        j += 4
        while j<target.shape[0]:
            print(DRQN_planner.decode_state(target[j][:nins*nway*ndigits], nway, ndigits, nins),
                DRQN_planner.decode_state(target[j][nins*nway*ndigits:], nway, ndigits, nins))
            j += 1
    print(sum(nldms)/len(nldms))
    exit()
    '''

    if config_i.deploy_only:
        model_i.deploy(model_a)
    else:
        shutil.copyfile('./configs/drqn_xworld_planner.py', config_i.output_path+'config.py')
        shutil.copy(os.path.realpath(__file__), config_i.output_path)
        shutil.copy(config_i.map_config_file, config_i.output_path)
        model_i.run(model_a, exp_schedule, lr_schedule)
