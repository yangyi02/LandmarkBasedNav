import os
import numpy as np
import logging
import time
import sys
from collections import deque
from xworld import xworld_navi_goal_obs_crop, xworld_args
from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer_drqn import ReplayBuffer
from utils.schedule import LinearExploration, LinearSchedule
import tensorflow as tf
import tensorflow.contrib.layers as layers
from configs.drqn_xworld_agent import config
import shutil
import copy

class DRQN(object):
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


    def get_best_action(self, state, goal_state, h_state, slen, past_a):
        """
        Return best action

        Args:
            state: observation
            h_state: hidden state for rnn
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values, h_state_out = self.sess.run([self.q, self.hs_out], feed_dict={self.s: state, self.gs: goal_state, self.hs: h_state, self.slen: slen, self.past_a: past_a})
        return np.argmax(action_values[0]), action_values[0], h_state_out

    def get_action(self, state, goal_state, h_state, slen, past_a):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
            h_state: hidden state for rnn
        """
        best_action, q_values, h_state_out = self.get_best_action(state, goal_state, h_state, slen, past_a)
        if np.random.random() < self.config.soft_epsilon:
            return self.env.agent.random_action(), q_values, h_state_out
        else:
            return best_action, q_values, h_state_out
            
    @property
    def policy(self):
        """
        model.policy(state, h_state) = action
        """
        return lambda state, goal_state, h_state, slen, past_a: self.get_action(state, goal_state, h_state, slen, past_a)

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
        state_shape = list([self.env.args.visible_radius_unit_front+1, 2*self.env.args.visible_radius_unit_side+1, len(self.env.state.xmap.item_class_id)+2])
        self.s = tf.placeholder(tf.bool, shape=(None, None, state_shape[0], state_shape[1], state_shape[2]))
        self.gs = tf.placeholder(tf.bool, shape=(None, 1, 3, 3, state_shape[2], 4))
        self.hs = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.slen = tf.placeholder(tf.int32, shape=(None))
        self.sp = tf.placeholder(tf.bool, shape=(None, None, state_shape[0], state_shape[1], state_shape[2]))
        self.hsp = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=(None, self.config.h_size)),tf.placeholder(tf.float32, shape=(None, self.config.h_size)))
        self.splen = tf.placeholder(tf.int32, shape=(None))

        self.a = tf.placeholder(tf.int32, shape=(None)) # (nb*state_history,)
        self.past_a = tf.placeholder(tf.int32, shape=(None)) # (nb*state_history,)
        self.r = tf.placeholder(tf.float32, shape=(None)) # (nb*state_history,)
        self.done_mask = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.seq_mask = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.lr = tf.placeholder(tf.float32, shape=(None))

    def get_q_values_op(self, state, goal_state, past_a, seq_len, h_state, scope, reuse=False):
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
        num_actions = self.env.agent.num_actions
        h_size = self.config.h_size
        max_seq_len = tf.shape(state)[1]
        past_a = tf.reshape(tf.one_hot(past_a, num_actions), shape=(-1, max_seq_len, num_actions))
        state_shape = list([self.env.args.visible_radius_unit_front+1, 2*self.env.args.visible_radius_unit_side+1, len(self.env.state.xmap.item_class_id)+2])
        goal_state = tf.reshape(goal_state, shape=[-1, 3, 3, state_shape[2], 4])
        with tf.variable_scope(scope, reuse = False):
            #heatmap = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], strides=[1,1,1,1], padding='SAME'), (state, goal_state), dtype=tf.float32)
            #### pad unknown ####
            pad_vec_w = tf.tile(tf.reshape(tf.one_hot(state_shape[2]-2, state_shape[2]),shape=[1,1,1,state_shape[2]]),[max_seq_len, state_shape[0], 1, 1])
            pad_vec_h = tf.tile(tf.reshape(tf.one_hot(state_shape[2]-2, state_shape[2]),shape=[1,1,1,state_shape[2]]),[max_seq_len, 1, state_shape[1]+2, 1])
            heatmap = tf.map_fn(lambda x: tf.nn.conv2d(tf.concat([pad_vec_h, tf.concat([pad_vec_w, x[0], pad_vec_w], axis=2), pad_vec_h], axis=1), x[1], strides=[1,1,1,1], padding='VALID'), (state, goal_state), dtype=tf.float32)
            ####
            heatmap = tf.reduce_max(heatmap, axis=4, keep_dims=True)
            heatmap = tf.equal(heatmap, 9)

            #heatmap = tf.reshape(heatmap, shape=(-1, max_seq_len, state_shape[0]*state_shape[1]))
            #state = tf.reshape(state, shape=[-1, max_seq_len, state_shape[0]*state_shape[1]*state_shape[2]])
            
            #out = tf.concat([tf.cast(heatmap, tf.float32), state], axis=4)
            #out = tf.reshape(out, shape=[-1, max_seq_len, state_shape[0]*state_shape[1]*(state_shape[2]+1)])
            out = tf.concat([tf.cast(heatmap, tf.float32), tf.expand_dims(state[:,:,:,:,0],4)], axis=4)
            out = tf.reshape(out, shape=[-1, max_seq_len, state_shape[0]*state_shape[1]*(1+1)])
            out = tf.concat([out, past_a], axis=2)
            

            #state = layers.fully_connected(layers.fully_connected(state, 200), 100)
            #heatmap = layers.fully_connected(layers.fully_connected(heatmap, 100), 100)
            #past_a = layers.fully_connected(layers.fully_connected(past_a, 20), 20)
            #out = tf.concat([state, heatmap, past_a], axis=2)
            #out = layers.fully_connected(out, 100)

            #out = tf.concat([state, heatmap, past_a], axis=2)
            out = layers.fully_connected(layers.fully_connected(out, 200),100)

            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
            out, h_state_out = tf.nn.dynamic_rnn(inputs=out, cell=lstm_cell, sequence_length=seq_len, dtype=tf.float32, initial_state=h_state)
            out = tf.reshape(out, shape=[-1,h_size])
            streamA, streamV = tf.split(out, 2, axis=1)
            advantage = layers.fully_connected(streamA, num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            value = layers.fully_connected(streamV, 1, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = value+tf.subtract(advantage,tf.reduce_mean(advantage, axis=1, keep_dims=True))

        '''
        with tf.variable_scope(scope, reuse = False):
            state = layers.conv2d(state, num_outputs = 50, kernel_size = 3, scope='siamese_l1')
            state = layers.conv2d(state, num_outputs = 50, kernel_size = 3, scope='siamese_l2')
            state = layers.max_pool2d(state, kernel_size = [3, 3], stride = 2, scope='siamese_l3')
        with tf.variable_scope(scope, reuse = True):
            goal_state = layers.conv2d(goal_state, num_outputs = 50, kernel_size = 3, scope='siamese_l1')
            goal_state = layers.conv2d(goal_state, num_outputs = 50, kernel_size = 3, scope='siamese_l2')
            goal_state = layers.max_pool2d(goal_state, kernel_size = [3, 3], stride = 2, scope='siamese_l3')
        with tf.variable_scope(scope, reuse = reuse):
            state = tf.reshape(state, shape=[-1, max_seq_len, 2*2*50])
            goal_state = tf.reshape(goal_state, shape=[-1, 1, 1*1*50])
            goal_state = tf.tile(goal_state, [1, max_seq_len, 1])
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size)
            out = tf.concat([state, goal_state, past_a], axis=2)
            out = layers.fully_connected(out, 100)
            out, h_state_out = tf.nn.dynamic_rnn(inputs=out, cell=lstm_cell, sequence_length=seq_len, dtype=tf.float32, initial_state=h_state)
            #### out here has shape (batch_size, state_history, h_size), h_state_out has shape (batch_size, h_size) ####
            out = tf.reshape(out, shape=[-1,h_size])
            streamA, streamV = tf.split(out, 2, axis=1)
            advantage = layers.fully_connected(streamA, num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            value = layers.fully_connected(streamV, 1, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = value+tf.subtract(advantage,tf.reduce_mean(advantage, axis=1, keep_dims=True))
        '''
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
        num_actions = self.env.agent.num_actions
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
            gs = self.process_state(self.gs)
            self.q, self.hs_out = self.get_q_values_op(s, gs, self.past_a, self.slen, self.hs, scope="q", reuse=False)

            # compute Q values of next state
            sp = self.process_state(self.sp)
            self.target_q, self.hsp_out = self.get_q_values_op(sp, gs, self.a, self.splen, self.hsp, scope="target_q", reuse=False)

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

        s_batch, slen_batch, gs_batch, a_batch, past_a_batch, r_batch, done_mask_batch, seq_mask_batch, sp_batch, splen_batch = replay_buffer.sample_batch(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.slen: slen_batch,
            self.gs: gs_batch,
            self.hs: (np.zeros([self.config.batch_size, self.config.h_size]),np.zeros([self.config.batch_size, self.config.h_size])),
            self.a: a_batch,
            self.past_a: past_a_batch,
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

    def train(self, exp_schedule, lr_schedule):
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

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset() # h x w x c
            goal_state = self.env.teacher.goal_obs_onehot_state # h x w x c
            h_state = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))
            slen = np.ones(1).astype('int32')
            action = 0
            for i in range(200):
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()

                #### for replay_buffer
                # replay memory stuff
                idx      = replay_buffer.store_frame(state, goal_state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values, h_state = self.get_best_action([q_input], goal_state[None][None], h_state, slen, [action])
                action                = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)
                # perform action in env
                new_state, reward, done = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

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
                scores_eval += [self.evaluate()]

        # last words
        self.logger.info("- Training done.")
        self.save(t)
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def evaluate(self, env=None, num_episodes=None):
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

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            goal_state = env.teacher.goal_obs_onehot_state # h x w x c
            h_state = (np.zeros([1,self.config.h_size]),np.zeros([1,self.config.h_size]))
            slen = np.ones(1).astype('int32')
            action = 0
            for j in range(50):
                if self.config.render_test: env.render()
                
                #### for replay_buffer
                # store last state in buffer
                idx     = replay_buffer.store_frame(state, goal_state)
                q_input = replay_buffer.encode_recent_observation()

                action, action_q, h_state = self.get_action([q_input], goal_state[None][None], h_state, slen, [action])
                #print(action, action_q)

                # perform action in env
                new_state, reward, done = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)     

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward

    def navi_goal(self, h_state, goal_state):
        state = self.env.state.onehot_state
        slen = np.ones(1).astype('int32')
        total_reward = 0
        action = 0
        for j in range(105):
            action, action_q, h_state = self.get_action(state[None][None], goal_state[None][None], h_state, slen, [action])
            state, reward, done = self.env.step(action)
            if self.env.state.is_render_image: self.env.render(2)
            total_reward += reward
            if done:
                break
        return total_reward

    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()
        # model
        self.train(exp_schedule, lr_schedule)

    def deploy(self):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()
        # model
        self.evaluate()  


if __name__ == '__main__':
    config1 = config()
    # make env
    args = xworld_args.parser().parse_args()
    args.visible_radius_unit_side = config1.visible_radius_unit_side
    args.visible_radius_unit_front = config1.visible_radius_unit_front
    args.ego_centric = config1.ego_centric
    args.map_config = config1.map_config_file
    args.goal_id = 0
    args.israndom_goal = False
    env = xworld_navi_goal_obs_crop.XWorldNaviGoal(args)

    # exploration strategy
    exp_schedule = LinearExploration(env.agent.num_actions, config1.eps_begin, 
            config1.eps_end, config1.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config1.lr_begin, config1.lr_end,
            config1.lr_nsteps)

    # train model
    g1 = tf.Graph()
    model = DRQN(env, config1, g1)

    g2 = tf.Graph()
    model2 = DRQN(env, config1, g2)

    #with g1.as_default():
    #    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
    #print(params)
    #exit()

    if config1.deploy_only:
        model.deploy()
    else:
        shutil.copyfile('./configs/drqn_xworld.py', config1.output_path+'config.py')
        shutil.copy(os.path.realpath(__file__), config1.output_path)
        shutil.copy(config1.map_config_file, config1.output_path)
        model.run(exp_schedule, lr_schedule)
