import os
import numpy as np
import logging
import time
import sys
from collections import deque
from graph_world import GraphWorld
from utils.general import get_logger, Progbar, export_plot
from utils.schedule import LinearSchedule, ExpSchedule, CurriculumSchedule
from utils.dnc import DNC
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python import debug as tf_debug
from configs.drqn_wmem_graph_world import config

class DRQNwMemGraphWorld(object):
    """
    Implement DRQN with Tensorflow
    """
    def __init__(self, env, config, logger=None):
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

    def get_action(self, pred_action, gt_action, beta):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
            h_state: hidden state for rnn
        """
        if np.random.random() < beta:
            return gt_action
        else:
            return pred_action

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.eval_acc = 0

    def update_averages(self, scores_eval):
        """
        Update the averages

        Args:
            scores_eval: list
        """
        if len(scores_eval) > 0:
            self.eval_acc = scores_eval[-1]

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.
        """
        # here, typically, a state shape is (2*4*3+2)

        self.s = tf.placeholder(tf.bool, shape=(None, None, self.config.ndigits*self.config.nway*3+2))
        self.hs = DNC.state_placeholder(self.config)
        self.slen = tf.placeholder(tf.int32, shape=(None))
        self.pred_flag = tf.placeholder(tf.bool, shape=(None)) # (nb*state_history,)
        self.target_action = tf.placeholder(tf.float32, shape=(None, self.config.num_actions)) # (nb*state_history, num_actions)
        self.lr = tf.placeholder(tf.float32, shape=(None))

    def get_q_values_op(self, state, seq_len, h_state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, seq_len, img_w, img_h, nchannel)
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
        num_actions = self.env.num_actions
        dnc_h_size = self.config.dnc_h_size
        out_raw = state

        with tf.variable_scope(scope, reuse = reuse):
            dnc_cell = DNC(config=self.config, output_size=dnc_h_size, clip_value=self.config.dnc_clip_val)

            #out_raw = tf.reshape(out_raw, shape=[-1, self.config.ndigits*self.config.nway*3+2])
            #out = layers.fully_connected(out_raw[:,:-2], 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #out = layers.fully_connected(out, 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #out = tf.concat([out,out_raw[:,-2:]], axis=1)
            #out = tf.reshape(out, shape=[-1, tf.shape(state)[1], 64+2])

            out_raw = tf.reshape(out_raw, shape=[-1, self.config.ndigits*self.config.nway*3+2])
            out = layers.fully_connected(out_raw, 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = layers.fully_connected(out, 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = tf.reshape(out, shape=[-1, tf.shape(state)[1], 64])

            #out = out_raw

            out, h_state_out = tf.nn.dynamic_rnn(inputs=out, cell=dnc_cell, sequence_length=seq_len, dtype=tf.float32, initial_state=h_state)
            # out here has shape (batch_size, state_history, dnc_h_size)
            out = layers.fully_connected(tf.reshape(out, shape=[-1,dnc_h_size]), 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = layers.fully_connected(out, 64, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = layers.fully_connected(out, num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
        return out, h_state_out


    def add_loss_op(self, pred_action, target_action, pred_flag):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            pred_action: (tf tensor) shape = (batch_size*state_history, num_actions)
            target_action: (tf tensor) shape = (batch_size*state_history, num_actions)
            pred_flag: (tf tensor) shape = (batch_size*state_history,)
        """
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=target_action, logits=pred_action, weights=pred_flag)

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
        self.eval_acc_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_acc")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", self.grad_norm)

        tf.summary.scalar("Eval_Acc", self.eval_acc_placeholder)
            
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
        #self.saver = tf.train.import_meta_graph(self.config.model_output+'-'+str(t)+'.meta')
        self.saver.restore(self.sess, self.config.model_output+'-'+str(t))

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q_logits, self.hs_out = self.get_q_values_op(s, self.slen, self.hs, scope="q", reuse=False)
        self.q = tf.nn.softmax(self.q_logits)

        # add softmax cross entropy loss
        self.add_loss_op(self.q_logits, self.target_action, self.pred_flag)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        configtmp = tf.ConfigProto()
        configtmp.gpu_options.allow_growth=True
        self.sess = tf.Session(config=configtmp)
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # for saving networks weights
        self.saver = tf.train.Saver(max_to_keep=50)

        if self.config.restore_param:
            self.restore(self.config.restore_t)

    def train_step(self, t, lr, batch_data):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, lr, batch_data)
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save(t)

        return loss_eval, grad_eval

    def update_step(self, t, lr, batch_data):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        fd = {
            # inputs
            self.s: batch_data[0],
            self.pred_flag: batch_data[1],
            self.target_action: batch_data[2],
            self.slen: batch_data[3],
            self.hs: DNC.zero_state(self.config, self.config.batch_size),
            self.lr: lr, 
            # extra info
            self.eval_acc_placeholder: self.eval_acc
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval

    def train(self, beta_schedule, lr_schedule, cr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        self.init_averages()

        t = last_eval = curri_idx = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        curriculum_batch_size = np.ceil(self.config.nsteps_train/cr_schedule.n_curriculum).astype('int32')

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            t += 1
            last_eval += 1
            config = self.config
            config.n_node, config.k_ring, config.p_rewiring, config.path_len_limit = cr_schedule[curri_idx]
            self.env.reset(config) # h x w x c
            h_state = DNC.zero_state(config, batch_size=1)
            encoding, predflag, target_action = self.env.prepare_seq()
            slen = np.array(encoding.shape[0]).astype('int32')
            # describe graph, query and planning
            h_state = self.sess.run(self.hs_out, feed_dict={self.s: encoding[None], self.hs: h_state, self.slen: slen})
            past_state = -1
            past_action_onehot = -1
            encoding_a = np.zeros([config.max_step_len, encoding.shape[1]])
            predflag_a = np.zeros(config.max_step_len)
            target_action_a = np.zeros([config.max_step_len, target_action.shape[1]])
            if self.config.use_transition_only_during_answering:
                current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[-1, -1, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
            else:
                #current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[past_state, self.env.current_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[self.env.current_state, self.env.target_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
            current_encoding = np.concatenate([current_encoding, np.array([[1,1]])], axis=1)
            for i in range(config.max_step_len):
                pred_action, h_state = self.sess.run([self.q, self.hs_out], feed_dict={self.s: current_encoding[None], self.hs: h_state, self.slen: np.ones(1).astype('int32')})
                gt_action = self.env.get_gt_action()
                action = self.get_action(pred_action[0], gt_action, beta_schedule.epsilon)
                past_state = self.env.current_state
                _, done, past_action_onehot = self.env.step(action)
                encoding_a[i,:] = current_encoding[0]
                predflag_a[i] = 1
                target_action_a[i] = gt_action
                slen += 1
                if done:
                    break
                if self.config.use_transition_only_during_answering:
                    current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[-1, -1, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                else:
                    #current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[past_state, self.env.current_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                    current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[self.env.current_state, self.env.target_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                current_encoding = np.concatenate([current_encoding, np.array([[0,1]])], axis=1)

            batch_data = (np.concatenate([encoding, encoding_a], axis=0)[None],
                np.concatenate([predflag, predflag_a], axis=0),
                np.concatenate([target_action, target_action_a], axis=0), slen)

            # perform a training step
            loss_eval, grad_eval = self.train_step(t, lr_schedule.epsilon, batch_data)

            # logging stuff
            if ((t % config.log_freq == 0) and (t % config.learning_freq == 0)):
                self.update_averages(scores_eval)
                beta_schedule.update(t)
                lr_schedule.update(t)
                prog.update(t + 1, exact=[("Loss", loss_eval), ("Grads", grad_eval), ("lr", lr_schedule.epsilon)])

            if t >= config.nsteps_train:
                break

            if last_eval >= config.eval_freq:
                # evaluate our policy
                last_eval = 0
                print("")
                self.logger.info("Global step: %d"%(t))
                scores_eval += [self.evaluate(cr_schedule, curri_idx)]
                if scores_eval[-1]>0.8:
                    curri_idx += 1
                    msg = "Upgrade to lesson {:d}".format(int(curri_idx))
                    self.logger.info(msg)
                    self.logger.info("----------Start Computing Final Score----------")
                    scores_eval += [self.evaluate(cr_schedule)]
                    self.logger.info("----------Finish Computing Final Score----------")

        # last words
        self.logger.info("- Training done.")
        self.save(t)
        scores_eval += [self.evaluate(cr_schedule)]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def evaluate(self, cr_schedule, curri_idx=None, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        if curri_idx is None:
            curri_idx = -1
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        accs = []
        gt_len = []

        for i in range(num_episodes):
            config = self.config
            config.n_node, config.k_ring, config.p_rewiring, config.path_len_limit = cr_schedule[curri_idx]
            env.reset(config) # h x w x c
            h_state = DNC.zero_state(config, batch_size=1)
            encoding, predflag, target_action = env.prepare_seq()
            slen = np.array(encoding.shape[0]).astype('int32')
            # describe graph, query and planning
            h_state = self.sess.run(self.hs_out, feed_dict={self.s: encoding[None], self.hs: h_state, self.slen: slen})
            past_state = -1
            past_action_onehot = -1
            path_len = 0
            if self.config.use_transition_only_during_answering:
                current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[-1, -1, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
            else:
                #current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[past_state, env.current_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[env.current_state, env.target_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
            current_encoding = np.concatenate([current_encoding, np.array([[1,1]])], axis=1)
            for i in range(config.max_step_len):
                pred_action, h_state = self.sess.run([self.q, self.hs_out], feed_dict={self.s: current_encoding[None], self.hs: h_state, self.slen: np.ones(1).astype('int32')})
                past_state = env.current_state
                _, done, past_action_onehot = env.step(pred_action[0])
                path_len += 1
                if done:
                    break
                if self.config.use_transition_only_during_answering:
                    current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[-1, -1, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                else:
                    #current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[past_state, env.current_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                    current_encoding = GraphWorld.convert_triplets_to_encoding(np.array([[env.current_state, env.target_state, past_action_onehot]]).astype('int32'), config.ndigits, config.nway)
                current_encoding = np.concatenate([current_encoding, np.array([[0,1]])], axis=1)
            accs.append(len(env.path[env.src_state])==path_len and env.current_state==env.target_state)
            gt_len.append(len(env.path[env.src_state]))

        avg_acc = np.mean(accs)
        if num_episodes > 1:
            msg = "Average acc: {:04.2f}".format(avg_acc)
            self.logger.info(msg)
        return avg_acc


    def run(self, beta_schedule, lr_schedule, cr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()
        # model
        self.train(beta_schedule, lr_schedule, cr_schedule)

    def deploy(self, cr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()
        # model
        self.evaluate(cr_schedule, -1)  


if __name__ == '__main__':
    # make env
    env = GraphWorld(config)

    # exploration strategy
    beta_schedule  = ExpSchedule(config.beta_begin, config.beta_end,
            config.beta_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    cr_schedule = CurriculumSchedule()

    # train model
    model = DRQNwMemGraphWorld(env, config)
    if config.deploy_only:
        model.deploy(cr_schedule)
    else:
        model.run(beta_schedule, lr_schedule, cr_schedule)
