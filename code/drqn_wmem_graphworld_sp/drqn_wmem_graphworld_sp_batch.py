import os
import numpy as np
import logging
import time
import sys
import collections
from graph_world_sp import GraphWorld
from utils.general import get_logger, Progbar, export_plot
from utils.schedule import LinearSchedule, ExpSchedule, CurriculumSchedule
from utils.dnc import DNC
import tensorflow as tf
import tensorflow.contrib.layers as layers
#from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from configs.drqn_wmem_graph_world_sp import config

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations','target', 'mask','seqlen'))

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

        self.s = tf.placeholder(tf.float32, shape=(None, None, self.config.ndigits*self.config.nway*3+2))
        self.hs = DNC.state_placeholder(self.config)
        self.slen = tf.placeholder(tf.int32, shape=(None))
        self.pred_flag = tf.placeholder(tf.float32, shape=(None, None)) # (nb, state_history)
        self.target_action = tf.placeholder(tf.float32, shape=(None, None, self.config.ndigits*self.config.nway*3)) # (nb, state_history, num_actions)
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
        dnc_h_size = self.config.dnc_h_size
        out = state

        with tf.variable_scope(scope, reuse = reuse):
            dnc_cell = DNC(config=self.config, output_size=dnc_h_size, clip_value=self.config.dnc_clip_val)

            #out = tf.reshape(out, shape=[-1, self.config.ndigits*self.config.nway*3+2])
            #out = layers.fully_connected(out, 32, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #out = layers.fully_connected(out, 32, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #out = tf.reshape(out, shape=[-1, tf.shape(state)[1], 32])

            out, h_state_out = tf.nn.dynamic_rnn(inputs=out, cell=dnc_cell, sequence_length=seq_len, dtype=tf.float32, initial_state=h_state)
            #### out has size (nb, seq_len, dnc_h_size) #####

            out = layers.fully_connected(out, 32, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = layers.fully_connected(out, self.config.ndigits*self.config.nway*3, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #### out has size (nb, seq_len, self.config.ndigits*self.config.nway*3) #####
        return out, h_state_out


    def add_loss_op(self, pred_action, target_action, pred_flag):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            pred_action: (tf tensor) shape = (batch_size, state_history, num_actions)
            target_action: (tf tensor) shape = (batch_size, state_history, num_actions)
            pred_flag: (tf tensor) shape = (batch_size, state_history)
        """
        batch_size = tf.shape(pred_action)[0]
        state_history = tf.shape(pred_action)[1]
        pred_action = tf.reshape(pred_action, shape=[batch_size, state_history, self.config.ndigits*3, self.config.nway])
        target_action = tf.reshape(target_action, shape=[batch_size, state_history, self.config.ndigits*3, self.config.nway])
        smx = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_action, logits=pred_action), axis=2)
        smx = tf.reduce_sum(smx*pred_flag, axis=1)
        self.loss = tf.reduce_mean(smx)

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
        batch_size = tf.shape(self.q_logits)[0]
        state_history = tf.shape(self.q_logits)[1]
        trip_len = tf.shape(self.q_logits)[2]
        self.q = tf.nn.softmax(tf.reshape(self.q_logits, shape=[batch_size, state_history, self.config.ndigits*3, self.config.nway]))
        self.q = tf.one_hot(tf.argmax(self.q, axis=3), depth=self.config.nway)
        self.q = tf.reshape(self.q, shape=(batch_size, state_history, trip_len))

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
        configtmp = tf.ConfigProto(allow_soft_placement=True)
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
            self.s: batch_data.observations,
            self.target_action: batch_data.target,
            self.pred_flag: batch_data.mask,
            self.slen: batch_data.seqlen,
            self.hs: DNC.zero_state(self.config, self.config.batch_size),
            self.lr: lr, 
            # extra info
            self.eval_acc_placeholder: self.eval_acc
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)

        '''
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        '''
        
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

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            t += 1
            last_eval += 1
            encoding_batch = []
            predflag_batch = []
            target_action_batch = []
            slen_batch = []
            max_len = 0
            for i in range(self.config.batch_size):
                config = self.config
                config.n_node, config.k_ring, config.p_rewiring, config.path_len_limit, config.planning_len = cr_schedule[curri_idx]
                self.env.reset(config) # h x w x c
                encoding, target_action, predflag = self.env.prepare_seq()
                encoding_batch.append(encoding[None])
                predflag_batch.append(predflag[None])
                target_action_batch.append(target_action[None])
                slen_batch.append(encoding.shape[0])
                if encoding.shape[0]>max_len:
                    max_len = encoding.shape[0]

            batch_data = DatasetTensors(np.concatenate([np.concatenate([x, np.zeros([1, max_len-x.shape[1], x.shape[2]])], axis=1) for x in encoding_batch], axis=0),
                np.concatenate([np.concatenate([x, np.zeros([1, max_len-x.shape[1], x.shape[2]])], axis=1) for x in target_action_batch], axis=0),
                np.concatenate([np.concatenate([x, np.zeros([1, max_len-x.shape[1]])], axis=1) for x in predflag_batch], axis=0), np.array(slen_batch).astype('int32'))

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

        for i in range(num_episodes):
            encoding_batch = []
            predflag_batch = []
            target_action_batch = []
            slen_batch = []
            max_len = 0
            for j in range(self.config.batch_size):
                config = self.config
                config.n_node, config.k_ring, config.p_rewiring, config.path_len_limit, config.planning_len = cr_schedule[curri_idx]
                self.env.reset(config) # h x w x c
                encoding, target_action, predflag = self.env.prepare_seq()
                encoding_batch.append(encoding[None])
                predflag_batch.append(predflag[None])
                target_action_batch.append(target_action[None])
                slen_batch.append(encoding.shape[0])
                if encoding.shape[0]>max_len:
                    max_len = encoding.shape[0]

            batch_data = DatasetTensors(np.concatenate([np.concatenate([x, np.zeros([1, max_len-x.shape[1], x.shape[2]])], axis=1) for x in encoding_batch], axis=0),
                np.concatenate([np.concatenate([x, np.zeros([1, max_len-x.shape[1], x.shape[2]])], axis=1) for x in target_action_batch], axis=0),
                np.concatenate([np.concatenate([x, np.zeros([1, max_len-x.shape[1]])], axis=1) for x in predflag_batch], axis=0), np.array(slen_batch).astype('int32'))

            h_state = DNC.zero_state(config, batch_size=self.config.batch_size)
            pred_action, h_state = self.sess.run([self.q, self.hs_out], feed_dict={self.s: batch_data.observations, self.hs: h_state, self.slen: batch_data.seqlen})
            for j in range(self.config.batch_size):
                accs.append((pred_action[j]*np.expand_dims(batch_data.mask[j],1) == batch_data.target[j]*np.expand_dims(batch_data.mask[j],1)).reshape(-1).all())

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
        self.evaluate(cr_schedule, 0)  


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
