from xworld import xworld_navi_goal, xworld_args

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.schedule import LinearExploration, LinearSchedule
from core.dqn import DQN

from configs.dqn_xworld import config


class FC(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.
        """
        # here, typically, a state shape is (5,3,22)

        state_shape = list([self.config.state_history, self.env.args.visible_radius_unit_front+1, 2*self.env.args.visible_radius_unit_side+1, len(self.env.state.xmap.item_class_id)+1])
        self.s = tf.placeholder(tf.bool, shape=(None, state_shape[0], state_shape[1], state_shape[2], state_shape[3]))
        self.sp = tf.placeholder(tf.bool, shape=(None, state_shape[0], state_shape[1], state_shape[2], state_shape[3]))

        #self.s = tf.placeholder(tf.int32, shape=(None, 45))
        #self.sp = tf.placeholder(tf.int32, shape=(None, 45))

        self.a = tf.placeholder(tf.int32, shape=(None))
        self.r = tf.placeholder(tf.float32, shape=(None))
        self.done_mask = tf.placeholder(tf.bool, shape=(None))
        self.lr = tf.placeholder(tf.float32, shape=(None))


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img_w, img_h, nchannel)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.agent.num_actions
        out = state

        with tf.variable_scope(scope, reuse = reuse):
            out = layers.flatten(out)
            out = layers.fully_connected(out, 200, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = layers.fully_connected(out, 100, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #out = layers.fully_connected(out, 100, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            #out = layers.fully_connected(out, 50, activation_fn = tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())
            out = layers.fully_connected(out, num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.zeros_initializer())

        return out


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
        self.loss = tf.reduce_mean(tf.square(q_pred-q_sample))


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


if __name__ == '__main__':
    # make env
    args = xworld_args.parser().parse_args()
    args.visible_radius_unit_side = config.visible_radius_unit_side
    args.visible_radius_unit_front = config.visible_radius_unit_front
    args.ego_centric = config.ego_centric
    args.map_config = config.map_config_file
    env = xworld_navi_goal.XWorldNaviGoal(args)
    env.teacher.israndom_goal = False
    env.teacher.goal_id = 0

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = FC(env, config)
    model.run(exp_schedule, lr_schedule)
