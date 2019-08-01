# Assignment 2 - Stanford CS234 Reinforcement Learning
# Deepmind Deep Q Network with Experience Replay
# Team members: Yann BOUTEILLER, Amine BELLAHSEN
# Question 2
# ------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule
from configs.q2_linear import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        state_shape = list(self.env.observation_space.shape)

        self.s = tf.placeholder(tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history))
        self.a = tf.placeholder(tf.int32, shape=(None, ))
        self.r = tf.placeholder(tf.float32, shape=(None, ))
        self.sp = tf.placeholder(tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history))
        self.done_mask = tf.placeholder(tf.bool, shape=(None, ))
        self.lr = tf.placeholder(tf.float32, shape=())


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        with tf.variable_scope(scope, reuse=reuse):
            return layers.fully_connected(layers.flatten(state), num_actions, activation_fn=None)


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network
        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope
        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        q_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        self.update_target_op = tf.group(*[tf.assign(ref=target_q_variables[i], value=q_variables[i]) for i in range(len(q_variables))])


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        q_samp = tf.where(self.done_mask, self.r, self.r + self.config.gamma * tf.reduce_max(target_q, axis=1))
        q_val = tf.reduce_sum(tf.multiply(tf.one_hot(self.a, num_actions), q), axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_val, q_samp))


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        grads_vars = optimizer.compute_gradients(loss=self.loss, var_list=variables)
        if self.config.grad_clip: # clip by global norm
            grads_vars = [(tf.clip_by_norm(grad, self.config.clip_val), var) if grad is not None else (grad, var) for grad, var in grads_vars]

        self.train_op = optimizer.apply_gradients(grads_and_vars=grads_vars)
        self.grad_norm = tf.global_norm([grad for grad, _ in grads_vars]) # we log the clipped norm here


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))
    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
