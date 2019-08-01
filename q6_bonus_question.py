# Assignment 2 - Stanford CS234 Reinforcement Learning
# Deepmind Deep Q Network with Experience Replay
# Team members: Yann BOUTEILLER, Amine BELLAHSEN
# Question 6 - Bonus : implementing a Dueling architecture
# ------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear
from configs.q6_bonus_question import config
import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from q1_schedule import LinearExploration, LinearSchedule


class DuelingQN(Linear):
    """
    Dueling Q network
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        with tf.variable_scope(scope, reuse=reuse):

            output = layers.conv2d(inputs=state,
            num_outputs=32,
            kernel_size=8,
            stride=4,
            padding='SAME',
            activation_fn=tf.nn.relu)

            output = layers.conv2d(inputs=output,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.relu)

            output = layers.conv2d(inputs=output,
            num_outputs=64,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu)

            output = layers.flatten(inputs=output)

            output_v = layers.fully_connected(inputs=output,
            num_outputs=512,
            activation_fn=tf.nn.relu)

            output_v = layers.fully_connected(inputs=output_v,
            num_outputs=1,
            activation_fn=None)

            output_a = layers.fully_connected(inputs=output,
            num_outputs=512,
            activation_fn=tf.nn.relu)

            output_a = layers.fully_connected(inputs=output_a,
            num_outputs=num_actions,
            activation_fn=None)

            q_vals = output_a - tf.tile(tf.expand_dims(tf.reduce_mean(output_a, axis=1),-1), [1,num_actions]) + tf.tile(output_v, [1,num_actions])
        return q_vals


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = DuelingQN(env, config)
    model.run(exp_schedule, lr_schedule)
