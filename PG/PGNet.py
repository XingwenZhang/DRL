""" Tensor Flow Networks
functions for creating different network architectures.
"""

import tensorflow as tf
import TFUtil
import PGConfig
import math


def build_pn(num_action):
    # Prediction Network
    with tf.variable_scope('policy_network'):
        # input
        with tf.name_scope('inputs'):
            state_placeholder = tf.placeholder(
                name  = 'state',
                shape = (None, PGConfig.frame_size), 
                dtype = tf.float32)
            action_placeholder = tf.placeholder(
                name  = 'taken_action', 
                shape = (None, ), 
                dtype = tf.int32)
            advantage_placeholder = tf.placeholder(
                name  = 'advantage',
                shape = (None, ),
                dtype = tf.float32)

        # hid1
        with tf.variable_scope('hid1'):
            weights = tf.Variable(
                tf.truncated_normal([PGConfig.frame_size, PGConfig.num_hid],
                stddev=1.0 / math.sqrt(float(PGConfig.frame_size))),
                name='weights')
            biases = tf.Variable(
                tf.zeros([PGConfig.num_hid]),
                name='biases')
            hid = tf.nn.relu(tf.matmul(state_placeholder, weights) + biases)

        # softmax_logits
        with tf.variable_scope('logits'):
            weights = tf.Variable(
                tf.truncated_normal([PGConfig.num_hid, num_action],
                stddev=1.0 / math.sqrt(float(PGConfig.num_hid))),
                name='weights')
            biases = tf.Variable(
                tf.zeros([num_action]),
                name='biases')
            logits = tf.matmul(hid, weights) + biases

        # compute policy gradient
        with tf.name_scope('policy_gradient'):
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                name   = 'log_prob',
                labels = action_placeholder,
                logits = logits)
            network_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="policy_network")
            policy_gradient = zip(tf.gradients(log_prob, network_variables, advantage_placeholder), network_variables)
            loss = tf.reduce_mean(advantage_placeholder * log_prob)
        
        # train_op
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate = PGConfig.lr,
            decay = PGConfig.decay)
        train_op = optimizer.apply_gradients(policy_gradient)

        # sample_action
        sample_action = tf.multinomial(logits, 1)

        # reward_history
        with tf.name_scope('reward_history'):
            reward_history_placeholder = tf.placeholder(
                name = 'reward_history',
                shape = (None, ),
                dtype = tf.float32)
            average_reward = tf.reduce_mean(reward_history_placeholder)

        # summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('average_reward_over_100_episodes', average_reward)

    return (state_placeholder, action_placeholder, advantage_placeholder, reward_history_placeholder), \
           (train_op, sample_action), (loss, average_reward)

