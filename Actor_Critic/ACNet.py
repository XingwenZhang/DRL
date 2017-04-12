""" Tensor Flow Networks
functions for creating different network architectures.
"""

import tensorflow as tf
import TFUtil
import ACConfig
import math


def build_actor_critic_network(num_action):
    # Prediction Network
    with tf.variable_scope('actor_critic_network'):
        # input
        with tf.name_scope('inputs'):
            state_placeholder = tf.placeholder(
                name  = 'state',
                shape = (None, ACConfig.frame_size), 
                dtype = tf.float32)
            action_placeholder = tf.placeholder(
                name  = 'taken_action', 
                shape = (None, ), 
                dtype = tf.int32)
            q_value_placeholder = tf.placeholder(
                name  = 'q_value',
                shape = (None, ),
                dtype = tf.float32)
            advantage_placeholder = tf.placeholder(
                name  = 'advantage',
                shape = (None, ),
                dtype = tf.float32)

        # hid1
        with tf.variable_scope('hid1'):
            weights = tf.Variable(
                tf.truncated_normal([ACConfig.frame_size, ACConfig.num_hid],
                stddev=1.0 / math.sqrt(float(ACConfig.frame_size))),
                name='weights')
            biases = tf.Variable(
                tf.zeros([ACConfig.num_hid]),
                name='biases')
            hid = tf.nn.relu(tf.matmul(state_placeholder, weights) + biases)

        # actor logits
        with tf.variable_scope('actor_logits'):
            weights = tf.Variable(
                tf.truncated_normal([ACConfig.num_hid, num_action],
                stddev=1.0 / math.sqrt(float(ACConfig.num_hid))),
                name='weights')
            biases = tf.Variable(
                tf.zeros([num_action]),
                name='biases')
            actor_logits = tf.matmul(hid, weights) + biases
            actor_probs = tf.nn.softmax(logits = actor_logits)

        # critic (state) value
        with tf.variable_scope('critic_value'):
            weights = tf.Variable(
                tf.truncated_normal([ACConfig.num_hid, 1],
                stddev=1.0 / math.sqrt(float(ACConfig.num_hid))),
                name='weights')
            biases = tf.Variable(
                tf.zeros(1),
                name='biases')
            critic_value = tf.reshape(tf.matmul(hid, weights) + biases, (-1, ))
            
        # compute loss
        with tf.name_scope('loss'):
            # policy loss
            log_prob = - tf.nn.sparse_softmax_cross_entropy_with_logits(
                name   = 'log_prob',
                labels = action_placeholder,
                logits = actor_logits)
            policy_loss = - tf.reduce_sum(log_prob * advantage_placeholder) / ACConfig.batch_size
            policy_entropy = - tf.reduce_sum(actor_probs * tf.log(actor_probs + 1e-15)) / ACConfig.batch_size
            # value_loss
            value_loss = tf.reduce_sum(tf.square(q_value_placeholder - critic_value)) / ACConfig.batch_size
            # need to tweak weight
            loss = policy_loss + 0.5 * value_loss - 0.005 * policy_entropy
            
    # train_op
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate = ACConfig.lr,
        decay = ACConfig.decay)
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate = ACConfig.lr)
    
    """
    grad_var = optimizer.compute_gradients(loss)
    clipped_grad_var = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_var]
    train_op = optimizer.apply_gradients(clipped_grad_var)
    """
    train_op = optimizer.minimize(loss)

    # sample_action
    sample_action = tf.multinomial(actor_logits, 1)

    # reward_history
    with tf.name_scope('reward_history'):
        reward_history_placeholder = tf.placeholder(
            name = 'reward_history',
            shape = (None, ),
            dtype = tf.float32)
        average_reward = tf.reduce_mean(reward_history_placeholder)

    # summary
    with tf.name_scope('summary'):
        tf.summary.scalar('average_reward_over_100_episodes', average_reward)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('policy_entropy', policy_entropy)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.scalar('loss', loss)

    return (state_placeholder, action_placeholder, q_value_placeholder, advantage_placeholder, reward_history_placeholder), \
           (train_op, sample_action), (critic_value, average_reward)

