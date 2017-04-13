""" Tensor Flow Networks
functions for creating different network architectures.
"""

import tensorflow as tf
import TFUtil
import ACConfig
import math


def build_actor_critic_network(num_action):
    with tf.variable_scope('actor_critic_network'):
        # Inputs
        with tf.name_scope('inputs'):
            state_placeholder = tf.placeholder(
                name='state', 
                shape=(None, 84, 84, 4), 
                dtype=tf.float32)
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

        # Main network
        with tf.variable_scope('shared_network'):
            variable_dict = {}
            # inference
            conv1 = TFUtil.conv_layer('conv1', state_placeholder, shape=[8, 8, 4, 32], stride=4, variable_dict=variable_dict)
            conv2 = TFUtil.conv_layer('conv2', conv1, shape=[4, 4, 32, 64], stride=2, variable_dict=variable_dict)
            conv3 = TFUtil.conv_layer('conv3', conv2, shape=[3, 3, 64, 64], stride=1, variable_dict=variable_dict)
            conv3_flatten = TFUtil.flatten(conv3, feature_length=(7*7*64))
        
        # outputs
        with tf.variable_scope('actor_network'):
            fc4_actor = TFUtil.fc_layer('fc4_actor', conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=variable_dict) 
            actor_logits = TFUtil.fc_layer('logits', fc4_actor, input_size=512, num_neron=num_action, activation=None, variable_dict=variable_dict)
            policy_probs = tf.nn.softmax(name = 'policy_probs', logits = actor_logits)

        with tf.variable_scope('critic_network'):
            fc4_critic = TFUtil.fc_layer('fc4_critic', conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=variable_dict) 
            state_value = tf.squeeze(TFUtil.fc_layer('value', fc4_critic, input_size=512, num_neron=1, activation=None, variable_dict=variable_dict), axis = 1)

        with tf.variable_scope('loss'):
            # policy loss
            log_prob = - tf.nn.sparse_softmax_cross_entropy_with_logits(
                name   = 'log_prob',
                labels = action_placeholder,
                logits = actor_logits)
            policy_loss = - tf.reduce_sum(log_prob * advantage_placeholder) / ACConfig.batch_size
            policy_entropy = - tf.reduce_sum(policy_probs * tf.log(policy_probs + 1e-15)) / ACConfig.batch_size
            # value_loss
            value_loss = tf.reduce_sum(tf.square(q_value_placeholder - state_value)) / ACConfig.batch_size
            # need to tweak weight
            loss = policy_loss + 0.5 * value_loss - 0.0005 * policy_entropy
            
        # train_op
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
           (train_op, sample_action), (actor_logits, state_value, average_reward)

