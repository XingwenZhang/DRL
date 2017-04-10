""" Tensor Flow Networks
functions for creating different network architectures.
"""

import tensorflow as tf
import TFUtil
import DQNConfig


def build_dqn(num_action, dueling_dqn=False):
    with tf.variable_scope('DQN'):
        # Prediction Network
        with tf.variable_scope('Prediction'):
            pn_variable_dict = {}
            # inference
            pn_states = tf.placeholder(name='input', shape=(None, 84, 84, 4), dtype=tf.float32)
            pn_conv1 = TFUtil.conv_layer('conv1', pn_states, shape=[8, 8, 4, 32], stride=4, variable_dict=pn_variable_dict)
            pn_conv2 = TFUtil.conv_layer('conv2', pn_conv1, shape=[4, 4, 32, 64], stride=2, variable_dict=pn_variable_dict)
            pn_conv3 = TFUtil.conv_layer('conv3', pn_conv2, shape=[3, 3, 64, 64], stride=1, variable_dict=pn_variable_dict)
            pn_conv3_flatten = TFUtil.flatten(pn_conv3, feature_length=(7*7*64))
            if dueling_dqn:
                pn_fc4_a = TFUtil.fc_layer('fc4_a', pn_conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=pn_variable_dict) 
                pn_value = TFUtil.fc_layer('value', pn_fc4_a, input_size=512, num_neron=1, activation=None, variable_dict=pn_variable_dict)
                pn_fc4_b = TFUtil.fc_layer('fc4_b', pn_conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=pn_variable_dict)
                pn_advantage = TFUtil.fc_layer('advantage', pn_fc4_b, input_size=512, num_neron=num_action, activation=None, variable_dict=pn_variable_dict)
                pn_Q = (pn_advantage - tf.reshape(tf.reduce_mean(pn_advantage, axis=1), (-1,1))) + tf.reshape(pn_value, (-1,1))
            else:
                pn_fc4 = TFUtil.fc_layer('fc4', pn_conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=pn_variable_dict)
                pn_Q = TFUtil.fc_layer('Q', pn_fc4, input_size=512, num_neron=num_action, activation=None, variable_dict=pn_variable_dict)

            # loss
            pn_q_target = tf.placeholder(name='q_target', shape=(None,), dtype=tf.float32)
            pn_actions = tf.placeholder(name='action', shape=(None,), dtype=tf.int32)
            pn_actions_one_hot = tf.one_hot(pn_actions, depth=num_action)
            pn_delta = tf.reduce_sum(pn_actions_one_hot * pn_Q, axis=1) - pn_q_target
            pn_loss = tf.reduce_sum(TFUtil.huber_loss(pn_delta)) / DQNConfig.batch_size

            # summary
            tf.summary.scalar('pn_loss', pn_loss)
            tf.summary.scalar('averaged pn_Q', tf.reduce_mean(pn_Q))

            # optimizer
            pn_train = tf.train.RMSPropOptimizer(learning_rate=DQNConfig.lr).minimize(pn_loss)

        # Target Network
        with tf.variable_scope('Target'):
            tn_variable_dict = {}
            # inference
            tn_states = tf.placeholder(name='input', shape=(None, 84, 84, 4), dtype=tf.float32)
            tn_conv1 = TFUtil.conv_layer('conv1', tn_states, shape=[8, 8, 4, 32], stride=4, variable_dict=tn_variable_dict)
            tn_conv2 = TFUtil.conv_layer('conv2', tn_conv1, shape=[4, 4, 32, 64], stride=2, variable_dict=tn_variable_dict)
            tn_conv3 = TFUtil.conv_layer('conv3', tn_conv2, shape=[3, 3, 64, 64], stride=1, variable_dict=tn_variable_dict)
            tn_conv3_flatten = TFUtil.flatten(tn_conv3, feature_length=(7 * 7 * 64))
            if dueling_dqn:
                tn_fc4_a = TFUtil.fc_layer('fc4_a', tn_conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=tn_variable_dict) 
                tn_value = TFUtil.fc_layer('value', tn_fc4_a, input_size=512, num_neron=1, activation=None, variable_dict=tn_variable_dict)
                tn_fc4_b = TFUtil.fc_layer('fc4_b', tn_conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=tn_variable_dict)
                tn_advantage = TFUtil.fc_layer('advantage', tn_fc4_b, input_size=512, num_neron=num_action, activation=None, variable_dict=tn_variable_dict)
                tn_Q = (tn_advantage - tf.reshape(tf.reduce_mean(tn_advantage, axis=1), (-1,1))) + tf.reshape(tn_value, (-1,1))
            else:
                tn_fc4 = TFUtil.fc_layer('fc4', tn_conv3_flatten, input_size=(7*7*64), num_neron=512, variable_dict=tn_variable_dict)
                tn_Q = TFUtil.fc_layer('Q', tn_fc4, input_size=512, num_neron=num_action, activation=None, variable_dict=tn_variable_dict)

        # Network Cloning
        with tf.variable_scope('Prediction_to_Target'):
            network_cloning_ops = []
            assert (tn_variable_dict.keys() == pn_variable_dict.keys())
            for k in tn_variable_dict.keys():
                network_cloning_ops.append(tf.assign(tn_variable_dict[k], pn_variable_dict[k]))
    
    return (pn_states, pn_Q, pn_loss, pn_actions, pn_q_target, pn_train), (tn_states, tn_Q), network_cloning_ops


