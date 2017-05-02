""" THORNet
functions for creating NN for the THOR environment
"""
import sys
import tensorflow as tf
import TFUtil
import DQNConfig as config
import math


def build_network(scope, num_action, dueling_dqn):
    with tf.variable_scope(scope):
        with tf.variable_scope('Prediction'):
            with tf.name_scope('inputs'):
                # resnet feature
                pn_state_placeholder = tf.placeholder(
                    name  = 'state', 
                    shape = (None, config.num_history_frames, 2048), # (n, 4, 2048)
                    dtype = tf.float32)
                # target feature
                pn_target_placeholder = tf.placeholder(
                    name = 'target',
                    shape = (None, config.num_history_frames, 2048), # (n, 4, 2048)
                    dtype = tf.float32)
                pn_q_target = tf.placeholder(
                    name='q_target', 
                    shape=(None,), 
                    dtype=tf.float32)
                pn_actions = tf.placeholder(
                    name='action', 
                    shape=(None,), 
                    dtype=tf.int32)
            
            # compute embedded feature given the input image feature
            pn_variable_dict = {}
            with tf.variable_scope('shared_layers') as scope:
                # fc1 
                state_flattened = tf.reshape(pn_state_placeholder, (-1, config.num_history_frames * 2048)) # (n, 4 * 2048)
                fc1_state = TFUtil.fc_layer('fc1', state_flattened, input_size=config.num_history_frames * 2048, num_neron=512, variable_dict=pn_variable_dict) # (n, 512)
                scope.reuse_variables()
                target_flattened = tf.reshape(pn_target_placeholder, (-1, config.num_history_frames * 2048)) # (n, 4 * 2048)
                fc1_target = TFUtil.fc_layer('fc1', target_flattened, input_size=config.num_history_frames * 2048, num_neron=512, variable_dict=pn_variable_dict) # (n, 512)
            with tf.variable_scope('shared_layers'):
                # fc2:
                fc2 = TFUtil.fc_layer('fc2',
                                      tf.concat((fc1_state, fc1_target), axis = 1), # (n, 1024)
                                      input_size=1024, num_neron=512, variable_dict=pn_variable_dict)  # (n, 512)

            # output: copied and modified from DQNNet.py
            if dueling_dqn:
                pn_fc4_a = TFUtil.fc_layer('fc4_a', fc2, input_size=(512), num_neron=512, variable_dict=pn_variable_dict) 
                pn_value = TFUtil.fc_layer('value', pn_fc4_a, input_size=512, num_neron=1, activation=None, variable_dict=pn_variable_dict)
                pn_fc4_b = TFUtil.fc_layer('fc4_b', fc2, input_size=(512), num_neron=512, variable_dict=pn_variable_dict)
                pn_advantage = TFUtil.fc_layer('advantage', pn_fc4_b, input_size=512, num_neron=num_action, activation=None, variable_dict=pn_variable_dict)
                pn_Q = (pn_advantage - tf.reshape(tf.reduce_mean(pn_advantage, axis=1), (-1,1))) + tf.reshape(pn_value, (-1,1))
            else:
                pn_fc4 = TFUtil.fc_layer('fc4', fc2, input_size=(512), num_neron=512, variable_dict=pn_variable_dict)
                pn_Q = TFUtil.fc_layer('Q', pn_fc4, input_size=512, num_neron=num_action, activation=None, variable_dict=pn_variable_dict)

            # loss
            pn_actions_one_hot = tf.one_hot(pn_actions, depth=num_action)
            pn_delta = tf.reduce_sum(pn_actions_one_hot * pn_Q, axis=1) - pn_q_target
            
            pn_importance_weight = tf.placeholder(name = 'importance_weight', shape = (None), dtype = tf.float32)
            pn_weighted_delta = tf.multiply(pn_delta, pn_importance_weight)
            
            pn_loss = tf.reduce_sum(TFUtil.huber_loss(pn_delta)) / config.batch_size

            # summary
            summary_pn_loss = tf.summary.scalar('pn_loss', pn_loss)
            summary_averaged_pn_Q = tf.summary.scalar('averaged_pn_Q', tf.reduce_mean(pn_Q))

            # optimizer
            pn_train = tf.train.RMSPropOptimizer(learning_rate=config.lr).minimize(pn_loss)

        with tf.variable_scope('Target'):
            with tf.name_scope('inputs'):
                # resnet feature
                tn_state_placeholder = tf.placeholder(
                    name  = 'state', 
                    shape = (None, config.num_history_frames, 2048), # (n, 4, 2048)
                    dtype = tf.float32)
                # target feature
                tn_target_placeholder = tf.placeholder(
                    name = 'target',
                    shape = (None, config.num_history_frames, 2048), # (n, 4, 2048)
                    dtype = tf.float32)
            
            # compute embedded feature given the input image feature
            tn_variable_dict = {}
            with tf.variable_scope('shared_layers') as scope:
                # fc1 
                state_flattened = tf.reshape(tn_state_placeholder, (-1, config.num_history_frames * 2048)) # (n, 4 * 2048)
                fc1_state = TFUtil.fc_layer('fc1', state_flattened, input_size=config.num_history_frames * 2048, num_neron=512, variable_dict=tn_variable_dict) # (n, 512)
                scope.reuse_variables()
                target_flattened = tf.reshape(tn_target_placeholder, (-1, config.num_history_frames * 2048)) # (n, 4 * 2048)
                fc1_target = TFUtil.fc_layer('fc1', target_flattened, input_size=config.num_history_frames * 2048, num_neron=512, variable_dict=tn_variable_dict) # (n, 512)
            with tf.variable_scope('shared_layers'):
                # fc2:
                fc2 = TFUtil.fc_layer('fc2',
                                      tf.concat((fc1_state, fc1_target), axis = 1), # (n, 1024)
                                      input_size=1024, num_neron=512, variable_dict=tn_variable_dict)  # (n, 512)

            # output: copied and modified from DQNNet.py
            if dueling_dqn:
                tn_fc4_a = TFUtil.fc_layer('fc4_a', fc2, input_size=(512), num_neron=512, variable_dict=tn_variable_dict) 
                tn_value = TFUtil.fc_layer('value', tn_fc4_a, input_size=512, num_neron=1, activation=None, variable_dict=tn_variable_dict)
                tn_fc4_b = TFUtil.fc_layer('fc4_b', fc2, input_size=(512), num_neron=512, variable_dict=tn_variable_dict)
                tn_advantage = TFUtil.fc_layer('advantage', tn_fc4_b, input_size=512, num_neron=num_action, activation=None, variable_dict=tn_variable_dict)
                tn_Q = (tn_advantage - tf.reshape(tf.reduce_mean(tn_advantage, axis=1), (-1,1))) + tf.reshape(tn_value, (-1,1))
            else:
                tn_fc4 = TFUtil.fc_layer('fc4', fc2, input_size=(512), num_neron=512, variable_dict=tn_variable_dict)
                tn_Q = TFUtil.fc_layer('Q', tn_fc4, input_size=512, num_neron=num_action, activation=None, variable_dict=tn_variable_dict)
    
        # Network Cloning
        with tf.variable_scope('Prediction_to_Target'):
            network_cloning_ops = []
            assert (tn_variable_dict.keys() == pn_variable_dict.keys())
            for k in tn_variable_dict.keys():
                network_cloning_ops.append(tf.assign(tn_variable_dict[k], pn_variable_dict[k]))

        # Performance Evaluation
        with tf.variable_scope('performance_evaluation'):
            episode_reward = tf.placeholder(name='episode_reward', shape=(), dtype=tf.int32)
            summary_avg_episode_reward = tf.summary.scalar('episode_reward', episode_reward)
            episode_steps = tf.placeholder(name='episode_steps', shape=(), dtype=tf.int32)
            summary_avg_episode_steps = tf.summary.scalar('episode_steps', episode_steps)        
    
    return (pn_state_placeholder, pn_target_placeholder, pn_Q, pn_loss, pn_actions, pn_q_target, pn_train, pn_importance_weight, pn_delta), (tn_state_placeholder, tn_target_placeholder, tn_Q), network_cloning_ops, (summary_pn_loss, summary_averaged_pn_Q), (episode_reward, summary_avg_episode_reward, episode_steps, summary_avg_episode_steps)

