""" Tensor Flow Networks
functions for creating different network architectures.
"""

import tensorflow as tf
import TFUtil
import ThorConfig
import math


def build_actor_critic_network(input_feature, num_action, num_scene):
    with tf.variable_scope('actor_critic_network'):
        # Inputs
        with tf.name_scope('inputs'):
            global_step = tf.placeholder(
                name = 'global_step', 
                shape = None,
                dtype = tf.int32)
            action_placeholder = tf.placeholder(
                name  = 'taken_action', 
                shape = (None, ), 
                dtype = tf.int32)
            q_value_placeholder = tf.placeholder(
                name  = 'q_value',
                shape = (None, ),
                dtype = tf.float32)
            scene_placeholder = tf.placeholder(
                name = 'current_scene',
                shape = (None, ),
                dtype = tf.int32
            )

        # compute embedded feature given the input image feature
        variable_dict = {}
        with tf.variable_scope('shared_layers'):
            fc1 = TFUtil.fc_layer('fc1', tf.stop_gradient(input_feature), input_size=2048, num_neron=512, variable_dict=variable_dict)
            fc1_flattened = tf.reshape(fc1, size(-1, 2560), name = "fc1_flattened") # 5n * 512 -> n * 2560
            state_feature = input_feature_flattened[:, 0:2048]
            state_feature = TFUtil.fc_layer('state_feature', state_feature, input_size=2048, num_neron=512, variable_dict=variable_dict)
            target_feature = input_feature_flattened[:, 2048:]
            embedded_feature = TFUtil.fc_layer('embedded_feature',
                                               tf.concat((state_feature, target_feature), axis = 1), 
                                               input_size=1024, num_neron=512, variable_dict=variable_dict)

        # outputs
        policy_logits_list = []
        policy_prob_list = []
        state_value_list = []
        with tf.variable('scene_specific_layers'):
            for i in xrange(num_scene):
                with ('scene_%02i' %(i)):
                    with tf.variable_scope('policy_network'):
                        policy_fc = TFUtil.fc_layer('policy_fc', embedded_feature, input_size=512, num_neron=512, variable_dict=variable_dict) 
                        policy_logits = TFUtil.fc_layer('policy_logits', policy_fc, input_size=512, num_neron=num_action, activation=None, variable_dict=variable_dict)
                        policy_probs = tf.nn.softmax(name = 'policy_probs', logits = policy_logits)
                    with tf.variable_scope('value_network'):
                        value_fc = TFUtil.fc_layer('value_fc', embedded_feature, input_size=512, num_neron=512, variable_dict=variable_dict) 
                        state_value = tf.squeeze(TFUtil.fc_layer('value', value_fc, input_size=512, num_neron=1, activation=None, variable_dict=variable_dict), axis = 1)                
                policy_logits_list.append(policy_logits)
                policy_prob_list.append(policy_probs)
                state_value_list.append(state_value)

        with tf.variable_scope('loss'):
            scene_loss = []
            for i in xrange(num_scene):
                # policy loss
                log_prob = - tf.nn.sparse_softmax_cross_entropy_with_logits(
                    name   = 'log_prob',
                    labels = action_placeholder,
                    logits = policy_logits_list[i])
                policy_loss = - tf.reduce_sum(log_prob * tf.stop_gradient(q_value_placeholder - state_value_list[i])) / A3CConfig.batch_size
                policy_entropy = - 0.005 * tf.reduce_sum(policy_prob_list[i] * tf.log(policy_prob_list[i] + 1e-20)) / A3CConfig.batch_size
                # value_loss
                value_loss = 0.5 * tf.reduce_sum(tf.square(q_value_placeholder - state_value_list[i])) / A3CConfig.batch_size
                # need to tweak weight
                scene_loss.append(policy_loss + value_loss - policy_entropy)
            loss = tf.reduce_sum(scene_loss * scene_placeholder) # scene_placeholder is a one hot vector
            
        # train_op
        """
        learning_rate = tf.train.exponential_decay(
            learning_rate = A3CConfig.learning_rate, 
            global_step   = global_step, 
            decay_steps   = A3CConfig.decay_step,
            decay_rate    = A3CConfig.decay_rate)
        """
        optimizer = tf.train.AdamOptimizer(learning_rate = A3CConfig.learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate = A3CConfig.learning_rate, momentum = A3CConfig.momentum)
        
        grad_var = optimizer.compute_gradients(loss)
        clipped_grad_var = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_var]
        train_op = optimizer.apply_gradients(clipped_grad_var)
        
        #train_op = optimizer.minimize(loss)

        # sample_action
        sample_action = []
        for i in xrange(num_scene):
            tf.multinomial(policy_logits_list[i], 1)

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
            #tf.summary.scalar('policy_loss', policy_loss)
            #tf.summary.scalar('policy_entropy', policy_entropy)
            #tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('loss', loss)

    return (global_step, action_placeholder, q_value_placeholder, scene_placeholder, reward_history_placeholder), \
           (train_op, sample_action), (policy_logits_list, state_value_list, average_reward)

