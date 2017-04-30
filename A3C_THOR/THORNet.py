""" Tensor Flow Networks
functions for creating different network architectures.
"""

import tensorflow as tf
import TFUtil
import A3CConfig
from THOR import THORConfig
import math


def build_actor_critic_network(scope, num_action, num_scene):
    # input: 
    #   num_action   : number of available actions
    #   num_scene    : number of available scene ### maybe better to use list of scene names?
    # Inputs
    with tf.variable_scope(scope):
        # get the nodes of input images and output features
        with tf.name_scope('inputs'):
            global_step = tf.placeholder(
                name = 'global_step', 
                shape = None,
                dtype = tf.int32)
            state_placeholder = tf.placeholder(
                name  = 'state', 
                shape = (None, 2048), 
                dtype = tf.float32)
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
                shape = (None, num_scene),
                dtype = tf.float32)
        

        # compute embedded feature given the input image feature
        variable_dict = {}
        with tf.variable_scope('shared_layers'):
            fc1 = TFUtil.fc_layer('fc1', state_placeholder, input_size=2048, num_neron=512, variable_dict=variable_dict)
            fc1_flattened = tf.reshape(
                fc1, 
                shape = (-1, (A3CConfig.num_history_frames+1) * 512),
                name = "fc1_flattened") # (num_histroy_frmae + 1) * n * 512 -> n * (num_histroy_frmae * 512)
            #state_feature = fc1_flattened[:, 0:(A3CConfig.num_history_frames * 512)]
            #state_feature = TFUtil.fc_layer('state_feature', state_feature, input_size=2048, num_neron=512, activation=None, variable_dict=variable_dict)
            #target_feature = fc1_flattened[:, (A3CConfig.num_history_frames * 512):]
            embedded_feature = TFUtil.fc_layer('embedded_feature',
                                                fc1_flattened, 
                                                input_size=2560, num_neron=512, variable_dict=variable_dict)

        # outputs
        policy_logits_list = []
        policy_prob_list = []
        state_value_list = []
        for i in xrange(num_scene):
            with tf.variable_scope('policy_network_{0:02d}'.format(i)):
                policy_fc = TFUtil.fc_layer('policy_fc_{0:02d}'.format(i), embedded_feature, input_size=512, num_neron=512, variable_dict=variable_dict) 
                policy_logits = TFUtil.fc_layer('policy_logits_{0:02d}'.format(i), policy_fc, input_size=512, num_neron=num_action, activation=None, variable_dict=variable_dict)
                policy_probs = tf.nn.softmax(name = 'policy_probs_{0:02d}'.format(i), logits = policy_logits)
            with tf.variable_scope('value_network_{0:02d}'.format(i)):
                value_fc = TFUtil.fc_layer('value_fc_{0:02d}'.format(i), embedded_feature, input_size=512, num_neron=512, variable_dict=variable_dict) 
                state_value = tf.squeeze(TFUtil.fc_layer('value_{0:02d}'.format(i), value_fc, input_size=512, num_neron=1, activation=None, variable_dict=variable_dict), axis = 1)                
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
                advantage = tf.stop_gradient(q_value_placeholder - state_value_list[i])
                policy_loss = - tf.reduce_mean(log_prob * advantage)
                policy_entropy = - 0.001 * tf.reduce_sum(policy_prob_list[i] * tf.log(policy_prob_list[i] + 1e-20))
                # value_loss
                value_loss = 1 * tf.reduce_mean(tf.square(q_value_placeholder - state_value_list[i]))
                # need to tweak weight
                scene_loss.append(policy_loss + value_loss - policy_entropy)
                with tf.name_scope('secen_summary_%02i' %(i)): 
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('policy_entropy', policy_entropy)
                    tf.summary.scalar('value_loss', value_loss)

            loss = tf.reduce_sum(tf.transpose(scene_loss) * scene_placeholder) # scene_placeholder is one-hot vectors
                
            
            # train_op
            # optional: varying learning_rate
            """
            learning_rate = tf.train.exponential_decay(
                learning_rate = A3CConfig.learning_rate, 
                global_step   = global_step, 
                decay_steps   = A3CConfig.decay_step,
                decay_rate    = A3CConfig.decay_rate)
            """
            # create optizer
            #optimizer = tf.train.AdamOptimizer(learning_rate = A3CConfig.learning_rate)
            optimizer = tf.train.RMSPropOptimizer(learning_rate = A3CConfig.learning_rate)#, momentum = A3CConfig.momentum)
            train_op = optimizer.minimize(loss)

            """
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
            grad_var = optimizer.compute_gradients(loss, var_list = local_vars)
            
            # optional: gradient clipping
            
            clipped_grad_var = []
            for grad, var in grad_var:
                if grad is not None:
                    clipped_grad_var.append((tf.clip_by_value(grad, -10., 10.), var))
                else:
                    clipped_grad_var.append((None, var))
            grad_var = clipped_grad_var
            
                
            
            if scope != 'global':
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'global')
                tmp = []
                for i in xrange(len(grad_var)):
                    tmp.append((grad_var[i][0], global_vars[i]))
                grad_var = tmp
                    
                
            train_op = optimizer.apply_gradients(grad_var)
            """

        # ops to sample_action using multinomial distribution given unnomalized log probability logits
        action_sampler_ops = []
        for i in xrange(num_scene):
            action_sampler_ops.append(tf.multinomial(
                name = 'action_sampler_%02i' %(i), 
                logits = policy_logits_list[i], 
                num_samples = 1))

        # reward_history
        with tf.name_scope('reward_history'):
            reward_history_placeholder = tf.placeholder(
                name = 'reward_history',
                shape = (None, ),
                dtype = tf.float32)
            average_reward = tf.reduce_mean(reward_history_placeholder)

        # step_history
        with tf.name_scope('step_number_history'):
            step_history_placeholder = tf.placeholder(
                name = 'step_history',
                shape = (None, ),
                dtype = tf.float32)
            average_step = tf.reduce_mean(step_history_placeholder)

        # summary
        with tf.name_scope('summary'):
            tf.summary.scalar('average_reward_over_100_episodes', average_reward)
            tf.summary.scalar('average_steps_over_100_episodes', average_step)
            tf.summary.scalar('loss', loss)

        return (global_step, state_placeholder, action_placeholder, q_value_placeholder, \
                scene_placeholder, reward_history_placeholder, step_history_placeholder), \
               (train_op, action_sampler_ops), \
               (policy_logits_list, state_value_list, average_reward, average_step)

