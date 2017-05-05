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
    with tf.variable_scope(scope):
        # get the nodes of input images and output features
        with tf.name_scope('inputs'):
            state_placeholder = tf.placeholder(
                name  = 'state', 
                shape = (None, 2048), 
                dtype = tf.float32)
            target_placeholder = tf.placeholder(
                name = 'target',
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
            advantage_placeholder = tf.placeholder(
                name  = 'advantage',
                shape = (None, ),
                dtype = tf.float32)
        

        # compute embedded feature given the input image feature
        variable_dict = {}
        with tf.variable_scope('shared_layers') as tmp_scope:
            # fc1 
            state_flattened = tf.reshape(state_placeholder, (-1, A3CConfig.num_history_frames * 2048))
            fc1_state = TFUtil.fc_layer('fc1', state_flattened, input_size=A3CConfig.num_history_frames * 2048, num_neron=512, variable_dict=variable_dict)
            tmp_scope.reuse_variables()
            target_flattened = tf.reshape(target_placeholder, (-1, A3CConfig.num_history_frames * 2048))
            fc1_target = TFUtil.fc_layer('fc1', target_flattened, input_size=A3CConfig.num_history_frames * 2048, num_neron=512, variable_dict=variable_dict)
        with tf.variable_scope('shared_layers'):    
            # fc2
            fc2 = TFUtil.fc_layer('fc2',
                                  tf.concat((fc1_state, fc1_target), axis = 1), 
                                  input_size=1024, num_neron=512, variable_dict=variable_dict)

        # outputs
        policy_logits_dict = {}
        policy_prob_dict = {}
        state_value_dict = {}
        for scene in THORConfig.supported_envs:
            with tf.variable_scope(scene, reuse = False):
                # fc3 shared for policy and value output
                fc3 = TFUtil.fc_layer('fc_3', fc2, input_size=512, num_neron=512, variable_dict=variable_dict)
                # policy output
                policy_logits = TFUtil.fc_layer('policy_logits', fc3, input_size=512, num_neron=num_action, activation=None, variable_dict=variable_dict)
                policy_probs = tf.nn.softmax(name = 'policy_probs', logits = policy_logits)
                # value output
                state_value = tf.squeeze(TFUtil.fc_layer('value', fc3, input_size=512, num_neron=1, activation=None, variable_dict=variable_dict), axis = 1)                
                # add output to list
                policy_logits_dict[scene] = policy_logits
                policy_prob_dict[scene] = policy_probs
                state_value_dict[scene] = state_value 

        with tf.variable_scope('loss'):
            scene_losses = {}
            for scene in THORConfig.supported_envs:
                # policy loss
                log_prob = - tf.nn.sparse_softmax_cross_entropy_with_logits(
                    name   = 'log_prob',
                    labels = action_placeholder,
                    logits = policy_logits_dict[scene])
                policy_loss = - tf.reduce_sum(log_prob * tf.stop_gradient(q_value_placeholder - state_value_dict[scene])) # regularization for A3C delay
                policy_entropy = - 0.01 * tf.reduce_sum(policy_prob_dict[scene] * tf.log(tf.clip_by_value(policy_prob_dict[scene], 1e-20, 1)))
                # value_loss
                value_loss = 0.5 * tf.reduce_sum(tf.square(q_value_placeholder - state_value_dict[scene]))
                # need to tweak weight
                scene_losses[scene] = policy_loss + value_loss - policy_entropy
                with tf.name_scope(scene): 
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('policy_entropy', policy_entropy)
                    tf.summary.scalar('value_loss', value_loss)

    with tf.name_scope('train_ops'):
        train_ops = {}
        # train_op
        # optional: varying learning_rate
        """
        learning_rate = tf.train.exponential_decay(
            learning_rate = A3CConfig.learning_rate, 
            global_step   = global_step, 
            decay_steps   = A3CConfig.decay_step,
            decay_rate    = A3CConfig.decay_rate)
        """
        
        # create optimizer
        #optimizer = tf.train.AdamOptimizer(learning_rate = A3CConfig.learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate = A3CConfig.learning_rate, decay = A3CConfig.decay_rate, epsilon = 0.1)#, momentum = A3CConfig.momentum)
        
        for scene in THORConfig.supported_envs:
            # get local trainable variables and compute gradients
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
            grad_var = optimizer.compute_gradients(scene_losses[scene], var_list = local_vars)
            
            # optional: gradient clipping
            clipped_grad_var = []
            for grad, var in grad_var:
                if grad is not None:
                    clipped_grad_var.append((tf.clip_by_norm(grad, 40.), var))
                else:
                    clipped_grad_var.append((None, var))
            grad_var = clipped_grad_var
            
            # apply gradient to global variables
            if scope != 'global':
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'global')
                tmp = []
                for i in xrange(len(grad_var)):
                    tmp.append((grad_var[i][0], global_vars[i]))
                grad_var = tmp
                    
            train_ops[scene] = optimizer.apply_gradients(grad_var)
    
    # get summary ops
    with tf.variable_scope('global', reuse = (scope != "global")):
        tmp_scope.reuse_variables()
        reward = tf.get_variable(
            name = 'global/reward',
            shape = (1, ),
            dtype = tf.float32,
            trainable = False)
        num_step = tf.get_variable(
            name = 'global/num_steps',
            shape = (1, ),
            dtype = tf.float32,
            trainable = False)
    if scope == "global":
        with tf.name_scope("global_summary"):
            tf.summary.scalar('reward', reward)
            tf.summary.scalar('average_steps_over_100_episodes', num_step)

    return (state_placeholder, target_placeholder, action_placeholder, q_value_placeholder), \
           (policy_prob_dict, state_value_dict), \
           (train_ops, ), \
           (reward, num_step)

