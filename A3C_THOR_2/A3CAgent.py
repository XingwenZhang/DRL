import sys
sys.path.append('../')
sys.path.append('../THOR')
import random
import tensorflow as tf
import numpy as np

from THOR import THORConfig
import TFUtil
import THORNet
from THOR.THOREnv import THOREnvironment
from THOR import THOROfflineEnv

import A3CUtil
import A3CConfig
import ImgUtil
import threading
from collections import deque

global global_frame_num
global global_episode_num

class A3CAgent:
    num_frames = 0
    num_episodes = 0
    tf_resnet_input = None
    tf_resnet_output = None
    tf_resnet_saver = None
    # summary ops
    tf_global_summary_op = None
    tf_summary_writer = None
    episode_step_buffer = deque(maxlen = 100)
    episode_reward_buffer = deque(maxlen = 100)

    def __init__(self, sess, scope, feature_mode = False):
        # scope name
        self._scope = scope
        self._env = None

        # environment related objects
        self._feature_mode = feature_mode
        self._supported_scens = THORConfig.supported_envs
        self._supported_actions = THORConfig.supported_actions
        self._num_actions = len(THORConfig.supported_actions)
        THOREnvironment.pre_load(feat_mode = feature_mode)

        # declare history buffer
        self._history_buffer = A3CUtil.HistoryBuffer()
        
        #self._replay_memory = PGUtil.ExperienceReplayMemory(A3CConfig.replay_memory)
        #self._histroy_frames = PGUtil.FrameHistoryBuffer(A3CConfig.frame_size, A3CConfig.num_history_frames)
        self._tf_sess = sess
        self._iter_idx = 0
        self._num_frames = 0

        # tensors
        self._tf_acn_state = None
        self._tf_acn_target = None
        self._tf_acn_action = None
        self._tf_acn_q_value = None
        self._tf_acn_advantage = None
        self._tf_acn_policy_prob_dict = None
        self._tf_acn_state_value_dict = None
        
        # ops
        self._tf_acn_train_ops = None
        self._tf_acn_summary_op = None

        # saver
        self._saver = None


    def learn(self, use_gpu, gpu_id=None):
        self._env = THOREnvironment(feat_mode = self._feature_mode)
        done = True
        while A3CAgent.num_episodes < A3CConfig.max_iterations:
            # if this is not global agent, copy parameters from global agent
            if self._scope != "global":
                self._copy_network_var(from_scope = "global", to_scope = "local_variable")

            # if episode is done, request new episode
            if done:
                A3CAgent.num_episodes += 1
                state, target = self._request_new_episode()
                env_idx = self._env._env_idx; target_idx = self._env._target_idx
                print "Episode: {0}, Env: {1}, Target idx: {2}".format(A3CAgent.num_episodes, THORConfig.supported_envs[env_idx], target_idx)
                total_steps = 0
                total_rewards = 0

            for _ in xrange(A3CConfig.max_steps):
                # sample and perform action, store history
                action = self._select_action(state, target)
                next_state, reward, done = self._perform_action(state, target, action)
                total_rewards += reward
                total_steps += 1
                cur_frame_num = A3CAgent.num_frames = A3CAgent.num_frames + 1
                state = next_state
                #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                #    print (('ep %d: game finished, reward: %f' % (episode_idx + 1, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
                if done:
                    break

            if done: 
                value = 0
            else:
                value = self._tf_sess.run(
                    self._tf_acn_state_value_dict[THORConfig.supported_envs[env_idx]],
                    feed_dict = {self._tf_acn_state: state, 
                                 self._tf_acn_target: target})[0]
            
            # update models
            states = self._history_buffer._state_buffer
            actions = self._history_buffer._action_buffer
            q_values = self._history_buffer.compute_q_value(value)
            _, summary = self._tf_sess.run(
                [self._tf_acn_train_ops[THORConfig.supported_envs[env_idx]], self._tf_acn_summary_op],
                feed_dict={ self._tf_global_step : A3CAgent.num_frames,
                            self._tf_acn_state   : states,
                            self._tf_acn_target  : np.tile(target, (len(actions), 1)),
                            self._tf_acn_action  : actions,
                            self._tf_acn_q_value : q_values})

            A3CAgent.tf_summary_writer.add_summary(summary, global_step = A3CAgent.num_frames)

            self._history_buffer.clean_up()

            # reset environment if done
            if done:
                # save number of steps
                A3CAgent.episode_step_buffer.append(total_steps)
                print("Number of steps for this episode: {}".format(total_steps))
                print("Average number of steps for last 100 episodes: {}".format(np.mean(A3CAgent.episode_step_buffer)))
                # save episode reward
                A3CAgent.episode_reward_buffer.append(total_rewards)
                print("Reward for this episode: {}".format(total_rewards))
                print("Average reward for last 100 episodes: {}".format(np.mean(A3CAgent.episode_reward_buffer)))
                summary = self._tf_sess.run(
                    A3CAgent.tf_global_summary_op,
                    feed_dict={ self._tf_num_steps   : [A3CAgent.episode_step_buffer[-1]], 
                                self._tf_reward      : [A3CAgent.episode_reward_buffer[-1]]})

                A3CAgent.tf_summary_writer.add_summary(summary, global_step = A3CAgent.num_frames)

    def save_model_monitor (self, last_save_frame, model_save_path, model_save_interval):
        assert(self._scope == "global")
        cur_frame = 0
        while cur_frame < A3CConfig.max_iterations:
            cur_frame = A3CAgent.num_frames
            if cur_frame - last_save_frame > model_save_interval:
                print("Model saved after {} frames".format(cur_frame))
                self.save(model_save_path, global_step = cur_frame)
                last_save_frame = cur_frame


    def test(self, check_point, use_gpu, gpu_id=None):
        assert(len(envs) == 1)
        env = envs[0]
        history_buffer = A3CUtil.HistoryBuffer()

        # build network
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.device(device_str):
            #self._init_network()
            # initialize all variables from the model
            self.load(self._model_save_path, check_point)

        # start new episode
        episode_idx = 0
        total_rewards = 0
        state, target = self._request_new_episode(env)

        # perform testing
        while episode_idx <= A3CConfig.max_iterations:
            # sample and perform action, store history
            action = self._select_action(env, state, target, test_mode = True)
            next_state, reward, done = self._perform_action(env, state, target, action, history_buffer)
            total_rewards += reward
            state = next_state

            if done:
                episode_idx += 1
                print('total_reward received: {0}'.format(total_rewards))
                history_buffer.clean_up()
                state = self._request_new_episode(env)
                total_rewards = 0

    def _init_network(self, check_point = None):
        # build a3c network

        # load shared resnet input
        if self._scope == 'global':
            A3CAgent.tf_resnet_saver = tf.train.import_meta_graph(A3CConfig.resnet_meta_graph)
            graph = tf.get_default_graph()
            A3CAgent.tf_resnet_input = graph.get_tensor_by_name("images:0") 
            A3CAgent.tf_resnet_output = graph.get_tensor_by_name("avg_pool:0")
            
        
        # build network
        placeholders, dicts, ops, variables = THORNet.build_actor_critic_network(self._scope, self._num_actions, self._supported_scens)
        self._tf_global_step, self._tf_acn_state, self._tf_acn_target, self._tf_acn_action, self._tf_acn_q_value = placeholders
        self._tf_acn_policy_prob_dict, self._tf_acn_state_value_dict = dicts
        self._tf_acn_train_ops, self._tf_acn_summary_op = ops
        self._tf_reward, self._tf_num_steps = variables
        
        # create initializer
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self._scope)
        if self._scope == "global":
            self._saver = tf.train.Saver(var_list = var_list)
        """
        #if self._scope == "global":
        #    for var in var_list:
        #        print var.name
        #    exit(1)
        init_op = tf.variables_initializer(var_list)
  
        # initialize or load network variables
        if self._scope == "global":
            if check_point is None:
                self._saver = tf.train.Saver(var_list = var_list)
                # initialize all variable
                self._tf_sess.run(init_op)
                A3CConfig.num_frames = 0
            else:
                self.load(self._model_save_path, check_point)
                A3CConfig.num_frames = check_point
        else:
            self._tf_sess.run(init_op)
        """
    
    def create_global_summary_ops(self):
        assert(self._scope == "global")
        # get summary ops
        global_summaries = []
        with tf.variable_scope('global', reuse = True):
            fc1_weight = tf.get_local_variable(
                name = 'shared_layers/fc1/fc_weights')
            fc1_bias = tf.get_variable(
                name = 'shared_layers/fc1/fc_bias')
            fc2_weight = tf.get_local_variable(
                name = 'shared_layers/fc2/fc_weights')
            fc2_bias = tf.get_variable(
                name = 'shared_layers/fc2/fc_bias')
            fc3_weight = tf.get_local_variable(
                name = 'env_t1/fc3/fc_weights')
            fc3_bias = tf.get_variable(
                name = 'env_t1/fc3/fc_bias')
            
        with tf.name_scope("global_summary"):
            global_summaries.append(tf.summary.scalar('reward', tf.reduce_mean(self._tf_reward)))
            global_summaries.append(tf.summary.scalar('num_step', tf.reduce_mean(self._tf_num_steps)))
            global_summaries.append(tf.summary.histogram('fc1_weight', fc1_weight))
            global_summaries.append(tf.summary.histogram('fc1_bias', fc1_bias))
            global_summaries.append(tf.summary.histogram('fc2_weight', fc2_weight))
            global_summaries.append(tf.summary.histogram('fc2_bias', fc2_bias))
            global_summaries.append(tf.summary.histogram('fc3_weight', fc3_weight))
            global_summaries.append(tf.summary.histogram('fc3_bias', fc3_bias))
        A3CAgent.tf_global_summary_op = tf.summary.merge(global_summaries)
        A3CAgent.tf_summary_writer = tf.summary.FileWriter(A3CConfig.summary_folder, self._tf_sess.graph)

    def _evaluate_q(self, state):
        pass
        """
        assert(self._tf_sess is not None)
        assert(self._tf_pn_Q is not None)
        Q = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: state})
        return Q
        """
    
    def _copy_network_var(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
        for from_var,to_var in zip(from_vars,to_vars):
            to_var.assign(from_var)

    def _select_action(self, state, target, test_mode = False):
        if test_mode:
            exploration_prob = A3CConfig.test_exploration
        else:
            if A3CAgent.num_frames >= A3CConfig.final_exploration_frame:
                exploration_prob = A3CConfig.final_exploration
            else:
                exploration_prob = A3CConfig.initial_exploration + A3CAgent.num_frames * A3CConfig.exploration_change_rate

        if random.random() < exploration_prob:
            action = random.randrange(0, self._num_actions)
        else:
            if test_mode:
                #action = np.argmax(self._tf_sess.run(, {network_state_op: state})[0])
                action = np.argmax(self._tf_sess.run(
                    self._tf_acn_policy_prob_dict[THORConfig.supported_envs[self._env._env_idx]],
                    {self._tf_acn_state: state,
                     self._tf_acn_target: target})[0])
            else:
                #action = self._tf_sess.run(sampler_op, {network_state_op: state})[0][0]
                action_probs = self._tf_sess.run(
                    self._tf_acn_policy_prob_dict[THORConfig.supported_envs[self._env._env_idx]],
                    {self._tf_acn_state: state,
                     self._tf_acn_target: target})[0]
                action = np.random.choice(np.arange(self._num_actions), p = action_probs)
        return action

    def _perform_action(self, state, target, action):
        if not self._feature_mode:
            # perfrom action and get next frame
            next_frame, reward, done, success = self._env.step(action)
            if success:
                next_frame = self.preprocess_frame(next_frame)
                next_frame_feat = self._tf_sess.run(
                    A3CAgent.tf_resnet_output, 
                    {A3CAgent.tf_resnet_input: next_frame[np.newaxis, :, :, :]})
                next_state = np.concatenate((state[1:, :], next_frame_feat), axis = 0)
            else:
                next_state = state
        else:
            # perfrom action, get next frame idx, and retrieve the feature from table
            next_feature, reward, done, success = self._env.step(action)
            """
            if success:
                next_state = np.concatenate((state[1:, :], next_feature[np.newaxis, :]),axis = 0)
            else:
                next_state = state
            """
            next_state = np.concatenate((state[1:, :], next_feature[np.newaxis, :]),axis = 0)
        # get the value of the current state
        #value = self._tf_sess.run( value_op, feed_dict = {state_op: state})[0]
        value = self._tf_sess.run(
            self._tf_acn_state_value_dict[THORConfig.supported_envs[self._env._env_idx]],
            {self._tf_acn_state: state, 
             self._tf_acn_target: target}
            )[0]
        
        # store roll out
        self._history_buffer.store_rollout(state, reward, action, value)
        
        return next_state, reward, done

    def _request_new_episode(self):
        if not self._feature_mode:
            frame = self.preprocess_frame(self._env.reset_random())
            # extract resnet feature
            state = self._tf_sess.run(self._tf_resnet_output, 
                                    {self._tf_resnet_input: frame[np.newaxis, :, :, :]})
            state = np.tile(state, (A3CConfig.num_history_frames, 1))
            target = self._tf_sess.run(self._tf_resnet_output,
                                    {self._tf_resnet_input: self.preprocess_frame(env._target_img)[np.newaxis, :, :, :]})
            target = np.tile(target, (A3CConfig.num_history_frames, 1))
        else:
            state_feature = self._env.reset(0, 59)#np.random.choice([44, 75, 98, 59, 91, 2, 35, 46, 65, 78]))
            state = np.tile(state_feature[np.newaxis, :], (A3CConfig.num_history_frames, 1))
            target_feature = self._env.get_target_feat()
            target = np.tile(target_feature[np.newaxis, :], (A3CConfig.num_history_frames, 1))
        return state, target

    def _perform_random_action(self):
        pass
        """
        action = random.randrange(0, env.get_num_actions())
        return self._perform_action(action)
        """

    def _get_current_state(self):
        pass
        """
        return self._histroy_frames.copy_content()
        """

    def save(self, model_save_path, global_step):
        print('saving model to {0}'.format(model_save_path))
        self._saver.save(self._tf_sess, save_path=model_save_path, global_step = global_step)
        print('model saved.')

    def load(self, model_load_path, check_point):
        checkpoint_file = model_load_path + "-" + str(check_point)
        print('loading model from {0}'.format(model_load_path))
        self._saver.restore(self._tf_sess, checkpoint_file)
        print('model loaded.')
    
    @staticmethod
    def add_episode(e_num = 1):
        A3CAgent.num_episodes += e_num

    @staticmethod
    def add_frame(f_num = 1):
        A3CAgent.num_frames += f_num

    @staticmethod
    def preprocess_frame(frame):
        new_size = (THORConfig.net_input_width, THORConfig.net_input_height)
        return ImgUtil.resize_img(frame, new_size)

    @staticmethod
    def decompose_experiences(experiences):
        pass
        """
        states = np.array([experience[0] for experience in experiences])
        actions = np.array([experience[1] for experience in experiences])
        states_new = np.array([experience[2] for experience in experiences])
        rewards = np.array([experience[3] for experience in experiences])
        dones = np.array([experience[4] for experience in experiences])
        return states, actions, states_new, rewards, dones
        """
