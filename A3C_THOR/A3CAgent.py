import sys
sys.path.append('../')
import random
import tensorflow as tf
import numpy as np

from THOR import THORConfig
import TFUtil
import THORNet
from THOR.THOREnv import THOREnvironment

import A3CUtil
import A3CConfig
import ImgUtil
import threading

from collections import deque

class A3CAgent:
    """ A3C Agent
    """
    def __init__(self, num_threads, model_save_frequency, model_save_path, use_extracted_feature = False):
        # model related setting
        self._model_save_freq = model_save_frequency
        self._model_save_path = model_save_path

        # environment related objects
        self._num_threads = num_threads
        self._supported_scens = THORConfig.supported_envs
        self._num_scenes = len(self._supported_scens)
        self._use_extracted_feature = use_extracted_feature
        if self._use_extracted_feature:
            self._resnet_feature_table = np.load(THORConfig.feature_table_path)

        self._envs = [THOREnvironment() for _ in xrange(self._num_threads)]
        self._num_actions = len(THORConfig.supported_actions)
        self._episode_reward_buffer = deque(maxlen=100)
        self._episode_reward_buffer.append(0)
        self._episode_step_buffer = deque(maxlen=100)

        #self._replay_memory = PGUtil.ExperienceReplayMemory(A3CConfig.replay_memory)
        #self._histroy_frames = PGUtil.FrameHistoryBuffer(A3CConfig.frame_size, A3CConfig.num_history_frames)
        self._tf_sess = None
        self._iter_idx = 0

        # tensors
        self._tf_resnet_input = None
        self._tf_resnet_output = None
        self._tf_global_step = None
        self._tf_acn_state = None
        self._tf_acn_image_feature = None
        self._tf_acn_action = None
        self._tf_acn_q_value = None
        self._tf_acn_sence = None
        self._tf_acn_policy_logit_list = None
        self._tf_acn_state_value_list = None

        # ops
        self._tf_acn_train_op = None
        self._tf_action_samplers = None

        # variables
        self._tf_reward_history = None
        self._tf_average_reward = None
        self._tf_step_history = None
        self._tf_average_step = None

        self._tf_summary_op = None
        self._summary_writer = None

        # saver
        self._saver = None

    def learn(self, check_point, use_gpu, gpu_id=None):
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.Graph().as_default():
            with tf.device(device_str):
                # create resnet if no extracted feature
                if not self._use_extracted_feature:
                    resnet_saver = tf.train.import_meta_graph(A3CConfig.resnet_meta_graph)
                    graph = tf.get_default_graph()
                    self._tf_resnet_input = graph.get_tensor_by_name("images:0") 
                    self._tf_resnet_output = graph.get_tensor_by_name("avg_pool:0")
                
                # build network
                self._init_network(scope = "global")
                self._saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global'))

                # create initializer
                init = tf.global_variables_initializer()

                # create auxiliary operations: summary and saver
                self._tf_summary_op = tf.summary.merge_all()
                self._summary_writer = tf.summary.FileWriter(A3CConfig.summary_folder, self._tf_sess.graph)

                # initialize or load network variables
                if check_point is None:
                    # initialize all variable
                    self._tf_sess.run(init)
                    if not self._use_extracted_feature:
                        # load pretrain resnet
                        resnet_saver.restore(self._tf_sess, A3CConfig.resnet_pretrain_model)
                    self._iter_idx = 0
                else:
                    if not self._use_extracted_feature:
                        # load pretrain resnet
                        resnet_saver.restore(self._tf_sess, A3CConfig.resnet_pretrain_model)
                    self.load(self._model_save_path, check_point)
                    self._iter_idx = check_point

        # start training
        # create learner threads
        learner_threads = [threading.Thread(target=self.learner_thread, args=(thread_id, ))\
                        for thread_id in xrange(self._num_threads)]
        for t in learner_threads:
            t.start()

        print('Training started, please open Tensorboard to monitor the training process.')
        # Show the agents training and write summary statistics
        """
        last_summary_time = 0
        while True:
            now = time.time()
            if now - last_summary_time > SUMMARY_INTERVAL:
                summary_str = session.run(summary_op)
                writer.add_summary(summary_str, float(T))
                last_summary_time = now
        """
        for t in learner_threads:
            t.join()



    def learner_thread(self, thread_id, ):
        env = self._envs[thread_id]
        # create local network
        """
        scope = "agent_%02i" %(thread_id)
        placeholders, ops, variables = self._init_network(scope = scope)

        local_step, local_acn_state, local_acn_action, local_acn_q_value, local_acn_sence,  _, _ = placeholders
        local_acn_train_op, local_action_samplers = ops
        local_acn_policy_logit_list, local_acn_state_value_list, _, _ = variables

        self._copy_network_var(from_scope = 'global', to_scope = scope)
        """

        state = self._request_new_episode(env)
        print "Target idx:", env._target_idx
        history_buffer = A3CUtil.HistoryBuffer()
        total_rewards = 0
        total_steps = 0
        while self._iter_idx < A3CConfig.max_iterations:
            for _ in xrange(A3CConfig.max_steps):
                # sample and perform action, store history
                action = self._select_action(
                    env = env,
                    state = state,
                    i = self._iter_idx)
                next_state, reward, done = self._perform_action(env, state, action, history_buffer)
                total_rewards += reward
                state = next_state
                #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                #    print (('ep %d: game finished, reward: %f' % (episode_idx + 1, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
                total_steps += 1
                if done:
                    break

            value = 0 if done else self._tf_sess.run(
                self._tf_acn_state_value_list[env._env_idx],
                feed_dict = {self._tf_acn_state: state})[0]


            # update models
            states = history_buffer._state_buffer
            actions = history_buffer._action_buffer
            q_values = history_buffer.compute_q_value(value)
            scene_one_hot = np.zeros((q_values.shape[0], self._num_scenes), dtype = np.float32)
            scene_one_hot[:, env._env_idx] = 1


            _, _, _, summary = \
            self._tf_sess.run([self._tf_acn_train_op, self._tf_average_reward, self._tf_average_step, self._tf_summary_op],
                            feed_dict={ self._tf_global        : self._iter_idx,
                                        self._tf_acn_state     : states,
                                        self._tf_acn_action    : actions,
                                        self._tf_acn_q_value   : q_values,
                                        self._tf_acn_sence     : scene_one_hot,
                                        self._tf_reward_history: self._episode_reward_buffer,
                                        self._tf_step_history  : self._episode_step_buffer
                                        })
            #self._copy_network_var(from_scope = 'global', to_scope = scope)
            history_buffer.clean_up()

            # reset environment if done
            if done:
                # save number of steps
                self._episode_step_buffer.append(total_steps)
                print("Number of steps for this episode: {}".format(total_steps))
                print("Average number of steps for last 100 episodes: {}".format(np.mean(self._episode_step_buffer)))
                # save episode reward
                self._episode_reward_buffer.append(total_rewards)
                print("Reward for this episode: {}".format(total_rewards))
                print("Average reward for last 100 episodes: {}".format(np.mean(self._episode_reward_buffer)))
                total_rewards = 0
                # reset
                state = self._request_new_episode(env)
                print "Target idx:", env._target_idx

            # record summary
            self._summary_writer.add_summary(summary, global_step = self._iter_idx)
            self._iter_idx += 1
            if self._iter_idx % self._model_save_freq == 0 or self._iter_idx == A3CConfig.max_iterations:
                print("Model saved after {} iterations".format(self._iter_idx))
                self.save(self._model_save_path, global_step = self._iter_idx)


    def test(self, check_point, use_gpu, gpu_id=None):
        assert(len(self._envs) == 1)
        env = self._envs[0]
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
        state = self._request_new_episode(env)

        # perform testing
        while episode_idx <= A3CConfig.max_iterations:
            # sample and perform action, store history
            action = self._select_action(env, state, episode_idx, test_mode = True)
            next_state, reward, done = self._perform_action(env, state, action, history_buffer)
            total_rewards += reward
            state = next_state

            if done:
                episode_idx += 1
                print('total_reward received: {0}'.format(total_rewards))
                history_buffer.clean_up()
                state = self._request_new_episode(env)
                total_rewards = 0


    def _init_network(self, scope = 'global'):
        # build a3c network
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self._tf_sess = tf.Session(config=config)
        
        placeholders, ops, variables = THORNet.build_actor_critic_network(scope, self._num_actions, self._num_scenes)
        if scope == 'global':
            self._tf_global_step, self._tf_acn_state, self._tf_acn_action, self._tf_acn_q_value, \
            self._tf_acn_sence, self._tf_reward_history, self._tf_step_history = placeholders
            self._tf_acn_train_op, self._tf_action_samplers = ops
            self._tf_acn_policy_logit_list, self._tf_acn_state_value_list, self._tf_average_reward, self._tf_average_step = variables
        return placeholders, ops, variables

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

    def _select_action(self, env, state, i, test_mode = False):
        if test_mode:
            exploration_prob = A3CConfig.test_exploration
        else:
            if i >= A3CConfig.final_exploration_frame:
                exploration_prob = A3CConfig.final_exploration
            else:
                exploration_prob = A3CConfig.initial_exploration + i * A3CConfig.exploration_change_rate

        if random.random() < exploration_prob:
            action = random.randrange(0, self._num_actions)
        else:
            if test_mode:
                #action = np.argmax(self._tf_sess.run(, {network_state_op: state})[0])
                action = np.argmax(self._tf_sess.run(
                    self._tf_policy_logits_list[env._env_idx],
                    {self._tf_acn_state: state})[0])
            else:
                #action = self._tf_sess.run(sampler_op, {network_state_op: state})[0][0]
                action = self._tf_sess.run(
                    self._tf_action_samplers[env._env_idx],
                    {self._tf_acn_state: state})[0]
        return action

    def _perform_action(self, env, state, action, history_buffer):
        if not self._use_extracted_feature:
            # perfrom action and get next frame
            next_frame, _, reward, done = env.step(action)
            next_frame = self.preprocess_frame(next_frame)
            next_frame_feat = self._tf_sess.run(
                self._tf_resnet_output, 
                {self._tf_resnet_input: next_frame[np.newaxis, :, :, :]})
            next_state = np.concatenate(
                (state[1:A3CConfig.num_history_frames, :], next_frame_feat, state[-1:, :]),
                axis = 0)
        else:
            # perfrom action, get next frame idx, and retrieve the feature from table
            next_idx, _, reward, done = env.step(action)
            next_state = np.concatenate(
                (state[1:A3CConfig.num_history_frames, :], self._resnet_feature_table[next_idx:next_idx+1, :], state[-1:, :]),
                axis = 0)
        # get the value of the current state
        #value = self._tf_sess.run( value_op, feed_dict = {state_op: state})[0]
        value = self._tf_sess.run(
            self._tf_acn_value_list[env._env_idx],
            {self._tf_acn_state: state}
            )[0]
        
        # store roll out
        history_buffer.store_rollout(state, reward, action, value)
        # get next state using current state and next frame
        
        return next_state, reward, done

    def _request_new_episode(self, env):
        if not self._use_extracted_feature:
            frame = self.preprocess_frame(env.reset_random())
            # extract resnet feature
            state = self._tf_sess.run(self._tf_resnet_output, 
                                    {self._tf_resnet_input: frame[np.newaxis, :, :, :]})
            state = np.tile(state, (A3CConfig.num_history_frames, 1))
            target = self._tf_sess.run(self._tf_resnet_output,
                                    {self._tf_resnet_input: self.preprocess_frame(env._target_img)[np.newaxis, :, :, :]})
            state = np.concatenate((state, target), axis = 0)
        else:
            idx = env.reset_random()
            # retrieve resnet feature from table
            feature = self._resnet_feature_table[idx, :]
            state = np.tile(feature[np.newaxis, :], (A3CConfig.num_history_frames, 1))
        return state

    def _perform_random_action(self):
        pass
        """
        action = random.randrange(0, self._env.get_num_actions())
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
