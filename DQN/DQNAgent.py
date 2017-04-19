import random

import tensorflow as tf
import numpy as np
import os

import DQNConfig
import DQNUtil
import DQNNet
import ImgUtil
import TFUtil

class DQNAgent:
    """ Agent with Deep Q Network
    """
    def __init__(self, env):
        self._env = env
        self._replay_memory = DQNUtil.ExperienceReplayMemory()
        self._histroy_frames = DQNUtil.FrameHistoryBuffer()
        self._tf_sess = None

        # tensors
        self._tf_pn_Q = None
        self._tf_pn_state = None
        self._tf_pn_actions = None
        self._tf_pn_loss = None
        self._tf_pn_Q_target = None
        self._tf_pn_train = None
        self._tf_tn_state = None
        self._tf_tn_Q = None
        self._tf_clone_ops = []
        self._tf_episode_reward = None

        self._tf_summary_pn_loss = None
        self._tf_summary_averaged_pn_Q = None
        self._tf_summary_episode_reward = None
        self._tf_episode_reward = None
        self._saver = None

    def learn(self, double_dqn, dueling_dqn, model_save_frequency, model_save_path, model_load_path, use_gpu, gpu_id):
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.device(device_str):
            self._init_dqn(dueling_dqn)

            # initialize all variables
            init = tf.global_variables_initializer()

            # create auxiliary operations: summary and saver
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(DQNConfig.summary_folder)

        # initialize
        if model_load_path is None:
            self._tf_sess.run(init)
        else:
            self.load(model_load_path)

        # start new episode
        self._request_new_episode()

        # first take some random actions to populate the replay memory before learning starts
        if model_load_path is None:
            print('Taking random actions to warm up...')
            for i in range(DQNConfig.replay_start_size):
                if i % 100 == 0:
                    print('{0}/{1}'.format(i, DQNConfig.replay_start_size))
                done = self._perform_random_action()
                if done:
                    self._request_new_episode()

        print('Training started, please open Tensorboard to monitor the training process.')
        episode_count=0
        for i in range(DQNConfig.max_iterations):

            if i % 1000 ==0:
                print('{0}/{1}'.format(i, DQNConfig.max_iterations))

            # save model
            if i % model_save_frequency == 0 and i != 0:
                self.save(model_save_path)

            # update target_network
            if i % DQNConfig.target_network_update_freq == 0:
                self._tf_sess.run(self._tf_clone_ops)

            # select and perform action
            state = self._get_current_state()
            state = state[np.newaxis]
            Q = self._evaluate_q(state)
            a = self._select_action(Q, i, test_mode=False)
            self._perform_action(a)
            if self._env.episode_done():
                episode_reward = self._env.get_total_reward()
                summary_episode_reward = self._tf_sess.run(self._tf_summary_episode_reward,
                                                           feed_dict={self._tf_episode_reward: episode_reward})
                summary_writer.add_summary(summary_episode_reward, global_step=episode_count)
                episode_count += 1
                self._request_new_episode()


            # sample mini-batch and perform training
            experiences = self._replay_memory.sample(DQNConfig.batch_size)
            states, actions, states_new, rewards, dones = DQNAgent.decompose_experiences(experiences)

            # compute q_targets
            if double_dqn:
                # double DQN: use prediction network for action selection, use target network for action's Q value evaluation
                q_new_p = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: states_new})
                action = np.argmax(q_new_p, axis=1)
                q_new_t = self._tf_sess.run(self._tf_tn_Q, feed_dict={self._tf_tn_state: states_new})
                q_new_max = np.array([q_new_t[j, action[j]] for j in range(DQNConfig.batch_size)])
            else:
                # DQN: use target network for action selection and evaluation
                q_new = self._tf_sess.run(self._tf_tn_Q, feed_dict={self._tf_tn_state: states_new})
                q_new_max = np.max(q_new, axis=1)

            q_targets = rewards + q_new_max * DQNConfig.discounted_factor * (1. - dones.astype(np.int))

            # train
            _, loss, summary_pn_loss, summary_averaged_pn_Q = self._tf_sess.run([self._tf_pn_train, self._tf_pn_loss, self._tf_summary_pn_loss, self._tf_summary_averaged_pn_Q],
                                                                                feed_dict={self._tf_pn_actions: actions,
                                                                                           self._tf_pn_state: states,
                                                                                           self._tf_pn_Q_target: q_targets})
            summary_writer.add_summary(summary_pn_loss, global_step=i)
            summary_writer.add_summary(summary_averaged_pn_Q, global_step=i)

        # save model after training
        self.save(model_save_path)

    def test(self, dueling_dqn, model_load_path, use_gpu, gpu_id=None):
        # build network
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.device(device_str):
            self._init_dqn(dueling_dqn)

        # initialize all variables from the model
        self.load(model_load_path)

        # start new episode
        self._request_new_episode()

        # perform testing
        for i in range(DQNConfig.max_iterations):
            # select and perform action
            state = self._get_current_state()
            state = state[np.newaxis]
            Q = self._evaluate_q(state)
            a = self._select_action(Q, i, test_mode=True)
            self._perform_action(a)
            if self._env.episode_done():
                print('total_reward received: {0}'.format(self._env.get_total_reward()))
                self._request_new_episode()

    def _init_dqn(self, dueling_dqn):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self._tf_sess = tf.Session(config=config)
        pn_nodes, tn_nodes, cloning_ops, train_summary_ops, evaluation = DQNNet.build_dqn(self._env.get_num_actions(), dueling_dqn=dueling_dqn)
        self._tf_pn_state, self._tf_pn_Q, self._tf_pn_loss, self._tf_pn_actions, self._tf_pn_Q_target, self._tf_pn_train = pn_nodes
        self._tf_tn_state, self._tf_tn_Q = tn_nodes
        self._tf_clone_ops = cloning_ops
        self._tf_summary_pn_loss, self._tf_summary_averaged_pn_Q = train_summary_ops
        self._tf_episode_reward, self._tf_summary_episode_reward = evaluation
        self._saver = tf.train.Saver()

    def _evaluate_q(self, state):
        assert(self._tf_sess is not None)
        assert(self._tf_pn_Q is not None)
        Q = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: state})
        return Q

    def _select_action(self, Q, i, test_mode=False):
        if test_mode:
            exploration_prob = DQNConfig.test_exploration
        else:
            if i >= DQNConfig.final_exploration_frame:
                exploration_prob = DQNConfig.final_exploration
            else:
                exploration_prob = DQNConfig.initial_exploration + i * DQNConfig.exploration_change_rate
        if random.random() < exploration_prob:
            action = random.randrange(0, self._env.get_num_actions())
        else:
            action = np.argmax(Q)
        return action

    def _perform_action(self, action):
        observation, reward, done = self._env.step(action)
        observation = DQNAgent.preprocess_frame(observation)
        self._histroy_frames.record(observation)
        self._replay_memory.add(action, reward, observation, done)
        return done

    def _request_new_episode(self):
        observation = self._env.reset()
        observation = DQNAgent.preprocess_frame(observation)
        assert(observation is not None)
        self._histroy_frames.fill_with(observation)

    def _perform_random_action(self):
        action = random.randrange(0, self._env.get_num_actions())
        return self._perform_action(action)

    def _get_current_state(self):
        return self._histroy_frames.copy_content()

    def save(self, model_save_path):
        print('saving model to {0}'.format(model_save_path))
        self._saver.save(self._tf_sess, save_path=model_save_path)
        self._histroy_frames.save(model_save_path + '.history_frame_buffer')
        # replay memory can be huge, so we first dump it to a tmp file then rename it
        # to the destination to prevent the process being interrupted during dumpping
        # replay memory
        if False:
            self._replay_memory.save(model_save_path + '.replay_memory.tmp')
            if os.path.exists(model_save_path + '.replay_memory'):
                os.remove(model_save_path + '.replay_memory')
            os.rename(model_save_path + '.replay_memory.tmp', model_save_path + '.replay_memory')
        print('model saved.')

    def load(self, model_load_path):
        print('loading model from {0}'.format(model_load_path))
        self._saver.restore(self._tf_sess, model_load_path)
        if False:
            self._replay_memory.load(model_load_path + '.replay_memory')
        self._histroy_frames.load(model_load_path + '.history_frame_buffer')
        print('model loaded.')

    @staticmethod
    def preprocess_frame(frame):
        return ImgUtil.rgb_to_luminance(ImgUtil.resize_img(frame, DQNConfig.frame_size))

    @staticmethod
    def decompose_experiences(experiences):
        states = np.array([experience[0] for experience in experiences])
        actions = np.array([experience[1] for experience in experiences])
        states_new = np.array([experience[2] for experience in experiences])
        rewards = np.array([experience[3] for experience in experiences])
        dones = np.array([experience[4] for experience in experiences])
        return states, actions, states_new, rewards, dones


