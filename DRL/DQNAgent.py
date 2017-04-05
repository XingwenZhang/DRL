import random

import tensorflow as tf
import numpy as np

import DQNConfig
import DQNUtil
import DQNNet
import ImgUtil


class DQNAgent:
    """ Agent with Deep Q Network
    """
    def __init__(self, env):
        self._env = env
        self._replay_memory = DQNUtil.ExperienceReplayMemory(DQNConfig.replay_memory)
        self._histroy_frames = DQNUtil.FrameHistoryBuffer(DQNConfig.frame_size, DQNConfig.num_history_frames)
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

    def learn(self, model_save_frequency, model_save_path, use_gpu, gpu_id=None):

        if use_gpu:
            device_id = gpu_id if gpu_id is not None else 0
            device_str = '/gpu:' + str(device_id)
        else:
            device_str = '/cpu:0'

        with tf.device(device_str):
            # build DQN network
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            self._tf_sess = tf.Session(config=config)
            pn_nodes, tn_nodes, cloning_ops = DQNNet.build_dqn(self._env.get_num_actions())
            self._tf_pn_state, self._tf_pn_Q, self._tf_pn_loss, self._tf_pn_actions, self._tf_pn_Q_target, self._tf_pn_train = pn_nodes
            self._tf_tn_state, self._tf_tn_Q = tn_nodes
            self._tf_clone_ops = cloning_ops

            # initialize all variables
            init = tf.global_variables_initializer()

            # create auxiliary operations: summary and saver
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(DQNConfig.summary_folder)
            saver = tf.train.Saver()

        # initialize network variables
        self._tf_sess.run(init)

        # first take some random actions to populate the replay memory before learning starts
        print('Taking random actions to warm up...')
        self._request_new_episode()
        for i in range(DQNConfig.replay_start_size):
            if i % 100 == 0:
                print('{0}/{1}'.format(i, DQNConfig.replay_start_size))
            done = self._perform_random_action()
            if done:
                self._request_new_episode()

        print('Training started, please open Tensorboard to monitor the training process.')
        for i in range(DQNConfig.max_iterations):

            # save model
            if i % model_save_frequency == 0:
                saver.save(self._tf_sess, save_path=model_save_path)

            # update target_network
            if i % DQNConfig.target_network_update_freq == 0:
                self._tf_sess.run(self._tf_clone_ops)

            # select and perform action
            state = self._get_current_state()
            state = state[np.newaxis]
            Q = self._evaluate_q(state)
            a = self._select_action(Q, i)
            self._perform_action(a)
            if self._env.episode_done():
                self._request_new_episode()

            # sample mini-batch and perform training
            experiences = self._replay_memory.sample(DQNConfig.batch_size)
            states, actions, states_new, rewards, dones = DQNAgent.decompose_experiences(experiences)

            # compute q_targets
            q_new = self._tf_sess.run(self._tf_tn_Q, feed_dict={self._tf_tn_state: states_new})
            q_new_max = np.max(q_new, axis=1)
            q_targets = rewards + q_new_max * DQNConfig.discounted_factor * (1. - dones.astype(np.int))

            # train
            _, loss, summary = self._tf_sess.run([self._tf_pn_train, self._tf_pn_loss, summary_op],
                                                 feed_dict={self._tf_pn_actions: actions,
                                                            self._tf_pn_state: states,
                                                            self._tf_pn_Q_target: q_targets})

            # record summary
            summary_writer.add_summary(summary, global_step=i)

        # save model after training
        saver.save(self._tf_sess, save_path=model_save_path)

    def test(self, model_load_path):
        pass

    def _evaluate_q(self, state):
        assert(self._tf_sess is not None)
        assert(self._tf_pn_Q is not None)
        Q = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: state})
        return Q

    def _select_action(self, Q, i):
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
        prev_state = self._histroy_frames.copy_content()
        observation, r, done = self._env.step(action)
        observation = DQNAgent.preprocess_frame(observation)
        self._histroy_frames.record(observation)
        s = self._histroy_frames.copy_content()
        experience = (prev_state, action, s, r, done)
        self._replay_memory.add(experience)
        return done

    def _request_new_episode(self):
        observation = self._env.reset()
        self._histroy_frames.fill_with(observation)

    def _perform_random_action(self):
        action = random.randrange(0, self._env.get_num_actions())
        return self._perform_action(action)

    def _get_current_state(self):
        return self._histroy_frames.copy_content()

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


