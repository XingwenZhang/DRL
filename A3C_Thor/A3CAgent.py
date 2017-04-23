import random

import tensorflow as tf
import numpy as np

import ThorConfig
import ThorUtil
import ThorNet
from A3CEnv import A3CEnvironment
import ImgUtil
import TFUtil
import threading

from collections import deque

class A3CAgent:
    """ A3C Agent
    """
    def __init__(self, num_threads, model_save_frequency, model_save_path, env_name, display, frame_skipping = False):
        # model related setting
        self._model_save_freq = model_save_frequency
        self._model_save_path = model_save_path

        # environment related objects
        self._num_threads = num_threads
        self._envs = [A3CEnvironment(environment_name=env_name, display=display, frame_skipping=frame_skipping) \
                      for _ in xrange(self._num_threads)]
        self._num_actions = self._envs[0].get_num_actions()
        self._episode_reward_buffer = deque(maxlen=100)
        
        #self._replay_memory = PGUtil.ExperienceReplayMemory(A3CConfig.replay_memory)
        #self._histroy_frames = PGUtil.FrameHistoryBuffer(A3CConfig.frame_size, A3CConfig.num_history_frames)
        self._tf_sess = None
        self._iter_idx = 0

        # tensors
        self._tf_global_step = None
        self._tf_acn_state = None
        self._tf_acn_action = None
        self._tf_acn_q_value = None
        self._tf_acn_advantage = None
        self._tf_acn_actor_logits = None
        self._tf_acn_critic_value = None
        self._tf_acn_train_op = None
        self._tf_sample_action = None
        self._tf_reward_history = None
        self._tf_average_reward = None
        self._tf_summary_op = None
        self._summary_writer = None

        # saver
        self._saver = None

    def learn(self, check_point, use_gpu, gpu_id=None):
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.Graph().as_default():
            with tf.device(device_str):
                # build network
                self._init_network()
                
                # initialize all variables
                init = tf.global_variables_initializer()

                # create auxiliary operations: summary and saver
                self._tf_summary_op = tf.summary.merge_all()
                self._summary_writer = tf.summary.FileWriter(A3CConfig.summary_folder, self._tf_sess.graph)
                
            # initialize or load network variables
            if check_point is None:
                self._tf_sess.run(init)
                self._iter_idx = 0
            else:
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


            
    def learner_thread(self, thread_id):
        env = self._envs[thread_id]
        state = self._request_new_episode(env)
        history_buffer = A3CUtil.HistoryBuffer()
        total_rewards = 0
        while self._iter_idx < A3CConfig.max_iterations:
            for _ in xrange(A3CConfig.max_steps):
                # sample and perform action, store history
                action = self._select_action(state, self._iter_idx)
                next_state, reward, done = self._perform_action(env, state, action, history_buffer)
                total_rewards += reward
                state = next_state
                #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                #    print (('ep %d: game finished, reward: %f' % (episode_idx + 1, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
                if done:
                    break
            
            if done:
                value = 0
                # reset
                state = self._request_new_episode(env)
                # save episode reward
                self._episode_reward_buffer.append(total_rewards)
                print("Reward for this episode: {}".format(total_rewards))
                print("Average reward for last 100 episodes: {}".format(np.mean(self._episode_reward_buffer)))
                total_rewards = 0
            else:
                value = self._tf_sess.run(self._tf_acn_critic_value, feed_dict = {self._tf_acn_state: state[np.newaxis, :]})[0]
                    

            # update models
            states = history_buffer._state_buffer
            actions = history_buffer._action_buffer
            q_values, advantages = history_buffer.compute_q_value_and_advantages(value)
            
            _, average_reward, summary = \
            self._tf_sess.run([self._tf_acn_train_op, self._tf_average_reward, self._tf_summary_op],
                              feed_dict={self._tf_global_step: self._iter_idx,
                                         self._tf_acn_state: states,
                                         self._tf_acn_action: actions,
                                         self._tf_acn_q_value: q_values,
                                         self._tf_acn_advantage: advantages,
                                         self._tf_reward_history: self._episode_reward_buffer
                                        })
            history_buffer.clean_up()

            # record summary
            self._summary_writer.add_summary(summary, global_step=self._iter_idx)
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
            action = self._select_action(state, episode_idx, test_mode = True)
            next_state, reward, done = self._perform_action(env, state, action, history_buffer)
            total_rewards += reward
            state = next_state
            
            if done:
                episode_idx += 1
                print('total_reward received: {0}'.format(total_rewards))
                history_buffer.clean_up()
                state = self._request_new_episode(env)
                total_rewards = 0

    
    def _init_network(self):
        # build a3c network
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self._tf_sess = tf.Session(config=config)
        
        placeholders, ops, variables = A3CNet.build_actor_critic_network(self._num_actions)
        self._tf_global_step, self._tf_acn_state, self._tf_acn_action, \
        self._tf_acn_q_value, self._tf_acn_advantage, self._tf_reward_history = placeholders
        self._tf_acn_train_op, self._tf_sample_action = ops
        self._tf_acn_actor_logits, self._tf_acn_critic_value, self._tf_average_reward = variables
        self._saver = saver = tf.train.Saver()
    
    def _evaluate_q(self, state):
        pass
        """
        assert(self._tf_sess is not None)
        assert(self._tf_pn_Q is not None)
        Q = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: state})
        return Q
        """

    def _select_action(self, state, i, test_mode = False):
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
                action = np.argmax(self._tf_sess.run(self._tf_acn_actor_logits, {self._tf_acn_state: state[np.newaxis, :]})[0])
            else:
                action = self._tf_sess.run(self._tf_sample_action, {self._tf_acn_state: state[np.newaxis, :]})[0][0]
        return action

    def _perform_action(self, env, state, action, history_buffer):
        # perfrom action and get next frame
        next_frame, reward, done = env.step(action)
        # get the value of the current state
        value = self._tf_sess.run(self._tf_acn_critic_value, feed_dict = {self._tf_acn_state: state[np.newaxis, :]})[0]
        # store roll out
        history_buffer.store_rollout(state, reward, action, value)
        # get next state using current state and next frame
        next_frame = self.preprocess_frame(next_frame)[:, :, np.newaxis]
        next_state = np.concatenate((state[:, :, 1:], next_frame), axis = 2)
        return next_state, reward, done

    def _request_new_episode(self, env):
        frame = self.preprocess_frame(env.reset())
        if len(frame.shape) == 2:
            frame = frame[:, :, np.newaxis]
        state = np.tile(frame, (1, 1, A3CConfig.num_history_frames))
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
        return ImgUtil.rgb_to_luminance(ImgUtil.resize_img(frame, ThorConfig.frame_size))

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