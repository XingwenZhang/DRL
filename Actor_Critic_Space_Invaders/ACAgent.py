import random

import tensorflow as tf
import numpy as np

import ACConfig
import ACUtil
import ACNet
import ImgUtil
import TFUtil

from collections import deque

class ACAgent:
    """ Agent with Policy Network
    """
    def __init__(self, env):
        # environment related objects
        self._env = env
        
        #self._replay_memory = PGUtil.ExperienceReplayMemory(ACConfig.replay_memory)
        #self._histroy_frames = PGUtil.FrameHistoryBuffer(ACConfig.frame_size, ACConfig.num_history_frames)
        self._history_buffer = ACUtil.HistoryBuffer(max_step = ACConfig.max_step)
        self._tf_sess = None

        # tensors
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

        # saver
        self._saver = None

    def learn(self, model_save_frequency, model_save_path, check_point, use_gpu, gpu_id=None):
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.Graph().as_default():
            with tf.device(device_str):
                # build AC network
                self._init_acn()
                
                # initialize all variables
                init = tf.global_variables_initializer()

                # create auxiliary operations: summary and saver
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(ACConfig.summary_folder, self._tf_sess.graph)
                
            # initialize
            if check_point is None:
                self._tf_sess.run(init)
                episode_idx = 0
            else:
                self.load(model_save_path, check_point)
                episode_idx = check_point
            
            reward_history = deque(maxlen=100)
            state = self._request_new_episode()
            total_rewards = 0
            
            # start training
            print('Training started, please open Tensorboard to monitor the training process.')

            while episode_idx < ACConfig.max_iterations:
                # sample and perform action, store history
                action = self._select_action(state, episode_idx)
                next_state, reward, done = self._perform_action(state, action)
                total_rewards += reward
                state = next_state

                print reward
                # if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                ##  print (('ep %d: game finished, reward: %f' % (episode_idx + 1, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

                if done:
                    episode_idx += 1
                    # record reward and reset
                    reward_history.append(total_rewards)
                    print("Reward for episode {}: {}".format(episode_idx, total_rewards))
                    state = self._request_new_episode()
                    total_rewards = 0
                    

                    # start to train if episode reaches 
                    if episode_idx % ACConfig.batch_size == 0: 
                        states = self._history_buffer._state_buffer
                        actions = self._history_buffer._action_buffer
                        q_values, advantages = self._history_buffer.compute_q_value_and_advantages()
                        
                        _, average_reward, summary = \
                        self._tf_sess.run([self._tf_acn_train_op, self._tf_average_reward, summary_op],
                                            feed_dict={self._tf_acn_state: states,
                                                       self._tf_acn_action: actions,
                                                       self._tf_acn_q_value: q_values,
                                                       self._tf_acn_advantage: advantages,
                                                       self._tf_reward_history: reward_history
                                                      })
                        self._history_buffer.clean_up()

                        # record summary
                        summary_writer.add_summary(summary, global_step=episode_idx)

                    
                        print("Episode {}".format(episode_idx))
                        print("Average reward for last 100 episodes: {}".format(average_reward))

                        
                    if (episode_idx) % model_save_frequency == 0 or episode_idx == ACConfig.max_iterations:
                            print("Model saved after {} episodes".format(episode_idx))
                            self.save(model_save_path, global_step = episode_idx)

    def test(self, model_save_path, check_point, use_gpu, gpu_id=None):
        # build network
        device_str = TFUtil.get_device_str(use_gpu=use_gpu, gpu_id=gpu_id)
        with tf.device(device_str):
            self._init_acn()

        # initialize all variables from the model
        self.load(model_save_path, check_point)

        # start new episode
        episode_idx = 0
        total_rewards = 0
        state = self._request_new_episode()
        
        # perform testing
        while episode_idx <= ACConfig.max_iterations:
            # sample and perform action, store history
            action = self._select_action(state, episode_idx)
            next_state, reward, done = self._perform_action(state, action)
            total_rewards += reward
            state = next_state
            
            if done:
                episode_idx += 1
                print('total_reward received: {0}'.format(total_rewards))
                state = self._request_new_episode()
                total_rewards = 0

    
    def _init_acn(self):
        # build actor-critic network
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self._tf_sess = tf.Session(config=config)
        
        placeholders, ops, variables = ACNet.build_actor_critic_network(self._env.get_num_actions())
        self._tf_acn_state, self._tf_acn_action, self._tf_acn_q_value, self._tf_acn_advantage, self._tf_reward_history = placeholders
        self._tf_acn_train_op, self._tf_sample_action = ops
        self._tf_acn_critic_value, self._tf_average_reward = variables
        self._saver = saver = tf.train.Saver()
    
    def _evaluate_q(self, state):
        pass
        """
        assert(self._tf_sess is not None)
        assert(self._tf_pn_Q is not None)
        Q = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: state})
        return Q
        """

    def _select_action(self, state, i):
        if i >= ACConfig.final_exploration_frame:
            exploration_prob = ACConfig.final_exploration
        else:
            exploration_prob = ACConfig.initial_exploration + i * ACConfig.exploration_change_rate
        
        if random.random() < exploration_prob:
            action = random.randrange(0, self._num_actions)
        else:
            action = self._tf_sess.run(self._tf_sample_action, {self._tf_acn_state: state[np.newaxis, :]})[0][0]
        return action

    def _perform_action(self, state, action):
        # perfrom action and get next frame
        next_frame, reward, done = self._env.step(action)
        # get the value of the current state
        value = self._tf_sess.run(self._tf_acn_critic_value, feed_dict = {self._tf_acn_state: state[np.newaxis, :]})[0]
        # store roll out
        self._history_buffer.store_rollout(state, reward, action, value)
        # get next state using current state and next frame
        next_frame = self.preprocess_frame(next_frame)
        next_state = np.stack((state[:, :, 1:], next_frame), axis = 2)
        return next_state, reward, done

    def _request_new_episode(self):
        frame = self.preprocess_frame(self._env.reset())
        if len(frame.shape) == 2:
            frame = frame[:, :, np.newaxis]
        state = np.tile(frame, (1, 1, ACConfig.num_history_frames))
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
        return ImgUtil.rgb_to_luminance(ImgUtil.resize_img(frame, ACConfig.frame_size))

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