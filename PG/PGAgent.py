import random

import tensorflow as tf
import numpy as np

import PGConfig
import PGUtil
import PGNet
import ImgUtil

from collections import deque

class PGAgent:
    """ Agent with Policy Network
    """
    def __init__(self, env):
        # environment related objects
        self._env = env
        self._num_actions = 2

        #self._replay_memory = PGUtil.ExperienceReplayMemory(PGConfig.replay_memory)
        #self._histroy_frames = PGUtil.FrameHistoryBuffer(PGConfig.frame_size, PGConfig.num_history_frames)
        self._history_buffer = PGUtil.PGHistoryBuffer()
        self._tf_sess = None

        # tensors
        self._tf_pn_state = None
        self._tf_pn_actions = None
        self._tf_pn_advantage = None
        self._tf_pn_loss = None
        self._tf_pn_logits = None
        self._tf_pn_train_op = None
        self._tf_sample_action = None
        self._tf_reward_history = None
        self._tf_average_reward = None

    def learn(self, model_save_frequency, model_save_path, use_gpu, gpu_id=None):
        if use_gpu:
            device_id = gpu_id if gpu_id is not None else 0
            device_str = '/gpu:' + str(device_id)
        else:
            device_str = '/cpu:0'
        with tf.Graph().as_default():
            with tf.device(device_str):
                # build PG network
                config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
                self._tf_sess = tf.Session(config=config)
                
                placeholders, ops, variables = PGNet.build_pn(self._num_actions)
                self._tf_pn_state, self._tf_pn_actions, self._tf_pn_advantage, self._tf_reward_history = placeholders
                self._tf_pn_train_op, self._tf_sample_action = ops
                self._tf_pn_loss, self._tf_average_reward =  variables

                # initialize all variables
                init = tf.global_variables_initializer()

                # create auxiliary operations: summary and saver
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(PGConfig.summary_folder, self._tf_sess.graph)
                saver = tf.train.Saver()
            
            # initialize network variables
            self._tf_sess.run(init)

            # start training
            print('Training started, please open Tensorboard to monitor the training process.')
            # setup variables
            reward_history = deque(maxlen=100)
            state = self._request_new_episode()
            total_rewards = 0
            episode_idx = 0
            #### add code to load model
            """
            if FLAGS.ckpt != None:
                print("load check point")
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt-')
                checkpoint_file += FLAGS.ckpt
                saver.restore(sess, checkpoint_file)
                episode_idx = int(FLAGS.ckpt)
            """

            while episode_idx < PGConfig.max_iterations:
                # sample and perform action, store history
                action = self._select_action(state, episode_idx)
                next_state, reward, done = self._perform_action(state, action)
                total_rewards += reward
                state = next_state

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
                    if episode_idx % PGConfig.batch_size == 0: 
                        states = self._history_buffer._state_buffer
                        actions = self._history_buffer._action_buffer
                        advantages = self._history_buffer.compute_discounted_rewards()

                        _, average_reward, summary = \
                        self._tf_sess.run([self._tf_pn_train_op, self._tf_average_reward, summary_op],
                                            feed_dict={self._tf_pn_state: states,
                                                        self._tf_pn_actions: actions,
                                                        self._tf_pn_advantage: advantages,
                                                        self._tf_reward_history: reward_history
                                                        })
                        self._history_buffer.clean_up()

                        # record summary
                        summary_writer.add_summary(summary, global_step=episode_idx)

                    
                        print("Episode {}".format(episode_idx))
                        print("Average reward for last 100 episodes: {}".format(average_reward))

                        """
                        if (episode_idx + 1) % 100 == 0 or (episode_idx + 1) == FLAGS.max_episode:
                            print("Environment {} saved after {} episodes".format(env_name, episode_idx+1))
                            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_file,
                                        global_step=episode_idx + 1)
                        """

    def test(self, model_load_path):
        pass

    def _evaluate_q(self, state):
        pass
        """
        assert(self._tf_sess is not None)
        assert(self._tf_pn_Q is not None)
        Q = self._tf_sess.run(self._tf_pn_Q, feed_dict={self._tf_pn_state: state})
        return Q
        """

    def _select_action(self, state, i):
        if i >= PGConfig.final_exploration_frame:
            exploration_prob = PGConfig.final_exploration
        else:
            exploration_prob = PGConfig.initial_exploration + i * PGConfig.exploration_change_rate
        
        if random.random() < exploration_prob:
            action = random.randrange(0, self._num_actions)
        else:
            action = self._tf_sess.run(self._tf_sample_action, {self._tf_pn_state: state[np.newaxis, :]})[0][0]

        return action

    def _perform_action(self, state, action):
        next_frame, reward, done = self._env.step(action + 2) # pong-specific
        self._history_buffer.store_rollout(state, action, reward)
        next_state = self.preprocess_frame(next_frame)
        return next_state, reward, done

    def _request_new_episode(self):
        state = self.preprocess_frame(self._env.reset())
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

    @staticmethod
    def preprocess_frame(frame):
        return ImgUtil.pg_preprocess(frame)

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