""" Environment
"""
import sys
sys.path.append("/home/mscvproject/users/yumao/vlr/gym/")
import gym
import numpy as np
import A3CConfig

class A3CEnvironment:
    def __init__(self, environment_name, display=False, frame_skipping=False):
        self._env = gym.make(environment_name)
        self._dispaly = display
        self._done = True
        self._total_episode_reward = 0
        self._frame_skipping = frame_skipping

    def step(self, action):
        # the agent only sees and select action on every k frame (the last action is repeated in the skipped frames)
        accumulated_reward = 0
        if self._frame_skipping:
            skip_interval = A3CConfig.frame_skip_interval
        else:
            skip_interval = 1
        for i in range(skip_interval):
            observation, reward, self._done, info = self._env.step(action)
            if self._dispaly:
                self._env.render()
            accumulated_reward += reward
            if self._done:
                break
        # as scores vary from game to game, we clipped reward at -1 and 1
        #accumulated_reward = np.clip(accumulated_reward, -1, 1)
        #reward = np.clip(accumulated_reward, -1, 1)

        self._total_episode_reward += accumulated_reward
        return observation, accumulated_reward, self._done

    def reset(self):
        observation = self._env.reset()
        self._done = False
        self._total_episode_reward = 0
        return observation

    def render(self):
        self._env.render()

    def get_num_actions(self):
        return self._env.action_space.n

    def episode_done(self):
        return self._done
    
    def get_total_reward(self):
        return self._total_episode_reward

