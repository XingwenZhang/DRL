""" Environment
"""

import gym
import numpy as np
import DQNConfig as config
import random


class DQNEnvironment:
    def __init__(self, environment_name, display=False, frame_skipping=True):
        self._env = gym.make(environment_name)
        self._dispaly = display
        self._done = True
        self._total_episode_reward = 0
        self._frame_skipping = frame_skipping

    def step(self, action):
        # the agent only sees and select action on every k frame (the last action is repeated in the skipped frames)
        start_lives = self._env.unwrapped.ale.lives()
        accumulated_reward = 0
        if self._frame_skipping:
            skip_interval = config.frame_skip_interval
        else:
            skip_interval = 1
        for i in range(skip_interval):
            observation, reward, self._done, info = self._env.step(action)
            accumulated_reward += reward
            if self._dispaly:
                self._env.render()
            if self._done:
                break
        end_lives = self._env.unwrapped.ale.lives()
        if config.life_drop_penalty:
            if end_lives < start_lives:
                accumulated_reward -= 1
        if config.reward_clipping:
            accumulated_reward = np.clip(accumulated_reward, -1, 1)

        self._total_episode_reward += accumulated_reward
        return observation, accumulated_reward, self._done

    def reset(self):
        observation = self._env.reset()
        self._done = False
        self._total_episode_reward = 0
        return observation

    def reset_random(self):
        observation = self.reset()
        for _ in range(random.randrange(0, config.random_start-1)):
            observation, _, done, _ = self._env.step(action=random.randrange(0, self.get_num_actions()))
            if done:
                observation = self.reset()
        return observation

    def render(self):
        self._env.render()

    def get_num_actions(self):
        return self._env.action_space.n

    def episode_done(self):
        return self._done

    def get_total_reward(self):
        return self._total_episode_reward

