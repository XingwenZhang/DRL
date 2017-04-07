""" Environment
"""

import gym
import numpy as np
import PGConfig


class PGEnvironment:
    def __init__(self, environment_name, display=False):
        self._env = gym.make(environment_name)
        self._dispaly = display
        self._done = True

    def step(self, action):
        # the agent only sees and select action on every k frame (the last action is repeated in the skipped frames)
        accumulated_reward = 0
        for i in range(PGConfig.frame_skip_interval):
            observation, reward, self._done, info = self._env.step(action)
            if self._dispaly:
                self._env.render()
            accumulated_reward += reward
            if self._done:
                break
        # as scores vary from game to game, we clipped reward at -1 and 1
        #np.clip(accumulated_reward, -1, 1)
        return observation, reward, self._done

    def reset(self):
        observation = self._env.reset()
        self._done = False
        return observation

    def render(self):
        self._env.render()

    def get_num_actions(self):
        return self._env.action_space.n

    def episode_done(self):
        return self._done

