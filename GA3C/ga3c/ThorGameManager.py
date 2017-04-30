""" ThorGameManager
Game Manager for Thor Environment
@Yu
"""
import sys
sys.path.append('../../THOR')
from THOREnv import THOREnvironment

class GameManager:

    @staticmethod
    def load_resources():
        THOREnvironment.pre_load(feat_mode = False)

    def __init__(self, game_name, display):
        self.game_name = game_name
        self.env = THOREnvironment(feat_mode = False)

    def reset(self):
        observation = self.env.reset_random()
        return observation

    def step(self, action):
        observation, reward, done, action_success = self.env.step(action)
        return observation, reward, done, action_success

    def get_num_actions(self):
        return self.env.get_num_actions()
