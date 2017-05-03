import skimage
import numpy as np
import random
import robosims.server
import THORConfig as config
import cv2
from THORTarget import THORTargetManager
from THOROfflineEnv import EnvSim

class THOREnvironment:
    def __init__(self, feat_mode = False):
        self._env = EnvSim(feat_mode = feat_mode)
        self._done = True
        self._total_episode_reward = 0
        self._env_idx = None
        self._target_idx = None
        self._target_img = None
        self._target_img_pose = None
        self._target_img_mgr = THORTargetManager(config.target_images_folder, enable_feat_load = feat_mode)
        self._step_count = 0
        self._cur_frame = None
        self._feat_mode = feat_mode

    def step(self, action_idx):
        assert(not self._done)
        prev_distance = self.get_L1_distance_to_target()
        observation, action_success = self._env.step(action_idx)
        cur_distance = self.get_L1_distance_to_target()
        self._step_count += 1
        if self._check_found_target(observation):
            self._done = True
            reward = config.reward_found
        else:
            if action_success:
                reward = config.reward_notfound
                if config.use_distance_reward:
                    if cur_distance < prev_distance:
                        reward += config.distance_decrease_reward
            else:
                reward = config.reward_notfound_notsuccess
        if self._step_count == config.episode_max_steps:
            self._done = True
        self._total_episode_reward += reward
        return observation, reward, self._done, action_success

    def reset(self, env_idx, target_idx):
        assert(self._done)
        assert(0 <= env_idx < len(config.supported_envs))
        assert(target_idx is not None)
        assert(0 <= target_idx < self.get_num_targets())
        self._env_idx = env_idx
        self._target_idx = target_idx
        env_name = config.supported_envs[self._env_idx]
        self._target_img = self._target_img_mgr.get_target_image(env_name, self._target_idx)
        self._target_img_pose = self._target_img_mgr.get_target_pose(env_name, self._target_idx)
        observation = self._env.reset(env_name)    
        if not config.diable_random_start:
            for _ in range(random.randrange(0, config.random_start + 1)):
                observation, _ = self._env.step(random.randrange(0, self.get_num_actions()))
        self._total_episode_reward = 0
        self._step_count = 0
        self._done = False
        return observation

    def reset_random(self):
        target_idx = random.randrange(0, self.get_num_targets())
        env_idx = random.randrange(0, len(config.supported_envs))
        return self.reset(env_idx, target_idx)

    def get_num_actions(self):
        return len(config.supported_actions)

    def get_num_targets(self):
        return config.targets_per_scene

    def get_env_name(self):
        return config.supported_envs[self._env_idx]

    def get_env_idx(self):
        assert(self._env_idx is not None)
        return self._env_idx

    def get_target_image(self):
        assert(self._target_img is not None)
        return self._target_img

    def get_target_feat(self):
        assert(self._feat_mode)
        return self._target_img_mgr.get_target_feat(config.supported_envs[self._env_idx], self._target_idx)

    def episode_done(self):
        return self._done

    def get_total_episode_reward(self):
        return self._total_episode_reward

    def get_steps_count(self):
        return self._step_count

    def get_L1_distance_to_target(self):
        assert(self._target_img_pose is not None)
        cur_location, _, _ = self._env.get_pose()
        target_location, _, _ = self._target_img_pose
        dist = abs(cur_location[0] - target_location[0]) + abs(cur_location[1] - target_location[1])
        return dist

    @staticmethod
    def pre_load(feat_mode, load_img_force=False):
        EnvSim.pre_load(feat_mode, load_img_force)

    def _check_found_target(self, observation):
        assert(self._target_img_pose is not None)
        return self._target_img_pose == self._env.get_pose()
