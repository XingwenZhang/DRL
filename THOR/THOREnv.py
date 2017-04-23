import skimage
import numpy as np
import random
import robosims.server
import THORConfig as config
from THORTarget import THORTargetManager

class THOREnvironment:
    def __init__(self):
        self._env = robosims.server.Controller(player_screen_width=config.screen_width,
                                               player_screen_height=config.screen_height,
                                               darwin_build=config.darwin_build,
                                               linux_build=config.linux_build,
                                               x_display=config.x_display)
        self._done = True
        self._total_episode_reward = 0
        self._env_name= None
        self._target_idx = None
        self._target_img = None
        self._target_img_mgr = THORTargetManager(config.target_images_folder)
        self._step_count = 0
        self._env.start()

    def step(self, action_idx):
        assert(not self._done)
        event = self._env.step(dict(action=config.supported_actions[action_idx]))
        action_success = event.metadata['lastActionSuccess']
        self._step_count += 1
        observation = skimage.img_as_float(event.frame)
        if self._check_found_target(observation):
            self._done = True
            reward = 1
        else:
            reward = -1
        if self._step_count == config.episode_max_steps:
            self._done = True
        self._total_episode_reward += reward
        return observation, action_success, reward, self._done

    def reset(self, env_idx=None, target_idx=None):
        assert(self._done)
        if env_idx == None:
            env_name = self._env_name
        else:
            assert(0 <= env_idx < len(config.supported_envs))
            env_name = config.supported_envs[env_idx]
        if target_idx == None:
            target_idx = self._target_idx
        assert(env_name is not None)
        assert(target_idx is not None)
        self._env_name = env_name
        self._target_idx = target_idx
        self._target_img = self._target_img_mgr.get_target_image(self._env_name, self._target_idx)
        event = self._env.reset(env_name)
        for _ in range(random.randrange(0, config.random_start + 1)):
            event = self._env.step(action=random.randrange(0, self.get_num_actions()))
        observation = skimage.img_as_float(event.frame)
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

    def get_env_name():
        assert(self._env_name is not None)
        return self._env_name

    def get_target_image(self):
        assert(self._target_img is not None)
        return self._target_img

    def episode_done(self):
        return self._done

    def get_total_episode_reward(self):
        return self._total_episode_reward

    def _check_found_target(self, observation):
        assert(self._target_img is not None)
        diff = np.sum(np.abs(self._target_img - observation))
        if diff < config.target_image_diff_threshold:
            return True 
        else:
            return False
    

