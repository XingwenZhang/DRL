""" This module implements a utility class that manages the target image
"""
import os
import shutil
import THORConfig as config
import skimage
import skimage.io
import skimage.exposure
import numpy as np
import random
import cPickle as pickle
from THOROfflineEnv import EnvSim

class THORTargetManager:
	def __init__(self, target_images_folder, enable_feat_load = False):
		self.target_images_folder = target_images_folder
		self.feat_dbs = None
		if enable_feat_load:
			self.feat_dbs = THORTargetManager._load_feature_db()
			self.pose_to_view_idx = THORTargetManager._load_pose_to_view_idx()

	def get_target_image(self, env_name, target_idx):
		folder = os.path.join(self.target_images_folder, env_name)
		file_path = os.path.join(self.target_images_folder, env_name, str(target_idx) + '.png')
		return skimage.img_as_float(skimage.io.imread(file_path))

	def get_target_pose(self, env_name, target_idx):
		folder = os.path.join(self.target_images_folder, env_name)
		file_path = os.path.join(self.target_images_folder, env_name, str(target_idx) + '.pose')
		return pickle.load(open(file_path, 'rb'))

	def get_target_feat(self, env_name, target_idx):
		assert self.feat_dbs is not None, 'Did you forget to construct THORTargetManager with enable_feat_load set to True?'
		assert self.pose_to_view_idx is not None, 'Did you forget to construct THORTargetManager with enable_feat_load set to True?'
		assert(env_name in self.feat_dbs)
		assert(env_name in self.pose_to_view_idx)
		pose = self.get_target_pose(env_name, target_idx)
		view_idx = self.pose_to_view_idx[env_name][pose]
		return self.feat_dbs[env_name].get_feat(view_idx)

	def collect_target_images(self, env_name):
		"""
		Find all target images of the scene and save them to dump_folder
		"""
		dump_folder = os.path.join(self.target_images_folder, env_name)
		if not os.path.exists(dump_folder):
			os.mkdir(dump_folder)
		num_images = config.targets_per_scene
		env = EnvSim()
		env.reset(env_name)
		target_images = np.empty((num_images, config.net_input_height, config.net_input_width, 3))
		target_images_poses = []
		i = 0
		while i < num_images:
			action_idx = random.randrange(0, len(config.supported_actions))
			action_str = config.supported_actions[action_idx]
			max_steps = 10 if action_str.startswith('Move') else 2
			for _ in range(random.randrange(1, max_steps)):
				frame, success = env.step(action_idx)
				if not success:
					break
			# check low contrast
			if skimage.exposure.is_low_contrast(frame):
				continue
			# check duplication in terms of appearance
			if i !=0:
				if (np.sum(np.abs(target_images[0:i] - frame), axis=(1,2,3)) < config.target_image_diff_threshold).any():
					continue
			target_images[i] = frame
			target_images_poses.append(env.get_pose())
			i+=1
			if i % 10==0:
				print(str(i) + ' target images collected.')
		# dump target images and poses
		assert(len(target_images_poses) == num_images)
		for i in range(num_images):
			skimage.io.imsave(os.path.join(dump_folder, str(i) + '.png'), target_images[i])
			pickle.dump(target_images_poses[i], open(os.path.join(dump_folder, str(i) + '.pose'), 'wb'))

	@staticmethod
	def _load_feature_db():
		env = EnvSim(feat_mode=True)
		env.pre_load()
		return EnvSim.get_feat_dbs()

	@staticmethod
	def _load_pose_to_view_idx():
		env = EnvSim(feat_mode=True)
		env.pre_load()
		return EnvSim.get_pose_to_idx()
