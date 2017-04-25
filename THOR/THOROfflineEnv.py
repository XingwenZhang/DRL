""" This module simulates THOR environment
by creating & loading a frame databse of THOR
"""
import os
import cPickle as pickle
import numpy as np
import skimage
import skimage.transform
import robosims.server
import THORConfig as config

import sys
sys.setrecursionlimit(1000000000)	# yeah, DFS, you know....

def unpack_thor_event(event):
	frame = skimage.transform.resize(event.frame, (config.net_input_height, config.net_input_width))
	success = event.metadata['lastActionSuccess']
	return frame, success

def get_reverse_action_idx(action_idx):
	reverse_action_idx = action_idx + 1 if action_idx % 2 == 0 else action_idx - 1
	return reverse_action_idx

class ImageDB:
	def __init__(self):
		self._storage = []

	def get_size(self):
		return len(self._storage)

	def register_img(self, img):
		self._storage.append(img)
		return len(self._storage) - 1

	def get_img(self, idx):
		return skimage.img_as_float(self._storage[idx])

	def optimize_memory_layout(self):
		self._storage = np.array(self._storage)

class PoseRecorder:
	left_transitions = np.array([[0,1], [-1,0], [0,-1], [1,0]])
	right_transitions = np.array([[0,1], [1,0], [0,-1], [-1,0]])
	def __init__(self):
		self._cur_location = np.array([0,0])
		self._forward = np.array([0,1])
		self._yaw = 0

	def reset(self):
		self._cur_location = np.array([0,0])

	def record(self, action_idx):
		assert 0 <= action_idx < len(config.supported_actions), 'action_idx {0} is invalid'.format(action_idx)
		action_str = config.supported_actions[action_idx]
		if action_str == 'MoveAhead':
			self._move_ahead()
		elif action_str == 'MoveBack':
			self._move_back()
		elif action_str == 'RotateLeft':
			self._turn_left()
		elif action_str == 'RotateRight':
			self._turn_right()
		elif action_str == 'MoveLeft':
			self._move_left()
		elif action_str == 'MoveRight':
			self._move_right()
		elif action_str == 'LookUp':
			self._look_up()
		elif action_str == 'LookDown':
			self._look_down()
		else:
			assert False

	def get_location(self):
		return self._cur_location

	def get_yaw(self):
		return self._yaw

	def get_forward_direction(self):
		return self._forward

	def get_pose(self):
		return tuple(self._cur_location), self._yaw, tuple(self._forward)

	def _move_ahead(self):
		self._cur_location += self._forward

	def _move_back(self):
		self._cur_location -= self._forward

	def _turn_left(self):
		for i in range(4):
			if (self._forward == PoseRecorder.left_transitions[i]).all():
				self._forward = PoseRecorder.left_transitions[(i+1)%4]
				return
		assert False

	def _turn_right(self):
		for i in range(4):
			if (self._forward == PoseRecorder.right_transitions[i]).all():
				self._forward = PoseRecorder.right_transitions[(i+1)%4]
				return
		assert False

	def _move_left(self):
		self._turn_left();
		self._move_ahead();
		self._turn_right();

	def _move_right(self):
		self._turn_right();
		self._move_ahead();
		self._turn_left();

	def _look_up(self):
		self._yaw += 1
		assert(self._yaw <= 1)

	def _look_down(self):
		self._yaw -= 1
		assert(self._yaw >= -1)


class EnvSim:

	_images_dbs = {}
	_pose_to_observations = {}

	def __init__(self):
		self._env_name = None
		self._img_db = None
		self._pose_recorder = None
		self._pose_to_observation = None
		self._env = None

	def build(self):
		self._env = robosims.server.Controller(player_screen_width=300,
											   player_screen_height=300,
			                                   darwin_build=config.darwin_build,
			                                   linux_build=config.linux_build,
			                                   x_display=config.x_display)
		self._env.start()
		for self._env_name in config.supported_envs:
			# reset 
			print('building db of environment {0}...'.format(self._env_name))
			self._img_db = ImageDB()
			self._pose_recorder = PoseRecorder()
			self._pose_to_observation = {}
			# initial observation
			event = self._env.reset(self._env_name)
			self._pose_recorder.reset()
			img, _ = unpack_thor_event(event)
			img_idx = self._img_db.register_img(img)
			self._pose_to_observation[self._pose_recorder.get_pose()] = img_idx
			# dfs scene traversal
			self._dfs_traverse_scene()
			# save
			dump_path = os.path.join(config.env_db_folder, self._env_name + '.npz') 
			print('saving environment db to {0}'.format(dump_path))
			self._img_db.optimize_memory_layout()
			blob = (self._img_db, self._pose_to_observation)
			np.savez_compressed(dump_path, img_db=self._img_db, pose_to_observation=self._pose_to_observation)

	def _dfs_traverse_scene(self):
		for action_idx in range(len(config.supported_actions)):
			# early cut-off if the resulting pose is visited
			self._pose_recorder.record(action_idx)
			future_pose = self._pose_recorder.get_pose()
			self._pose_recorder.record(get_reverse_action_idx(action_idx))
			if future_pose in self._pose_to_observation:
				continue

			action_str = config.supported_actions[action_idx]
			event = self._env.step(dict(action=action_str))
			img, success = unpack_thor_event(event)
			if success:
				self._pose_recorder.record(action_idx)
				pose = self._pose_recorder.get_pose()
				if pose not in self._pose_to_observation:
					img_idx = self._img_db.register_img(img)
					self._pose_to_observation[pose] = img_idx
					if self._img_db.get_size() % 100 == 0:
						print '{0} images collected'.format(self._img_db.get_size())
					self._dfs_traverse_scene()
				# back-tracking
				reverse_action_idx = get_reverse_action_idx(action_idx)
				reverse_action_str = config.supported_actions[reverse_action_idx]
				self._env.step(dict(action=reverse_action_str))
				self._pose_recorder.record(reverse_action_idx)

	def reset(self, env_name):
		assert env_name in config.supported_envs, 'invalid env_name {0}'.format(env_name)
		if env_name not in EnvSim._images_dbs:
			print('loading db of scene {0}...'.format(env_name))
			load_path = os.path.join(config.env_db_folder, env_name + '.npz')
			blob = np.load(load_path)
			EnvSim._images_dbs[env_name] = blob['img_db']
			EnvSim._pose_to_observations[env_name] = blob['pose_to_observation']
		self._env_name = env_name
		self._img_db = EnvSim._images_dbs[env_name]
		self._pose_to_observation = EnvSim._pose_to_observations[env_name]
		self._pose_recorder = PoseRecorder()
		img_idx = self._pose_to_observation[self._pose_recorder.get_pose()]
		return self._img_db.get_img(img_idx)

	def pre_load(self):
		for env_name in config.supported_envs:
			self.reset(env_name)
		self._env_name = None
		self._img_db = None
		self._pose_recorder = None
		self._pose_to_observation = None

	def step(self, action_idx):
		assert 0 <= action_idx < len(config.supported_actions), 'invalid action_idx {0}'.format(action_idx)
		success = False
		self._pose_recorder.record(action_idx)
		# do a dry run and see if it succeeds
		future_pose = self._pose_recorder.get_pose()
		success = future_pose in self._pose_to_observation
		if not success:
			# reverse
			reverse_action_idx = get_reverse_action_idx(action_idx)
			self._pose_recorder.record(reverse_action_idx)
		img_idx = self._pose_to_observation[self._pose_recorder.get_pose()]
		return self._img_db.get_img(img_idx), success

	def get_pose(self):
		return self._pose_recorder.get_pose()

	def get_num_images(self):
		return len(self._img_db)

