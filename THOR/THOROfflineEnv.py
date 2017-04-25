""" THORViewGraphBuilder
This module implements routines to build a 
veiw graph from THOR environment by doing a DFS search
"""
import os
import shutil
import cPickle as pickle
import numpy as np
import robosims.server
import THORConfig as config

def unpack_thor_event(event):
	frame = event.frame
	success = event.metadata['lastActionSuccess']
	return frame, success

class ImageDB:
	def __init__(self):
		self._storage = []

	def get_size(self):
		return len(self._storage)

	def register_img(self, img):
		self._storage.append(img)

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


class EnvDB:
	def __init__(self, env_name):
		self._env_name = env_name
		self._env = robosims.server.Controller(player_screen_width=config.screen_width,
											   player_screen_height=config.screen_height,
	                                 		   darwin_build=config.darwin_build,
	                                 		   linux_build=config.linux_build,
	                                 		   x_display=config.x_display)
		self._img_db = ImageDB()
		self._pose_recorder = PoseRecorder()
		self._pose_to_observation = {}

	def build(self):
		dump_path = os.path.join(config.env_db_folder, self._env_name + '.views')

		# initialize
		self._env.start()
		event = self._env.reset(self._env_name)
		self._pose_recorder.reset()
		img, _ = unpack_thor_event(event)
		img_idx = self._img_db.register_img(img)

		# dfs
		self._dfs_traverse_scene()

		# save 
		self._img_db.optimize_memory_layout()
		blob = (self._img_db, self._pose_to_observation)
		pickle.dump(blob, open(dump_path, 'wb'))

	def _dfs_traverse_scene(self):
		for action_idx in range(len(config.supported_actions)):
			action_str = config.supported_actions[action_idx]
			event = self._env.step(dict(action=action_str))
			img, success = unpack_thor_event(event)
			if success:
				self._pose_recorder.record(action_idx)
				pose = self._pose_recorder.get_pose()
				if pose not in self._pose_to_observation:
					img_idx = self._img_db.register_img(img)
					if self._img_db.get_size() % 100 == 0:
						print '{0} images collected'.format(self._img_db.get_size())
					self._pose_to_observation[pose] = img_idx
					self._dfs_traverse_scene()


if __name__ == '__main__':
	if os.path.exists(config.env_db_folder):
		shutil.rmtree(config.env_db_folder)
	os.mkdir(config.env_db_folder)
	for env_name in config.supported_envs:
		print('bulding view graph for ' + env_name + '...')
		db = EnvDB(env_name)
		db.build()
	print('done.')