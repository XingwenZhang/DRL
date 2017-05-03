""" This module simulates THOR environment
by creating & loading a frame databse of THOR
"""
import os
import json
import numpy as np
import skimage
import skimage.transform
import robosims
import THORConfig as config
import THORUtils as utils
import cv2

import sys
sys.setrecursionlimit(1000000000)   # yeah, DFS, you know....

def unpack_thor_event(event):
    frame = skimage.transform.resize(event.frame, (config.net_input_height, config.net_input_width))
    success = event.metadata['lastActionSuccess']
    return frame, success

def get_reverse_action(action):
    assert(action in config.action_reverse_table)
    return config.action_reverse_table[action]

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

class FeatureDB:
    def __init__(self):
        self._storage = []

    def get_size(self):
        return len(self._storage)

    def register_feat(self, feat):
        self._storage.append(feat)
        return len(self._storage) - 1   

    def get_feat(self, idx):
        return self._storage[idx]

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

    def record(self, action_str):
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

    def _turn_right(self):
        for i in range(4):
            if (self._forward == PoseRecorder.right_transitions[i]).all():
                self._forward = PoseRecorder.right_transitions[(i+1)%4]
                return

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

    def _look_down(self):
        self._yaw -= 1


class EnvSim:

    _images_dbs = {}
    _feats_dbs = {}
    _pose_to_observations = {}

    def __init__(self, feat_mode=False):
        self._env_name = None
        self._img_db = None
        self._pose_recorder = None
        self._pose_to_observation = None
        self._env = None
        self._feat_mode = feat_mode
        self._feat_db = None

    def build(self):
        self._env = robosims.controller.ChallengeController(unity_path=config.binary_build)
        self._env.start()
        t = json.loads(open(config.target_folder).read())
        self._env.initialize_target(t[1])
        for self._env_name in config.supported_envs:
            # reset 
            print('building db of environment {0}...'.format(self._env_name))
            self._img_db = ImageDB()
            self._pose_recorder = PoseRecorder()
            self._pose_to_observation = {}
            # initial observation
            event = self._env.step(action=dict(action='LookDown')) # nasty hack
            self._pose_recorder.reset()
            img, _ = unpack_thor_event(event)
            img_idx = self._img_db.register_img(img)
            self._pose_to_observation[self._pose_recorder.get_pose()] = img_idx
            self._collect_all_other_views_at_cur_position()
            # dfs scene traversal
            self._dfs_traverse_scene()
            print 'total: {0} images collected'.format(self._img_db.get_size())
            # save
            dump_path = os.path.join(config.env_db_folder, self._env_name + '.env') 
            print('saving environment db to {0}'.format(dump_path))
            self._img_db.optimize_memory_layout()
            blob = (self._img_db, self._pose_to_observation)
            utils.dump(blob, open(dump_path,'wb'))

    def _dfs_traverse_scene(self):
        for action_str in config.position_actions:
            # early cut-off if the resulting pose is visited
            self._pose_recorder.record(action_str)
            future_pose = self._pose_recorder.get_pose()
            self._pose_recorder.record(get_reverse_action(action_str))
            if future_pose in self._pose_to_observation:
                continue
            event = self._env.step(dict(action=action_str))
            img, success = unpack_thor_event(event)
            if success:
                self._pose_recorder.record(action_str)
                pose = self._pose_recorder.get_pose()
                if pose not in self._pose_to_observation:
                    img_idx = self._img_db.register_img(img)
                    self._pose_to_observation[pose] = img_idx
                    self._collect_all_other_views_at_cur_position()
                    if self._img_db.get_size() % 100 == 0:
                        print '{0} images collected'.format(self._img_db.get_size())
                    self._dfs_traverse_scene()
                # back-tracking
                reverse_action_str = get_reverse_action(action_str)
                event = self._env.step(dict(action=reverse_action_str))
                _, success = unpack_thor_event(event)
                assert success
                self._pose_recorder.record(reverse_action_str)
                
    def _collect_all_other_views_at_cur_position(self):
        prev_pose = self._pose_recorder.get_pose()
        for _ in range(3):
            event = self._env.step(dict(action='RotateLeft'))
            self._pose_recorder.record('RotateLeft')
            pose = self._pose_recorder.get_pose()
            assert(pose not in self._pose_to_observation)
            img, success = unpack_thor_event(event)
            img_idx = self._img_db.register_img(img)
            assert(success)
            self._pose_to_observation[pose] = img_idx
        self._env.step(dict(action='RotateLeft'))
        self._pose_recorder.record('RotateLeft')
        cur_pose = self._pose_recorder.get_pose()
        assert(cur_pose == prev_pose)

    def reset(self, env_name, load_img_force=False):
        assert env_name in config.supported_envs, 'invalid env_name {0}'.format(env_name)
        if env_name in EnvSim._feats_dbs:
            self._feat_db = EnvSim._feats_dbs[env_name]
        else:
            if self._feat_mode:
                print('loading feature db of scene {0}...'.format(env_name))
                load_path = os.path.join(config.env_feat_folder, env_name + '.feat')
                blob = utils.load(open(load_path,'rb'))
                EnvSim._feats_dbs[env_name] = blob[0]
                EnvSim._pose_to_observations[env_name] = blob[1]
                self._feat_db = EnvSim._feats_dbs[env_name]
        if env_name in EnvSim._images_dbs:
            self._img_db = EnvSim._images_dbs[env_name]
        else:
            if (not self._feat_mode) or load_img_force:
                print('loading image db of scene {0}...'.format(env_name))
                load_path = os.path.join(config.env_db_folder, env_name + '.env')
                blob = utils.load(open(load_path,'rb'))
                EnvSim._images_dbs[env_name] = blob[0]
                EnvSim._pose_to_observations[env_name] = blob[1]
                self._img_db = EnvSim._images_dbs[env_name]
        self._env_name = env_name
        self._pose_to_observation = EnvSim._pose_to_observations[env_name]
        self._pose_recorder = PoseRecorder()
        idx = self._pose_to_observation[self._pose_recorder.get_pose()]
        if config.display:
            EnvSim.render(self._img_db.get_img(idx), 'current_frame')
        if self._feat_mode:
            return self._feat_db.get_feat(idx)
        else:
            return self._img_db.get_img(idx)

    @staticmethod
    def pre_load(feat_mode=True, load_img_force=False):
        env = EnvSim(feat_mode = feat_mode)
        for env_name in config.supported_envs:
            env.reset(env_name, load_img_force)

    def step(self, action_idx):
        assert 0 <= action_idx < len(config.supported_actions), 'invalid action_idx {0}'.format(action_idx)
        success = False
        action_str = config.supported_actions[action_idx]
        self._pose_recorder.record(action_str)
        # do a dry run and see if it succeeds
        future_pose = self._pose_recorder.get_pose()
        success = future_pose in self._pose_to_observation
        if not success:
            reverse_action_str = get_reverse_action(action_str)
            self._pose_recorder.record(reverse_action_str)
        idx = self._pose_to_observation[self._pose_recorder.get_pose()]
        if config.display:
            EnvSim.render(self._img_db.get_img(idx), 'current_frame')
        if self._feat_mode:
            return self._feat_db.get_feat(idx), success
        else:
            return self._img_db.get_img(idx), success

    def get_pose(self):
        return self._pose_recorder.get_pose()

    def get_num_views(self):
        if self._feat_mode:
            return len(self._feat_db)
        else:
            return len(self._img_db)

    @staticmethod
    def render(frame, name):
        cv2.imshow(name, frame)
        cv2.waitKey(1000)

    @staticmethod
    def get_feat_dbs():
        assert len(EnvSim._feats_dbs) == len(config.supported_envs), 'feature database is not loaded completedly.'
        return EnvSim._feats_dbs

    @staticmethod
    def get_pose_to_idx():
        assert len(EnvSim._pose_to_observations) == len(config.supported_envs), 'database is not loaded completedly.'
        return EnvSim._pose_to_observations     

