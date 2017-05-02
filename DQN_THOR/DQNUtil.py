import random
import cPickle as pickle
import numpy as np
import DQNConfig as config
from collections import deque


class FrameHistoryBuffer:
    """ frame history buffer maintains a fixed number of history frames's feature
    """
    def __init__(self):
        buffer_size = config.num_history_frames
        assert buffer_size > 0
        self._buffer = np.zeros((buffer_size, config.feature_length))
        self._buffer_size = buffer_size
        self._frame_received = 0

    def copy_content(self, size=None):
        assert self._frame_received >= self._buffer_size
        if size is None:
            return np.array(self._buffer)
        else:
            assert(size > self._buffer_size)
            return np.array(self._buffer[0:size, :])

    def record(self, frame_feature):
        assert(frame_feature is not None)
        if self._buffer_size > 1:
            self._buffer[1:self._buffer_size, :] = self._buffer[0:self._buffer_size-1, :]
        self._buffer[0, :] = frame_feature
        self._frame_received += 1

    def fill_with(self, frame_feature):
        for _ in range(self._buffer_size):
            self.record(frame_feature)

    def clear(self):
        self._frame_received = 0

    def get_buffer_size(self):
        return self._buffer_size

    def save(self, file_path):
        f = open(file_path, 'wb')
        dump = {}
        dump['buffer'] = self._buffer
        dump['buffer_size'] = self._buffer_size
        dump['frame_received'] = self._frame_received
        pickle.dump(dump, f)

    def load(self, file_path):
        f = open(file_path, 'rb')
        dump = pickle.load(f)
        self._buffer = dump['buffer']
        self._buffer_size = dump['buffer_size']
        self._frame_received = dump['frame_received']


import sys
sys.path.append('prioritized-experience-replay')
import rank_based
ExpRepMem = rank_based.Experience

class PrioritizedExperienceReplayMemory:
    """ Prioritized Experience Replay Memory (rank-based method)
    """
    def __init__(self):
        conf = {'size': config.replay_memory,
                'learn_start': 0,
                'total_step': config.max_iterations,
                'batch_size': config.batch_size,
                'alpha': config.prioritized_experience_replay_alpha,
                'beta_zero': config.prioritized_experience_replay_beta}
        self._exprepmem = ExpRepMem(conf)
        self._count = 0

    def sample(self, training_step):
        out, weights, indices = self._exprepmem.sample(training_step)
        return out, weights, indices

    def add(self, cur_state, cur_target, action, reward, new_state, done):
        data = (cur_state, cur_target, action, reward, new_state, done)
        self._exprepmem.store(data)
        self._count += 1

    def clear(self):
        assert False, 'not implemented'

    def get_capacity(self):
        return config.replay_memory

    def get_grow_size(self):
        return self._count

    def update_priority(self, indices, priorities):
        self._exprepmem.update_priority(indices, priorities)

    def save(self, file_path):
        self._exprepmem.save(file_path)

    def load(self, file_path):
        self._exprepmem.load(file_path)


class ExperienceReplayMemory:
    """ Experience Replay Memory used in DQN
    """
    def __init__(self):
        self._observations = np.empty((config.replay_memory, config.feature_length), dtype=np.float32)
        self._actions = np.empty((config.replay_memory,), dtype=np.uint8)
        self._rewards = np.empty((config.replay_memory,), dtype=np.int32)
        self._dones = np.empty((config.replay_memory,), dtype=np.bool)
        self._targets = np.empty((config.replay_memory, config.feature_length), dtype=np.float32)
        self._count = 0
        self._capacity = config.replay_memory
        self._cur_idx = -1

    def _get_state(self, idx):
        assert idx < self._count, 'observation at index {0} does not exist'.format(idx)
        if idx < config.num_history_frames - 1:
            assert self._count == self._capacity, 'not enough history frames available to construct a state at index {0}'.format(idx)
            state = np.array([self._observations[(idx - i) % self._capacity] for i in range(config.num_history_frames)])
        else:
            state = np.flip(self._observations[idx - config.num_history_frames + 1: idx + 1], axis=0)
        return state

    def sample(self, size=1):
        result = []
        for _ in range(size):
            result.append(self._sample_single())
        return result

    def add(self, cur_target, action, reward, new_observation, done):
        self._cur_idx = (self._cur_idx + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)
        self._observations[self._cur_idx] = new_observation
        self._targets[self._cur_idx] = cur_target
        self._actions[self._cur_idx] = action
        self._rewards[self._cur_idx] = reward
        self._dones[self._cur_idx] = done

    def clear(self):
        self._count = 0

    def get_capacity(self):
        return self._capacity

    def get_grow_size(self):
        return self._count

    def save(self, file_path):
        f = open(file_path, 'wb')
        dump = {}        
        dump['observations'] = self._observations
        dump['actions'] = self._actions
        dump['rewards'] = self._rewards
        dump['dones'] = self._dones
        dump['count'] = self._count
        dump['capacity'] = self._capacity
        dump['cur_idx'] = self._cur_idx
        dump['targets'] = self._targets
        pickle.dump(dump, f)

    def load(self, file_path):
        f = open(file_path, 'rb')
        dump = pickle.load(f)        
        self._observations = dump['observations']
        self._actions = dump['actions']
        self._rewards = dump['rewards']
        self._dones = dump['dones']
        self._count = dump['count']
        self._capacity = dump['capacity']
        self._cur_idx = dump['cur_idx']
        self._targets = dump['targets']

    def _sample_single(self):
        # uniformly sample a valid index
        if self._count < self._capacity:
            idx = random.randrange(config.num_history_frames, self._count)
        else:
            while True:
                idx = random.randrange(0, self._count)
                if (self._cur_idx + 1) <= idx <= (self._cur_idx + config.num_history_frames):
                    continue
                break

        prev_state = self._get_state(idx-1)
        action = self._actions[idx]
        reward = self._rewards[idx]
        done = self._dones[idx]
        target = self._targets[idx]
        new_state = self._get_state(idx)
        experience = (prev_state, target, action, new_state, reward, done)
        return experience
