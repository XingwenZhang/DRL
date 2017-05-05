import random
import numpy as np
import cPickle as pickle
from THOR import THORConfig
from collections import deque

class HistoryBuffer:
    def __init__(self, gamma = 0.99):
        # initialize variables
        self._discount_factor = gamma

        # initialize buffers
        self._state_buffer  = np.zeros(shape = (0, 2048))
        self._reward_buffer = []
        self._action_buffer = []
        self._value_buffer = []

    def store_rollout(self, state, reward, action, value):
        self._state_buffer = np.concatenate((self._state_buffer, state), axis = 0)
        self._reward_buffer.append(reward)
        self._action_buffer.append(action)
        self._value_buffer.append(value)

    def clean_up(self):
        self._state_buffer  = np.zeros(shape = (0, 2048))
        self._reward_buffer = []
        self._action_buffer = []
        self._value_buffer = []
    
    def compute_q_value(self, final_value):
        N = len(self._reward_buffer)
        # compute q-values
        q_value = np.zeros(N)
        value = final_value * self._discount_factor
        for t in reversed(xrange(N)):
            value += self._reward_buffer[t]
            q_value[t] = value
            value *= self._discount_factor
        
        #advantage = q_value - self._value_buffer

        return q_value#, advantage

class FrameHistoryBuffer:
    """ frame history buffer maintains a fixed number of history frames
    """
    def __init__(self, image_size, buffer_size):
        assert buffer_size > 0
        assert len(image_size) == 2
        self._buffer = np.zeros((image_size[0], image_size[1], buffer_size))
        self._buffer_size = buffer_size
        self._image_size = image_size
        self._frame_received = 0

    def copy_content(self, size=None):
        assert self._frame_received >= self._buffer_size
        if size is None:
            return self._buffer.copy()
        else:
            assert(size > self._buffer_size)
            return self._buffer[:, :, 0:size].copy()

    def record(self, frame):
        if self._buffer_size > 1:
            self._buffer[:, :, 1:self._buffer_size] = self._buffer[:, :, 0:self._buffer_size-1]
        self._buffer[:, :, 0] = frame
        self._frame_received += 1

    def fill_with(self, frame):
        for _ in range(self._buffer_size):
            self.record(frame)

    def clear(self):
        self._frame_received = 0

    def get_buffer_size(self):
        return self._buffer_size

    def save(self, file_path):
        f = open(file_path, 'wb')
        dump = {}
        dump['buffer'] = self._buffer
        dump['buffer_size'] = self._buffer_size
        dump['image_size'] = self._image_size
        dump['frame_received'] = self._frame_received
        pickle.dump(dump, f)

    def load(self, file_path):
        f = open(file_path, 'rb')
        dump = pickle.load(f)
        self._buffer = dump['buffer']
        self._buffer_size = dump['buffer_size']
        self._image_size = dump['image_size']
        self._frame_received = dump['frame_received']

class ExperienceReplayMemory:
    """ Experience Replay Memory used in DQN
    """
    def __init__(self, capacity):
        self._memory = deque()
        self._count = 0
        self._capacity = capacity

    def sample(self, size=1):
        result = []
        for _ in range(size):
            result.append(self._sample_single())
        return result

    def add(self, experience):
        if self._count == self._capacity:
            self._memory.popleft()
            self._count -= 1
        self._memory.append(experience)
        self._count += 1

    def clear(self):
        self._memory.clear()
        self._count = 0

    def get_capacity(self):
        return self._capacity

    def get_grow_size(self):
        return self._count

    def save(self, file_path):
        f = open(file_path, 'wb')
        dump = {}        
        dump['memory'] = self._memory
        dump['count'] = self._count
        dump['capacity'] = self._capacity
        pickle.dump(dump, f)

    def load(self, file_path):
        f = open(file_path, 'rb')
        dump = pickle.load(f)        
        self._memory = dump['memory']
        self._count = dump['count']
        self._capacity = dump['capacity']

    def _sample_single(self):
        idx = random.randrange(0, self._count)
        return self._memory[idx]
