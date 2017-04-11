import random
import numpy as np
import cPickle as pickle
from collections import deque

class HistoryBuffer:
    def __init__(self, gamma = 0.99, max_step = 1):
        assert(max_step >= 1)

        # initialize variables
        self._discount_factor = gamma

        # record reward history for normalization
        self._advantage_history_buffer = []
        self._advantage_history_buffer_length = 1000000

        # initialize buffers
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        self._value_buffer = []

        # max time step
        self._max_step = max_step

    def store_rollout(self, state, reward, action, value):
        self._state_buffer.append(state)
        self._reward_buffer.append(reward)
        self._action_buffer.append(action)
        self._value_buffer.append(value)

    def clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
        self._value_buffer = []
    
    def compute_q_value_and_advantages(self):
        N = len(self._reward_buffer)
        # compute n-step q-value
        q_value = np.zeros(N)
        r = 0
        for t in xrange(N):
            discount = 1
            max_step_reached = False
            for s in xrange(self._max_step):
                if t + s >= N:
                    print N 
                    print t + s
                    print self._reward_buffer
                    break
                q_value[t] += discount * self._reward_buffer[t+s]
                discount *= self._discount_factor
                if self._reward_buffer[t+s] != 0:
                    max_step_reached = True
                    break
            if not max_step_reached and t + self._max_step < N:
                q_value[t] += discount * self._value_buffer[t+self._max_step] 
                    
        # update advantage history
        advantage = q_value - self._value_buffer
        #self._advantage_history_buffer += advantage.tolist()
        #self._advantage_history_buffer = self._advantage_history_buffer[-self._advantage_history_buffer_length:]

        # normalize advantage
        #advantage -= np.mean(self._advantage_history_buffer)
        #advantage /= np.std(self._advantage_history_buffer)

        return q_value, advantage

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
