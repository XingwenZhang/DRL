import random
import numpy as np
from collections import deque

class PGHistoryBuffer:
    def __init__(self, gamma = 0.99):
        # initialize variables
        self._discount_factor = gamma

        # record reward history for normalization
        self._all_rewards = []
        self._max_reward_length = 1000000

        # initialize buffers
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []

    def store_rollout(self, state, action, reward):
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._state_buffer.append(state)

    def clean_up(self):
        self._state_buffer  = []
        self._reward_buffer = []
        self._action_buffer = []
    
    def compute_discounted_rewards(self):
        N = len(self._reward_buffer)
        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        r = 0

        for t in reversed(xrange(N)):
            # future discounted reward from now on
            if self._reward_buffer[t] != 0: r = 0 # reset the sum, since this was a game boundary (pong specific!)
            r = self._reward_buffer[t] + self._discount_factor * r
            discounted_rewards[t] = r

        # update reward history
        self._all_rewards += discounted_rewards.tolist()
        self._all_rewards = self._all_rewards[:self._max_reward_length]

        # normalize rewards
        discounted_rewards -= np.mean(self._all_rewards)
        discounted_rewards /= np.std(self._all_rewards)

        return discounted_rewards

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
