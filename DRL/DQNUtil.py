import random
import numpy as np
from collections import deque


class FrameHistoryBuffer:
    """ frame history buffer maintains a fixed number of history frames
    """
    def __init__(self, image_size, buffer_size):
        assert buffer_size > 0
        assert len(image_size) == 2
        self._buffer = np.zeros((image_size[0], image_size[1], buffer_size))
        self._buffer_size = buffer_size
        self._image_size = image_size
        self.frame_received = 0

    def copy_content(self, size=None):
        assert self.frame_received >= self._buffer_size
        if size is None:
            return self._buffer.copy()
        else:
            assert(size > self._buffer_size)
            return self._buffer[:, :, 0:size].copy()

    def record(self, frame):
        if self._buffer_size > 1:
            self._buffer[:, :, 1:self._buffer_size] = self._buffer[:, :, 0:self._buffer_size-1]
        self._buffer[:, :, 0] = frame
        self.frame_received += 1

    def fill_with(self, frame):
        for _ in range(self._buffer_size):
            self.record(frame)

    def clear(self):
        self.frame_received = 0

    def get_buffer_size(self):
        return self._buffer_size


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

    def _sample_single(self):
        idx = random.randrange(0, self._count)
        return self._memory[idx]
