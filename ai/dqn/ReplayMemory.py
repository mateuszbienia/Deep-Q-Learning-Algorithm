from collections import deque
import random


class ReplayMemory():
    def __init__(self, min_mem, max_mem, batch_size):
        self.memory = deque([], maxlen=max_mem)
        self.min_memory = min_mem
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        if len(self.memory) < self.min_memory:
            return None
        return random.sample(self.memory, self.batch_size)
