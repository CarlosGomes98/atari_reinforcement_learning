import random
import torch
from torch import FloatTensor
from collections import namedtuple, deque
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))   

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> torch.FloatTensor:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(islice(self.memory, item.start, item.stop, item.step))
        return self.memory[item]

class ScreenMemory():

    def __init__(self, capacity: int, h:int, w: int):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.empty_screen = torch.zeros(1, 1, h, w, dtype=torch.float)
    
    def make_state(self):
        return torch.cat(self.padded_memory(), dim=1)
    
    def padded_memory(self):
        if len(self.memory) >= self.capacity:
            return list(self.memory)
        return [self.memory[i] if i < len(self.memory) else self.empty_screen for i in range(self.capacity)]
    
    def clear(self):
        self.memory.clear()
    
    def push(self, input: FloatTensor):
        self.memory.append(input)
    
    def make_stupid_state(self):
        if len(self.memory) == 1:
            return self.memory[0] - self.memory[0]
        return self.memory[1] - self.memory[0]