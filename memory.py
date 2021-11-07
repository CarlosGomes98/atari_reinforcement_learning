import random
import torch
from torch import FloatTensor
from collections import namedtuple, deque

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

class ScreenMemory():

    def __init__(self, capacity: int, screen_size: tuple):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.empty_screen = torch.zeros(screen_size, dtype=torch.float)
    
    def make_state(self):
        return torch.cat(list(self.padded_memory()))
    
    def padded_memory(self):
        if len(self.memory) >= self.capacity:
            return self.memory
        return [self.memory[i] if i < len(self.memory) else self.empty_screen for i in range(self.capacity)]
    
    def clear(self):
        self.memory.clear()
    
    def push(self, input: FloatTensor):
        self.memory.append(input)