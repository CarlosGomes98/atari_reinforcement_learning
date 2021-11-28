import math
import random
import torch
from torch import FloatTensor, tensor, nn
from model import DQN
from memory import ReplayMemory, Transition

class Actor:
    def __init__(self, model:DQN, eps_end:float, eps_start:float, eps_decay:float, device: torch.device, n_actions: int):
        self.model = model
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.device = device
        self.n_actions = n_actions
    
    def get_action(self, state, step) -> int:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * step / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.model.get_best_action(state)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def push_to_memory(self, cur_state: FloatTensor, action: int, next_state: FloatTensor, reward: FloatTensor):
        self.model.push_to_memory(cur_state, action, next_state, reward)