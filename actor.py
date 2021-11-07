from torch import FloatTensor, tensor
from model import Model

class Actor:
    def __init__(self, epsilon: float, model: Model):
        self.epsilon = epsilon
        self.model = model
    
    def get_action(self, state, step) -> int:
        return tensor([0], dtype=int)
    
    def push_to_memory(self, cur_state: FloatTensor, action: int, next_state: FloatTensor, reward: FloatTensor):
        self.model.push_to_memory(cur_state, action, next_state, reward)
    
    def optimize_model(self):
        self.model.optimize()