from torch import FloatTensor, Tensor
from memory import ReplayMemory
class Model:
    def __init__(self):
        self.net = None
        self.memory = ReplayMemory(10000)
    
    def predict(self, input):
        return 0
    
    def push_to_memory(self,  cur_state: FloatTensor, action: int, next_state: FloatTensor, reward: FloatTensor):
        self.memory.push(cur_state, action, next_state, reward)
    
    def optimize(self):
        return