from memory import ReplayMemory
import torch
from torch import FloatTensor, nn, tensor
from memory import Transition
import torch.nn.functional as F
from net import Net

class DQN:
    def __init__(self, h, w, outputs, gamma, device):
        super(DQN, self).__init__()
        self.device = device
        self.gamma = gamma
        self.policy_network = Net(h, w, outputs, device)
        self.target_network = Net(h, w, outputs, device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters())
        self.memory = ReplayMemory(10000)
        self.batch_size = 32

    def push_to_memory(self, cur_state: FloatTensor, action: int, next_state: FloatTensor, reward: FloatTensor):
        self.memory.push(cur_state, action, next_state, reward)            

    def optimize(self):
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_action_values = self.policy_network(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(self.batch_size, device=self.device)
            # check this
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

            expected_state_action_values = reward_batch + (next_state_values * self.gamma)
            
            # Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            # whats this?
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            print(loss)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def get_best_action(self, state:torch.Tensor):
        # use argmax instead?
        prediction = self.policy_network(state).max(1)[1].view(1, 1)
        print(prediction)
        return prediction