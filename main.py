import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from memory import ReplayMemory, ScreenMemory
from actor import Actor
from model import Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

screen_size=(180, 100)
episodes = 100
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_screen():
    env.render()
    return torch.zeros(screen_size, dtype=torch.float)

episode_durations = []
total_steps = 0
past_screens = ScreenMemory(4, screen_size)
actor = Actor(0.1, Model())
for episode in range(episodes):
    env.reset()
    past_screens.clear()
    past_screens.push(get_screen())
    cur_state = past_screens.make_state()
    for step in count():
        action = actor.get_action(cur_state, total_steps)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        else:
            past_screens.push(get_screen())
            next_state = past_screens.make_state()

        actor.push_to_memory(cur_state, action, next_state, reward)
        state = next_state

        actor.optimize_model()

        if done:
            episode_durations.append(step)
            break

print('Complete')
env.render()
env.close()


