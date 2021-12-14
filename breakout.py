import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from memory import ScreenMemory
from actor import Actor
from model import DQN
from screen import ScreenReader, BreakoutScreenReader
from utils import plot_durations

episodes = 10000
gamma = 0.999
eps_start = 0.9
eps_end = 0.01
eps_decay = 5000
target_update = 100
env = gym.make('Breakout-v0').unwrapped
n_actions = env.action_space.n

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

episode_durations = []
total_steps = 0
env.reset()
screen_reader = BreakoutScreenReader(env)
init_screen = screen_reader.get_screen()
_, _, screen_height, screen_width = init_screen.shape
past_screens = ScreenMemory(4, screen_height, screen_width)
model = DQN(screen_height, screen_width, n_actions, gamma, device)
actor = Actor(model, eps_end, eps_start, eps_decay, device, n_actions)
for episode in range(episodes):
    env.reset()
    past_screens.clear()
    past_screens.push(screen_reader.get_screen())
    cur_state = past_screens.make_state().to(device)
    for step in count():
        action = actor.get_action(cur_state, total_steps)
        _, reward, done, _ = env.step(action.item())
        total_steps += 1
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        else:
            past_screens.push(screen_reader.get_screen())
            next_state = past_screens.make_state().to(device)

        actor.push_to_memory(cur_state, action, next_state, reward)
        cur_state = next_state

        model.optimize()

        if done:
            episode_durations.append(step)
            plot_durations(episode_durations)
            break

    if episode % target_update == 0:
        model.update_target_network()

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()