import torch
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from memory import ScreenMemory
from actor import Actor
from model import DQN
from screen import ScreenReader

episodes = 500
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10
env = gym.make('CartPole-v0').unwrapped
n_actions = env.action_space.n

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

episode_durations = []
total_steps = 0
env.reset()
screen_reader = ScreenReader(env)
init_screen = screen_reader.get_screen()
_, _, screen_height, screen_width = init_screen.shape
past_screens = ScreenMemory(4, screen_height, screen_width)
model = DQN(screen_height, screen_width, n_actions, gamma, device)
actor = Actor(model, eps_end, eps_start, eps_decay, device, n_actions)
for episode in range(episodes):
    env.reset()
    past_screens.clear()
    past_screens.push(screen_reader.get_screen())
    past_screens.push(screen_reader.get_screen())
    cur_state = past_screens.make_stupid_state()
    for step in count():
        action = actor.get_action(cur_state, total_steps)
        _, reward, done, _ = env.step(action.item())
        total_steps += 1
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        else:
            past_screens.push(screen_reader.get_screen())
            next_state = past_screens.make_stupid_state()

        actor.push_to_memory(cur_state, action, next_state, reward)
        cur_state = next_state

        model.optimize()

        if done:
            episode_durations.append(step)
            plot_durations()
            break

    if episode % target_update == 0:
        model.update_target_network()

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()