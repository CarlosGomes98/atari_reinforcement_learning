import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import gym
import matplotlib.pyplot as plt

class ScreenReader:

    def __init__(self, env:gym.Env):
        self.env = env
        self.resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)
    
    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)

class BreakoutScreenReader:
    def __init__(self, env:gym.Env):
        self.env = env
        self.transform = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.Resize(200, interpolation=Image.CUBIC),
                        T.ToTensor()])
    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        self.env.render()
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = screen[20:300, 0:200]  # crop off score
        screen = torch.from_numpy(screen)
        return self.transform(screen).unsqueeze(0)