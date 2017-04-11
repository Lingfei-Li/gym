import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from PIL import Image


def preproc(I):
    """ prepro 210x160x3 state into 1x80x80x3 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, :]  # downsample by factor of 2
    # I[I == 144] = 0  # erase background (background type 1)
    # I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    # I = I[np.newaxis,:]
    return I.astype(np.float)


env = gym.make('SpaceInvaders-v0')
env.reset()

s, _, _, _ = env.step(0)

s = preproc(s)
s = s[:, :, 0]

plt.imshow(s)
plt.show()



