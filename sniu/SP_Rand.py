import numpy as np
import gym
import random

env = gym.make('SpaceInvaders-v0')
# Set learning parameters
num_episodes = 1000

epi_r = []

runnimg_mean = None

for i in range(num_episodes):
    env.reset()
    done = False
    one_r = []
    while not done:
        # action = env.action_space.sample()
        action = random.randint(0,5)
        new_s, reward, done, _ = env.step(action)
        # env.render()
        one_r.append(reward)


    epi_r.append(sum(one_r))

    if runnimg_mean is None:
        runnimg_mean = sum(one_r)
    else:
        runnimg_mean = 0.99*runnimg_mean + 0.01 * sum(one_r)
    print(sum(one_r), runnimg_mean)


