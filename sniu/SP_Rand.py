import numpy as np
import gym


env = gym.make('SpaceInvaders-v0')
# Set learning parameters
num_episodes = 1000

epi_r = []



for i in range(num_episodes):
    env.reset()
    done = False
    one_r = []
    while not done:
        action = env.action_space.sample()
        new_s, reward, done, _ = env.step(action)
        one_r.append(reward)

    print sum(one_r)
    epi_r.append(sum(one_r))


