import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
import random
env = gym.make('CartPole-v0')



# hyperparameters
H = 10 # number of hidden layer neurons
batch_size = 5 # every how many episodes to do a param update?
learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
gamma = 0.99 # discount factor for reward

D = 4 # input dimensionality


tf.reset_default_graph()

class Actor:
    def __init__(self):
        with tf.variable_scope('Actor'):
            self.observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
            self.W1 = tf.get_variable("W1", shape=[D, H],
                                 initializer=tf.contrib.layers.xavier_initializer())
            self.layer1 = tf.nn.relu(tf.matmul(self.observations,self.W1))
            self.W2 = tf.get_variable("W2", shape=[H, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            self.score = tf.matmul(self.layer1,self.W2)
            self.probability = tf.nn.sigmoid(self.score)

            #From here we define the parts of the network needed for learning a good policy.
            self.input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
            self.advantages = tf.placeholder(tf.float32,name="reward_signal")

            # The loss function. This sends the weights in the direction of making actions
            # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
            self.loglik = tf.log(self.input_y*(self.probability) +
                            (1 - self.input_y)*(1- self.probability))
            self.loss = -tf.reduce_mean(self.loglik * self.advantages)

            # Once we have collected a series of gradients from multiple episodes, we apply them.
            # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
            self.adam = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # Our optimizer
            self.updateGrads = self.adam.minimize(self.loss)

            self.init = tf.global_variables_initializer()

class Critic:
    def __init__(self):
        with tf.variable_scope('Critic'):
            self.observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
            self.W1 = tf.get_variable("W1", shape=[D, H],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.layer1 = tf.nn.relu(tf.matmul(self.observations,self.W1))
            self.W2 = tf.get_variable("W2", shape=[H, 2],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.score = tf.matmul(self.layer1,self.W2)
            self.Qout = tf.nn.sigmoid(self.score)
            self.prediction = tf.arg_max(self.Qout, 1)

            #From here we define the parts of the network needed for learning a good policy.
            self.Qtarg = tf.placeholder(tf.float32,name="Qtarg")

            # The loss function. This sends the weights in the direction of making actions
            # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
            self.loss = tf.reduce_mean(self.Qtarg - self.Qout)

            # Once we have collected a series of gradients from multiple episodes, we apply them.
            # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
            self.adam = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # Our optimizer
            self.updateGrads = self.adam.minimize(self.loss)

            self.init = tf.global_variables_initializer()


class Memory:
    def __init__(self):
        self.buffer = []
        self.maxSize = 10000

    def add(self, exp):
        if len(self.buffer) >= self.maxSize:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, num):
        return np.vstack(random.sample(self.buffer, num))



running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000

actor = Actor()
critic = Critic()
memory = Memory()

train_step = 0
update_freq = 32

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(actor.init)
    sess.run(critic.init)

    observation = env.reset()  # Obtain an initial observation of the environment


    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.

    while episode_number <= total_episodes:


        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        tfprob = sess.run(actor.probability, feed_dict={actor.observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        # step the environment and get new measurements
        observation1, reward, done, info = env.step(action)
        # train_step += 1
        reward_sum += reward

        exp = [observation, action, reward, observation1, done]
        memory.add(exp)

        # if train_step % update_freq == 0 and len(memory.buffer) > batch_size:
        if len(memory.buffer) > 1:
            samples = memory.sample(1)
            for sample in samples:
                s, a, r, s1, d = sample
                s = s.reshape((1, D))
                s1 = s1.reshape((1, D))

                #Calculate advantage function and update actor
                q = sess.run(critic.Qout, feed_dict={critic.observations: s})

                sess.run(actor.updateGrads, feed_dict={actor.observations: s,
                                                        actor.input_y: [[a]],
                                                        actor.advantages: q})

                #update critic
                q1, pred = sess.run([critic.Qout, critic.prediction], feed_dict={critic.observations: s1})
                q_target = q
                q_target[0, a] = r + gamma*q1[0, pred]
                sess.run(critic.updateGrads, feed_dict={critic.observations: s,
                                                        critic.Qtarg: q_target})

        observation = observation1

        if done:
            episode_number += 1

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('Episode#', episode_number)
                print('Average reward for episode %f. \n Total average reward %f.' % (
                reward_sum / batch_size, running_reward / batch_size))

                if reward_sum / batch_size > 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'Episodes completed.')
