

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import tensorflow as tf
import random

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
actor_learning_rate = 0.001
critic_learning_rate = 0.01
gamma = 0.9  # discount factor for reward

# model initialization
# D = 80 * 80  # input dimensionality: 80x80 grid
D = 4

class Actor:
    def __init__(self):
        self.graph = tf.Graph()

        # Build the graph when instantiated
        with self.graph.as_default():
            #forward
            self.input = tf.placeholder(tf.float32, [None, D])
            self.W1 = tf.get_variable('W1', dtype=tf.float32, shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
            self.h1 = tf.nn.relu(tf.matmul(self.input, self.W1))
            self.W2 = tf.get_variable('W2', dtype=tf.float32, shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
            self.probability = tf.nn.sigmoid(tf.matmul(self.h1, self.W2))

            #backward
            self.advantage = tf.placeholder(tf.float32, None, name="advantage")
            self.input_action = tf.placeholder(tf.float32, None, name="input_action")
            self.loglik = tf.log( (self.input_action)*(self.probability) + (1-self.input_action)*(1-self.probability)  )
            self.loss = -tf.reduce_mean(self.loglik * self.advantage)  #minus sign because we're doing gradient ascent
            self.optimizer = tf.train.AdamOptimizer(learning_rate=actor_learning_rate)
            self.update = self.optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            print ("Actor Graph Constructed")

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def policy_forward(self, input):
        return self.sess.run(self.probability, feed_dict={self.input:input})

    def policy_backward(self, input, advantage, input_action):
        self.sess.run(self.update, feed_dict={self.input: input, self.advantage:advantage, self.input_action: input_action})

    def policy_backward_single(self, input, advantage, input_action):
        input = input.reshape((-1, D))
        self.sess.run(self.update, feed_dict={self.input: input, self.advantage:advantage, self.input_action: input_action})

class Critic:
    def __init__(self):
        self.graph = tf.Graph()

        # Build the graph when instantiated
        with self.graph.as_default():
            #forward
            self.input = tf.placeholder(tf.float32, [None, D])
            self.W1 = tf.get_variable('critic_W1', dtype=tf.float32, shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
            self.h1 = tf.nn.relu(tf.matmul(self.input, self.W1))
            self.W2 = tf.get_variable('critic_W2', dtype=tf.float32, shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.matmul(self.h1, self.W2)

            #backward
            self.v_ = tf.placeholder(tf.float32, [None, 1], 'v_')
            self.rewards = tf.placeholder(tf.float32, None, 'r')
            self.td_error = self.rewards + (gamma * self.v_ - self.v)
            self.loss = tf.square(self.td_error)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=critic_learning_rate)
            self.update = self.optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            print ("Critic Graph Constructed")

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def learn(self, s, r, s_):
        s = s.reshape((-1, D))
        s_ = s_.reshape((-1, D))
        v_ = self.sess.run(self.v, feed_dict={self.input: s_})
        _, td_error = self.sess.run([self.update, self.td_error], feed_dict={self.input:s,
                                                                             self.v_: v_,
                                                                             self.rewards:r})
        return td_error


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


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



env = gym.make("CartPole-v0")
# env = gym.make("Pong-v0")
actor = Actor()
critic = Critic()
mem = Memory()
prev_x = None  # used in computing the difference frame
running_reward = None
reward_sum = 0
episode_number = 0
observation = env.reset()
while True:
    # cur_x = prepro(observation)
    # x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    # prev_x = cur_x
    x = observation.reshape((-1, D))


    prob = actor.policy_forward(x.reshape((-1, D)))
    print(prob)
    action = 1 if np.random.uniform() < prob else 0  # roll the dice!


    observation_, reward, done, info = env.step(action)    #map 0/1 to 2/3 as the actual action
    reward_sum += reward


    s = observation.reshape((-1, D))
    s_ = observation_.reshape((-1, D))

    td_error = critic.learn(s, reward, s_)

    actor.policy_backward_single(input=s,
                                 advantage=td_error,
                                 input_action=action)


    observation = observation_

    if done:  # an episode finished
        episode_number += 1

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('episode sum', reward_sum)
        print('running reward', running_reward)
        reward_sum = 0
        observations, rewards, actions = [], [], []
        observation = env.reset()  # reset env
        prev_x = None
