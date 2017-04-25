# code from http://karpathy.github.io/2016/05/31/rl/


""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
#import _pickle as pickle
import gym
import tensorflow as tf

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99  # discount factor for reward

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
# D = 4


class Learner:
    def __init__(self):
        self.graph = tf.Graph()

        # Build the graph when instantiated
        with self.graph.as_default():
            #forward
            self.input = tf.placeholder(tf.float32, [None, D])
            self.x2 = tf.reshape(self.input, [-1, 80, 80, 1])

            '''
            CNN for actor
            C_P_C_P_C_P_F_Out
            '''
            self.conv1 = tf.layers.conv2d(inputs=self.x2, filters=16, kernel_size=8, strides=(4 , 4),
                                          padding="valid", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=32, kernel_size=4, strides=(2, 2),
                                          padding="valid", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())


            self.falt_d = tf.contrib.layers.flatten(inputs = self.conv2)

            self.hidden1 = tf.layers.dense(
                inputs=self.falt_d,
                units=256,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='hidden1'
            )
            self.probability = tf.layers.dense(
                inputs=self.hidden1,
                units=1,  # output units
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='policy'
            )







            #self.probability = tf.nn.sigmoid(tf.matmul(self.h1, self.W2))

            #backward
            self.advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
            self.input_action = tf.placeholder(tf.float32, [None, 1], name="input_action")
            self.loglik = tf.log( (self.input_action)*(self.probability) + (1-self.input_action)*(1-self.probability)  )
            self.loss = -tf.reduce_mean(self.loglik * self.advantage)  #minus sign because we're doing gradient ascent
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.update = self.optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            print ("Policy Graph Constructed")

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def policy_forward(self, input):
        return self.sess.run(self.probability, feed_dict={self.input:input})

    def policy_backward(self, input, advantage, input_action):
        self.sess.run(self.update, feed_dict={self.input: input, self.advantage:advantage, self.input_action: input_action})

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


# env = gym.make("CartPole-v0")
env = gym.make("Pong-v0")
learner = Learner()
prev_x = None  # used in computing the difference frame
running_reward = None
reward_sum = 0
episode_number = 0
observation = env.reset()
observations, rewards, actions = [], [], []
while True:


    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    # x = observation.reshape((-1, D))
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    prob = learner.policy_forward(x.reshape((-1, D)))
    action = 1 if np.random.uniform() < prob else 0  # roll the dice!

    # record various intermediates (needed later for backprop)
    observations.append(x)  # observation
    actions.append(action)

    # step the environment and get new measurements

    observation, reward, done, info = env.step(action+2)    #map 0/1 to 2/3 as the actual action
    reward_sum += reward

    rewards.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(np.vstack(rewards))
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            learner.policy_backward(input=observations,
                                    advantage=discounted_epr,
                                    input_action=np.vstack(actions))

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('episode sum', reward_sum)
        print('running reward', running_reward)
        reward_sum = 0
        observations, rewards, actions = [], [], []
        observation = env.reset()  # reset env
        prev_x = None
