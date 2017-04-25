

import numpy as np
import tensorflow as tf
import gym
import random


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [None, n_features], "state")
        self.state2 = tf.reshape(self.state, [-1, 80, 80,1])
        self.action = tf.placeholder(tf.float32, [None,n_actions], "act")
        self.advantege = tf.placeholder(tf.float32,None , "advantege")  # advantege
        self.actor_lrate = lr

        with tf.variable_scope('Actor'):

            '''
            Fully connected network for actor
            '''
            self.conv1 = tf.layers.conv2d(inputs=self.state2, filters=16, kernel_size=8, strides=(4, 4),
                                          padding="valid", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=32, kernel_size=4, strides=(2, 2),
                                          padding="valid", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.falt_d = tf.contrib.layers.flatten(inputs=self.conv2)

            self.hidden1 = tf.layers.dense(
                inputs=self.falt_d,
                units=256,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='hidden1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=self.hidden1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            self.prob = tf.reduce_sum((self.acts_prob * self.action)[0,:])
            log_prob = tf.log(self.prob)
            self.exp_v = tf.reduce_mean(log_prob * self.advantege)  # advantage (advantege) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.actor_lrate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, state, action, td):
        feed_dict = {self.state: state, self.action: action, self.advantege: td}
        self.sess.run([self.train_op, self.exp_v], feed_dict)


    def choose_action(self, state):
        probs = self.sess.run(self.acts_prob, {self.state: state})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


def resize(I):

    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[np.newaxis,:]
    return I.astype(np.float).ravel()

def action_list(index,n_action):

    action = np.zeros(n_action,np.int32)
    action[index] = 1
    return action

def discount_rewards(rewards):
    discount_r = np.zeros_like(rewards)
    curAdd = 0
    for i in reversed(range(0, rewards.size)):
        curAdd = curAdd * discount + rewards[i]
        discount_r[i] = curAdd
    return discount_r


if __name__ == "__main__":
    np.random.seed(2)
    tf.set_random_seed(2)  # reproducible
    MAX_EPISODE = 40000
    MAX_EP_STEPS = 2000  # maximum time step in one episode
    discount = 0.9  # reward discount in TD error

    # Set the environment
    env = gym.make('SpaceInvaders-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped
    D = 6400  # input dimensionality

    N_Action = env.action_space.n
    sess = tf.Session()
    actor = Actor(sess, n_features=D, n_actions=N_Action)
    sess.run(tf.global_variables_initializer())
    epi_record = []

    for i_episode in range(MAX_EPISODE):
        state = env.reset()
        state = resize(state)
        t = 0
        m_state = []
        m_action = []
        m_reward = []
        done = False
        while not done:
            action = actor.choose_action([state])
            state_, r, done, info = env.step(action)
            state_ = resize(state_)
            m_reward.append(r)
            m_state.append(state)
            m_action.append(action_list(action,N_Action))
            state = state_
            t += 1
            if done or t >= MAX_EP_STEPS:
                discounted = discount_rewards(np.vstack(m_reward))
                discounted -= np.mean(discounted)
                discounted /= np.std(discounted)

                actor.learn(m_state,np.reshape(np.array(m_action),[len(m_action),N_Action]),np.reshape(np.array(discounted),[-1]))
                ep_rs_sum = sum(m_reward)
                epi_record.append(ep_rs_sum)
                mean_reward = sum(epi_record)/len(epi_record) if len(epi_record) < 100 else sum(epi_record[-100:])/ 100
                print "{} {} {}".format(i_episode, sum(m_reward), mean_reward)



