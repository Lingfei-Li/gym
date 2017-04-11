# import gym
# import numpy as np
# import tensorflow as tf
# import random
#
# def resize(I):
#     """ prepro 210x160x3 state into 1x80x80 1D float vector """
#     I = I[20:196]  # crop
#     I = I[::2, ::2, 0]  # downsample by factor of 2
#     I[I == 144] = 0  # erase background (background type 1)
#     I[I == 109] = 0  # erase background (background type 2)
#     I[I != 0] = 1  # everything else (paddles, ball) just set to 1
#     I = I[np.newaxis,:]
#     return I.astype(np.float).ravel()
# # Load the environment
#
# if __name__ == "__main__":
#
#     # env = gym.make("SpaceInvaders-v0")
#     # s = env.reset()
#     # print np.shape(s)
#     # for i in range(210):
#     #     print s[i,:,0].ravel()
#     #
#     # env = gym.make("Pong-v0")
#     # print "n\n\n\n\n\n\n\n\n\n\n"
#     # print np.shape(s)
#     # s = env.reset()
#     #
#     # for i in range(210):
#     #     print s[i,:,0].ravel()
#
#     print np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
#     # vn_l = range(10)
#     # v_l = range(10)
#     # r_l = range(10)
#     #
#     #
#     # v_next = tf.placeholder(tf.float32, None, "v_next")
#     # v = tf.placeholder(tf.float32, None, "v")
#     # reward = tf.placeholder(tf.float32, None, 'reward')
#     # discount = 1
#     # td_error = reward + discount * v_next - v
#     #
#     # init = tf.global_variables_initializer()
#     #
#     # with tf.Session() as sess:
#     #     sess.run(init)
#     #     td = sess.run(td_error,{v_next:v_l[:5],v:v_l[:5],reward:v_l[:5]})
#     #
#     #     print td
#
#
#
#     # x = range(100)
#     # y = range(100,200,1)
#     #
#     # x_sub, y_sub = zip(*random.sample(list(zip(x, y)), 5))
#     # print x_sub
#     # print y_sub
#     # data = np.array([])
#     # for i in range(100):
#     #     np.append(data,i)
#     # print np.shape(data)
#     # index = np.random.randint(0,100,10)
#     # print data[index]
#     # env = gym.make("SpaceInvaders-v0")
#     # s = env.reset()
#     # s2 = np.array(s)
#     # new_s, reward, d, _ = env.step(1)
#     # print reward
#     # state = tf.placeholder(tf.float32, [1,80, 80, 3], "state")
#     # input_fc = tf.reshape(state, [1,-1])
#     #
#     # init = tf.global_variables_initializer()
#     # with tf.Session() as sess:
#     #     sess.run(init)
#     #
#     #     value = sess.run(input_fc,feed_dict={state: resize(s)})
#     #
#     #     print np.shape(value)
#     #
#     #



import numpy as np
import tensorflow as tf
import gym
import random


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [None, n_features], "state")
        self.action = tf.placeholder(tf.int32, [None,n_actions], "act")
        self.advantege = tf.placeholder(tf.float32,None , "advantege")  # advantege
        self.actor_lrate = lr

        with tf.variable_scope('Actor'):

            '''
            Fully connected network for actor
            '''

            self.hidden1 = tf.layers.dense(
                inputs=self.state,
                units=20,  # number of hidden units
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


    def choose_action(self, state):
        probs = self.sess.run(self.acts_prob, {self.state: state})  # get probabilities for all actions
        return probs


def resize(I):

    I = I[20:196]  # crop
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
    MAX_EPISODE = 3000
    MAX_EP_STEPS = 2000  # maximum time step in one episode
    discount = 0.9  # reward discount in TD error

    # Set the environment
    env = gym.make('SpaceInvaders-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped
    D = 7040  # input dimensionality

    N_Action = env.action_space.n
    sess = tf.Session()
    actor = Actor(sess, n_features=D, n_actions=N_Action)
    sess.run(tf.global_variables_initializer())
    epi_record = []


    state = env.reset()
    state = resize(state)
    t = 0
    print actor.choose_action([state]) * action_list(1,6)
    print np.shape(actor.choose_action([state]))




