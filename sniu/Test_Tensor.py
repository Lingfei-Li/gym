import gym
import numpy as np
import tensorflow as tf
import random

def resize(I):
    """ prepro 210x160x3 state into 1x80x80 1D float vector """
    I = I[20:196]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[np.newaxis,:]
    return I.astype(np.float).ravel()
# Load the environment

if __name__ == "__main__":

    # env = gym.make("SpaceInvaders-v0")
    # s = env.reset()
    # print np.shape(s)
    # for i in range(210):
    #     print s[i,:,0].ravel()
    #
    # env = gym.make("Pong-v0")
    # print "n\n\n\n\n\n\n\n\n\n\n"
    # print np.shape(s)
    # s = env.reset()
    #
    # for i in range(210):
    #     print s[i,:,0].ravel()

    print np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
    # vn_l = range(10)
    # v_l = range(10)
    # r_l = range(10)
    #
    #
    # v_next = tf.placeholder(tf.float32, None, "v_next")
    # v = tf.placeholder(tf.float32, None, "v")
    # reward = tf.placeholder(tf.float32, None, 'reward')
    # discount = 1
    # td_error = reward + discount * v_next - v
    #
    # init = tf.global_variables_initializer()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     td = sess.run(td_error,{v_next:v_l[:5],v:v_l[:5],reward:v_l[:5]})
    #
    #     print td



    # x = range(100)
    # y = range(100,200,1)
    #
    # x_sub, y_sub = zip(*random.sample(list(zip(x, y)), 5))
    # print x_sub
    # print y_sub
    # data = np.array([])
    # for i in range(100):
    #     np.append(data,i)
    # print np.shape(data)
    # index = np.random.randint(0,100,10)
    # print data[index]
    # env = gym.make("SpaceInvaders-v0")
    # s = env.reset()
    # s2 = np.array(s)
    # new_s, reward, d, _ = env.step(1)
    # print reward
    # state = tf.placeholder(tf.float32, [1,80, 80, 3], "state")
    # input_fc = tf.reshape(state, [1,-1])
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #
    #     value = sess.run(input_fc,feed_dict={state: resize(s)})
    #
    #     print np.shape(value)
    #
    #
