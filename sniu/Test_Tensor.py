import gym
import numpy as np
import tensorflow as tf


def resize(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, :]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[np.newaxis,:]
    return I.astype(np.float)

# Load the environment

if __name__ == "__main__":

    env = gym.make("SpaceInvaders-v0")
    s = env.reset()
    s2 = np.array(s)
    new_s, reward, d, _ = env.step(1)
    print reward
    state = tf.placeholder(tf.float32, [1,80, 80, 3], "state")
    input_fc = tf.reshape(state, [1,-1])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        value = sess.run(input_fc,feed_dict={state: resize(s)})

        print np.shape(value)


