
#Q-Network with experience replay

import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


env = gym.make('Breakout-v0')

D = 80
H = 256
learning_rate = 0.01
discount = 0.99
epsilon = 0.1
weight_decay = 0.1

class network:
    def __init__(self, scope):
        self.scope = scope
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, [None, D, D], name='input')

            self.conv1 = tf.contrib.layers.conv2d(inputs=self.input,
                                             num_outputs=16,
                                             kernel_size=8,
                                             stride=4,
                                             padding='VALID',
                                             activation_fn=tf.nn.elu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             biases_initializer=tf.constant_initializer(-1),
                                             trainable=True )

            self.conv2 = tf.contrib.layers.conv2d(inputs=self.conv1,
                                                   num_outputs=16,
                                                   kernel_size=8,
                                                   stride=4,
                                                   padding='VALID',
                                                   activation_fn=tf.nn.elu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.constant_initializer(-1),
                                                   trainable=True )

            self.conv2_flat = tf.contrib.layers.flatten(inputs=self.conv2)

            self.fc = tf.contrib.layers.fully_connected(inputs=self.conv2_flat,
                                                       num_outputs=H,
                                                       activation_fn=tf.nn.elu,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.constant_initializer(-1),
                                                       trainable=True,
                                                       )

            self.q_out = tf.contrib.layers.fully_connected(inputs=self.fc,
                                                       num_outputs=6,
                                                       activation_fn=tf.nn.elu,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       trainable=True,
                                                       )


            self.act_out = tf.arg_max(self.q_out, 1)

            self.tvars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

            #minibatch update
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')

            self.act_target = tf.placeholder(tf.float32, [None, 6], name='act_target')

            q_out1 = tf.reduce_sum(tf.multiply(self.q_out, self.act_target), axis=1)

            loss = tf.reduce_mean( tf.square( self.q_target - q_out1  ) )

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.update = optimizer.minimize(loss)


    def copyFrom(self, onlineNet):
        for idx, val in enumerate(onlineNet.tvars):
            tf.assign(self.tvars[idx], val)



onlineNet = network('online')
targetNet = network('target')
# targetNet = onlineNet

init = tf.global_variables_initializer()
episode = 1
max_episode = 3000
batch_size = 32
updateFrequencyEpisode = 100

def preproc(state):
    img_grey = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    img_downsamp = block_reduce(img_grey, (2, 2))[25:, :]
    return np.reshape(img_downsamp, (1, 80, 80))

class experience_buffer:
    def __init__(self):
        self.buffer = []
        self.maxSize = 1000000

    def merge(self, array):
        if len(self.buffer) + len(array) >= self.maxSize:
            self.buffer = self.buffer[len(array):]
        self.buffer.extend(array)

    def sample(self, num):
        return np.vstack(random.sample(self.buffer, num))


with tf.Session() as sess:
    sess.run(init)


    allRewards = []
    exp_buffer = experience_buffer()
    while episode < max_episode:
        state = preproc(env.reset())

        actions_onehot = []
        episode_buffer = []
        episode_reward = 0


        while True:
            if episode > max_episode-10:
                env.render()

            q, act = sess.run( [targetNet.q_out, targetNet.act_out], feed_dict={targetNet.input: state} )
            act = act[0]
            act_onehot = np.zeros(6)

            if random.uniform(0, 1) < epsilon:
                act = random.randint(0, 5)

            act_onehot[act] = 1

            actions_onehot.append(act_onehot)

            env.step(act)
            for k in range(0, 3):
                state1, reward, done, info = env.step(act)
            state1 = preproc(state1)

            episode_reward += reward

            term_signal = 0 if done else 1
            exp = [state, act_onehot, reward, state1, term_signal]
            episode_buffer.append(exp)

            state = state1

            if done:
                episode += 1
                exp_buffer.merge(episode_buffer)

                print('episode#', episode)
                print(episode_reward)

                allRewards.append(episode_reward)
                print('avg:', np.mean(allRewards))

                exp_samples = exp_buffer.sample(min(len(exp_buffer.buffer), batch_size))

                state1 = exp_samples[:, 3]

                # the newer q value
                q_tmp = sess.run(targetNet.q_out, feed_dict={targetNet.input: np.vstack(exp_samples[:, 3])})
                q_tmp_max = np.max(q_tmp, axis=1)
                q_t = np.vstack(exp_samples[:, 2] + discount * q_tmp_max * exp_samples[:, 4])


                sess.run(onlineNet.update, feed_dict={onlineNet.input: np.vstack(exp_samples[:, 0]),
                                            onlineNet.q_target: q_t,
                                            onlineNet.act_target: np.vstack(exp_samples[:, 1])})

                if episode % updateFrequencyEpisode == 0:
                    targetNet.copyFrom(onlineNet)


                break



    plt.plot(allRewards)
    plt.show()










