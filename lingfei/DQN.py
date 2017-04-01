

import tensorflow as tf
import numpy as np
import gym
import random
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


D = 80
learning_rate = 0.01
batch_size = 5
discount = 0.99
epsilon = 0.1

#setup network structure
input = tf.placeholder(tf.float32, [None, D, D])

# conv1 = tf.contrib.layers.conv2d(inputs=input,
#                                  num_outputs=16,
#                                  kernel_size=8,
#                                  stride=4,
#                                  padding='VALID',
#                                  activation_fn=tf.nn.relu,
#                                  weights_initializer=tf.contrib.layers.xavier_initializer())

# conv2 = tf.contrib.layers.conv2d(inputs=conv1,
#                                num_outputs=32,
#                                kernel_size=4,
#                                stride=2,
#                                padding='VALID',
#                                activation_fn=tf.nn.relu,
#                                weights_initializer=tf.contrib.layers.xavier_initializer())

# conv2_flat = tf.reshape(conv2, [-1, 8*32])
input_flat = tf.reshape(input, [-1, D*D])
fc = tf.contrib.layers.fully_connected(inputs=input_flat,
                                       num_outputs=256,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.contrib.layers.xavier_initializer() )

output = tf.contrib.layers.fully_connected(inputs=fc,
                                       num_outputs=6,
                                       activation_fn=tf.nn.sigmoid,
                                       weights_initializer=tf.contrib.layers.xavier_initializer() )

prediction = tf.arg_max(output, 1)

q_teacher = tf.placeholder(tf.float32, [None, 6])

loss = tf.reduce_mean(tf.square(q_teacher - output))

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update = adam.minimize(loss)



def preproc(state):
    img_grey = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    img_downsamp = block_reduce(img_grey, (2, 2))[25:, :]
    return np.reshape(img_downsamp, (1, 80, 80))

init = tf.global_variables_initializer()

env = gym.make("Breakout-v0")
with tf.Session() as sess:
    sess.run(init)

    episode = 0
    max_episode = 100
    statesBuffer = []
    qTargetBuffer = []
    episode_reward = 0
    while episode < max_episode:
        state_raw = env.reset()

        state = preproc(state_raw)

        while True:
            action, q = sess.run([prediction, output],
                                 feed_dict={input:state})

            statesBuffer.append(state[0])
            state_raw1, reward, done, info = env.step(action)
            state1 = preproc(state_raw1)
            episode_reward += reward

            q1 = sess.run(output, feed_dict={input:state1})
            q_target = q
            q_target[0][action] = reward + discount * q1[0][action]

            qTargetBuffer.append(q_target[0])

            if done:
                print('episode', episode)
                print(episode, episode_reward)
                if episode % batch_size == 0:
                    sess.run(update, feed_dict={input:statesBuffer,
                                                q_teacher:qTargetBuffer})
                    statesBuffer = []
                    qTargetBuffer = []

                episode += 1
                break

















