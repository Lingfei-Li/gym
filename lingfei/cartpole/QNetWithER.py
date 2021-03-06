
#Q-Network with experience replay

import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')


#setting up network
D = 4
H = 128
learning_rate = 0.0001
discount = 0.99
epsilon = 0.2
weight_decay = 0.1

input = tf.placeholder(tf.float32, [None, D], name='input')

layer1 = tf.contrib.layers.fully_connected(inputs=input,
                                           num_outputs=H,
                                           activation_fn=tf.nn.elu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                                           )
dropout1 = tf.nn.dropout(layer1, 0.5)

layer2 = tf.contrib.layers.fully_connected(inputs=dropout1,
                                           num_outputs=H,
                                           activation_fn=tf.nn.elu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                                           )

dropout2 = tf.nn.dropout(layer2, 0.5)

q_out = tf.contrib.layers.fully_connected(inputs=dropout2,
                                           num_outputs=2,
                                           activation_fn=tf.nn.elu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                                           )
act_out = tf.arg_max(q_out, 1)

#minibatch update
q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')

act_target = tf.placeholder(tf.float32, [None, 2], name='act_target')

q_out1 = tf.reduce_sum(tf.multiply(q_out, act_target), axis=1)

loss = tf.reduce_mean( tf.square( q_target - q_out1  ) )

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

update = optimizer.minimize(loss)



init = tf.global_variables_initializer()
episode = 1
max_episode = 300
batch_size = 500


class experience_buffer:
    def __init__(self):
        self.buffer = []
        self.maxSize = 10000

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
        state = np.reshape(env.reset(), (-1, 4))

        actions_onehot = []
        episode_buffer = []
        episode_reward = 0


        while True:
            if episode > max_episode-10:
                env.render()
            q, act = sess.run( [q_out, act_out], feed_dict={input: state} )
            act = act[0]
            act_onehot = [0, 0]

            if random.uniform(0, 1) < epsilon/episode:
                act = random.randint(0, 1)

            act_onehot[act] = 1

            actions_onehot.append(act_onehot)

            state1, reward, done, info = env.step(act)
            state1 = np.reshape(state1, (-1, 4))

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
                q_tmp = sess.run(q_out, feed_dict={input: np.vstack(exp_samples[:, 3])})
                q_tmp_max = np.max(q_tmp, axis=1)
                q_t = np.vstack(exp_samples[:, 2] + discount * q_tmp_max * exp_samples[:, 4])


                sess.run(update, feed_dict={input: np.vstack(exp_samples[:, 0]),
                                            q_target: q_t,
                                            act_target: np.vstack(exp_samples[:, 1])})





                break



    plt.plot(allRewards)
    plt.show()










