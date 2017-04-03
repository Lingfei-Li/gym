
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
learning_rate = 0.01
discount = 0.99
epsilon = 0.5

input = tf.placeholder(tf.float32, [1, D], name='input')

# W1 = tf.get_variable('W1', [D, 2], tf.float32, tf.random_uniform_initializer(-0.01, 0.01))
W1 = tf.Variable(tf.random_uniform([4,2],0,0.01))
# h1 = tf.matmul(input, W1)
# W2 = tf.get_variable('W2', [H, 2], tf.float32, tf.contrib.layers.xavier_initializer())
q_out = tf.matmul(input, W1)


W1_d = W1
# h1_d = tf.matmul(input, W1_d)
# W2_d = W2
q_out_d = tf.matmul(input, W1_d)

act_out = tf.argmax(q_out, 1)

#update
q_target = tf.placeholder(tf.float32, [1, 2], name='q_target')

loss = tf.reduce_mean( tf.square( q_target - q_out ) )

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

update = optimizer.minimize(loss)



init = tf.global_variables_initializer()
episode = 1
max_episode = 1000
batch_size = 1
total_steps = 0

class experience_buffer:
    def __init__(self):
        self.buffer = []
        self.maxSize = 10000

    def merge(self, array):
        if len(self.buffer) + len(array) >= self.maxSize:
            self.buffer = self.buffer[len(array):]
        self.buffer.extend(array)

    def add(self, exp):
        self.buffer.append(exp)
        if len(self.buffer) > self.maxSize:
            self.buffer.pop(0)

    def sample(self, num):
        return np.vstack(random.sample(self.buffer, num))



with tf.Session() as sess:
    sess.run(init)


    allRewards = []
    exp_buffer = experience_buffer()
    while episode < max_episode:
        # sess.run([W1_d, W2_d])
        sess.run(W1_d)

        state = np.reshape(env.reset(), (-1, 4))

        actions_onehot = []
        episode_reward = 0


        while True:
            total_steps += 1
            q, act = sess.run( [q_out, act_out], feed_dict={input: state} )
            act = act[0]

            if random.uniform(0, 1) < epsilon:
                act = random.randint(0, 1)

            state1, reward, done, info = env.step(act)
            state1 = np.reshape(state1, (-1, 4))

            episode_reward += reward

            exp = [state, act, reward, state1]
            exp_buffer.add(exp)

            state = state1


            #update
            exp_samples = exp_buffer.sample(min(len(exp_buffer.buffer), batch_size))
            for sample in exp_samples:
                ss, sa, sr, ss1 = sample
                ss = ss.reshape((1, 4))
                ss1 = ss1.reshape((1, 4))

                q_old = sess.run(q_out, feed_dict={input: ss})
                act_main = sess.run(act_out, feed_dict={input: ss})
                q_double = sess.run(q_out_d, feed_dict={input: ss1 })

                #double q network
                q_t = q_old
                q_t[0, sa] = sr + discount*q_double[0, act_main]

                sess.run(update, feed_dict={input: ss,
                                            q_target: q_t
                                            })


            if done:
                episode += 1

                print('episode#', episode)
                print(episode_reward)

                allRewards.append(episode_reward)
                print('avg:', np.mean(allRewards))


                #decrease epsilon as training moves on
                epsilon = 3. / (episode / 2 + 10)


                break



    plt.plot(allRewards)
    plt.show()










