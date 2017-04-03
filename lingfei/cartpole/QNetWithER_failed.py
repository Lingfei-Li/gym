
#Q-Network with experience replay

import gym
import tensorflow as tf
import numpy as np
import random


env = gym.make('CartPole-v0')


#setting up network
D = 4
H = 20
learning_rate = 0.01
discount = 0.99
epsilon = 0.1

input = tf.placeholder(tf.float32, [None, D], name='input')

fc1 = tf.contrib.layers.fully_connected(inputs=input,
                                        num_outputs=H,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                        num_outputs=H,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                        num_outputs=H,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

q_val = tf.contrib.layers.fully_connected(inputs=fc3,
                                        num_outputs=2,
                                        activation_fn=tf.nn.sigmoid,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
predict = tf.arg_max(q_val, 1)



#mini-batch update
q_teacher = tf.placeholder(tf.float32, [None], name='q_teacher')
input_actions = tf.placeholder(tf.int32, [None], name='input_action')
input_one_hot = tf.one_hot(input_actions, 2)

# error = (q_teacher - q_val) * input_one_hot

# q_val = q_val * input_one_hot
q_val = tf.reduce_sum(tf.matmul(q_val, input_one_hot), axis=1)
error = q_teacher - q_val


loss = tf.reduce_mean(tf.square(error))
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update = adam.minimize(loss)


init = tf.global_variables_initializer()
episode = 0
max_episode = 100000
batch_size = 50


#each experience is <state, action, reward, next_state, done>
#done is included because some environments distinguish this
class experience_buffer:
    def __init__(self):
        self.buffer = []
        self.size = 50000
    def add(self, experience):
        if len(self.buffer) + len(experience) > self.size:
            self.buffer = self.buffer[len(experience):]
        self.buffer.extend(experience)
    def sample(self, batch_size):
        samples = np.array(random.sample(self.buffer, batch_size))
        return np.reshape(samples, [batch_size, 5])


with tf.Session() as sess:
    sess.run(init)

    episode_rewards = []
    states = []
    rewards = []
    actions = []
    q_targets = []
    exp_buffer = experience_buffer()

    while episode < max_episode:
        state = env.reset()
        state = np.reshape(state, (-1, 4))
        episode_reward = 0
        episode_exp_buffer = experience_buffer()

        while True:
            #calculate Q values, and choose the action with the max Q value
            action, q = sess.run([predict, q_val], feed_dict={input: state})
            action = action[0]
            # print(action, q)
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            action = int(action)

            #step the environment with the action()
            state1, reward, done, info = env.step(action)
            state1 = np.reshape(state1, (-1, 4))
            episode_reward += reward

            #get the max new Q value with the stepped environment.
            # set target q value to update the network
            q1= sess.run(q_val, feed_dict={input: state1})
            q1max= np.max(q1)
            q_target = q
            q_target[0, action] = discount*q1max + reward

            q_targets.append(q_target[0])
            states.append(state[0])

            exp = np.array([state, action, reward, state1, done])
            episode_exp_buffer.add(np.reshape((exp), [1, 5]))

            state = state1

            if done:
                episode_rewards.append(episode_reward)
                print('episode', episode)
                print('reward', episode_reward)
                print('avg', np.mean(episode_rewards))
                episode += 1

                if len(exp_buffer.buffer) > batch_size:
                    samples = exp_buffer.sample(batch_size=batch_size)

                    q1 = sess.run(q_val, feed_dict={input: np.vstack(samples[:, 3])})
                    q1max = np.max(q1)

                    q_targets = samples[:, 2] + discount * q1max

                    q_val, one_hot = sess.run([q_val, input_one_hot], feed_dict={input:np.vstack(samples[:, 0]),
                                                input_actions: samples[:, 1],
                                                q_teacher: q_targets})

                    print(q_val)
                    print(one_hot)

                    exit(0)

                    sess.run(update, feed_dict={input:np.vstack(samples[:, 0]),
                                                input_actions: np.vstack(samples[:, 1]),
                                                q_teacher: q_targets})

                states = []
                actions = []
                rewards = []
                q_targets = []

                exp_buffer.add(episode_exp_buffer.buffer)
                break
















