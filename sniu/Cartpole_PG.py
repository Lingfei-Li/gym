

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
import random


D = 4   #input dimensionality
H = 10  #hidden units
batch_size =5
learning_rate = 0.01
discount = 0.99
epsilon = 0.1


tf.reset_default_graph()

class network:
    def __init__(self):
        self.input = tf.placeholder(shape=[None,D],dtype = tf.float16)

        self.layer1 = tf.contrib.layers.fully_connected(inpputs = input)

#network structure
input = tf.placeholder(tf.float32, [None, D], name="input")

# layer = False
layer = True
if not layer:
    W1 = tf.get_variable("W1",
                         shape=[D, H],
                         dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer())

    h1 = tf.nn.relu(tf.matmul(input, W1))

    W2 = tf.get_variable("W2",
                         shape=[H, 1],
                         dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer())
    probability = tf.nn.sigmoid(tf.matmul(h1, W2))
else:
    fc1 = tf.contrib.layers.fully_connected(inputs=input,
                                            num_outputs=H,
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

    probability = tf.contrib.layers.fully_connected(inputs=fc1,
                                            num_outputs=1,
                                            activation_fn=tf.nn.sigmoid,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

#gradient calculation
#advantage: the Q value in Monte Carlo Policy Gradient
advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
input_action = tf.placeholder(tf.float32, [None, 1], name="input_action")
#the same probability is applied to a series of actions, because we don't update the probability before updating
loglik = tf.log( (input_action)*(probability) + (1-input_action)*(1-probability)  )
# loglik = tf.log(input_action*(input_action- probability) + (1 - input_action)*(input_action+ probability))
loss = -tf.reduce_mean(loglik * advantage)  #minus sign because we're doing gradient ascent

#mini batch update
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateGrads = adam.minimize(loss)



# takes an array of rewards. apply discount to compute value
def discount_rewards(rewards):
    discount_r = np.zeros_like(rewards)
    curAdd = 0
    for i in reversed(range(0, rewards.size)):
        curAdd = curAdd * discount + rewards[i]
        discount_r[i] = curAdd
    return discount_r




init = tf.global_variables_initializer()
env = gym.make('CartPole-v0')

episode = 1
max_episode = 500
render = False
with tf.Session() as sess:
    sess.run(init)

    allRewards = []

    while episode < max_episode:
        print('Episode#', episode)

        observation = env.reset()

        advantages = []
        observations = []
        actions = []
        rewards = []
        grads = []

        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
                pass

            x = np.reshape(observation, [1, D])
            prob = sess.run(probability, feed_dict={input: x})

            if np.random.uniform() < epsilon/episode:
                action = np.random.randint(0, 1)
            if np.random.uniform() < prob:
                action = 1
            else:
                action = 0

            observation, reward, done, _ = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done:
                # # an episodeis over
                discounted_rewards = discount_rewards(np.vstack(rewards))
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)

                advantages.append(discounted_rewards)

                if episode % batch_size == 0:
                    advantages = np.reshape(advantages, (-1, 1))
                    actions = np.reshape(actions, (-1, 1))
                    sess.run(updateGrads, feed_dict={input: observations,
                                                     input_action: actions,
                                                     advantage: advantages })
                    advantages = []
                    observations = []
                    actions = []
                    rewards = []
                    grads = []

                allRewards.append(np.sum(rewards))
                print('Episode reward:', np.sum(rewards))
                print('Total avg:', np.average(allRewards))
                if np.average(allRewards) > 75:
                    render = True

                break

        episode += 1

#
#
# plt.plot(allRewards)
# plt.ylabel('some numbers')
# plt.show()
#

#
# import gym
# import numpy as np
# import tensorflow as tf
# import random
#
# class Policy():
#     def __init__(self, ):
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             # self.inputs1 = tf.placeholder(shape=[None, 4], dtype=tf.float32)
#             self.hidden = tf.contrib.layers.fully_connected(inputs=self.inputs1,
#                                                        num_outputs=10,
#                                                        activation_fn=tf.nn.relu,
#                                                        weights_initializer=tf.contrib.layers.xavier_initializer())
#             self.prob = tf.contrib.layers.fully_connected(inputs=self.hidden,
#                                                            num_outputs=1,
#                                                            activation_fn=tf.nn.relu,
#                                                            weights_initializer=tf.contrib.layers.xavier_initializer())
#             '''
#             update the policy
#             Monte Carlo PG
#             '''
#             self.advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
#             self.input_action = tf.placeholder(tf.float32, [None, 1], name="input_action")
#             self.loglik = tf.log((self.input_action) * (self.prob) + (1 - self.input_action) * (1 - self.prob))
#             self.loss = -tf.reduce_mean(self.loglik * self.advantage)  # minus sign because we're doing gradient ascent
#             # mini batch update
#             self.adam = tf.train.AdamOptimizer(learning_rate=0.01)
#             self.updateGrads = self.adam.minimize(self.loss)
#
#     def getaction(self):
#         with tf.Session(graph=self.graph) as sess:
#
#
#
#
#
# def discount_rewards(rewards,discount):
#     discount_r = np.zeros_like(rewards)
#     curAdd = 0
#     for i in reversed(range(0, rewards.size)):
#         curAdd = curAdd * discount + rewards[i]
#         discount_r[i] = curAdd
#     return discount_r
#
#
# if __name__ == "__main__":
#
#     env = gym.make('CartPole-v0')
#     policy = Policy()
#     init = tf.global_variables_initializer()
#
#     episode = 1
#     max_episode = 500
#     render = False
#     batch_size = 5
#     learning_rate = 0.01
#     discount = 0.99
#     epsilon = 0.1
#
#
#     with tf.Session() as sess:
#         sess.run(init)
#         allRewards = []
#
#         while episode < max_episode:
#             print('Episode#', episode)
#
#             observation = env.reset()
#             advantages = []
#             observations = []
#             actions = []
#             rewards = []
#             grads = []
#             done = False
#             episode_reward = 0
#
#             while not done:
#                 if render:
#                     # env.render()
#                     pass
#
#                 x = np.reshape(observation, [1, 4])
#                 prob = sess.run(policy.prob, feed_dict={policy.inputs1: x})
#
#                 if np.random.uniform() < epsilon / episode:
#                     action = np.random.randint(0, 1)
#                 if np.random.uniform() < prob:
#                     action = 1
#                 else:
#                     action = 0
#
#                 observation, reward, done, _ = env.step(action)
#
#                 observations.append(observation)
#                 actions.append(action)
#                 rewards.append(reward)
#
#                 if done:
#                     # # an episodeis over
#                     discounted_rewards = discount_rewards(np.vstack(rewards),discount)
#                     discounted_rewards -= np.mean(discounted_rewards)
#                     discounted_rewards /= np.std(discounted_rewards)
#
#                     advantages.append(discounted_rewards)
#
#                     if episode % batch_size == 0:
#                         advantages = np.reshape(advantages, (-1, 1))
#                         actions = np.reshape(actions, (-1, 1))
#                         sess.run(policy.updateGrads, feed_dict={policy.inputs1: observations,
#                                                          policy.input_action: actions,
#                                                          policy.advantage: advantages})
#                         advantages = []
#                         observations = []
#                         actions = []
#                         rewards = []
#                         grads = []
#
#                     allRewards.append(np.sum(rewards))
#                     print('Episode reward:', np.sum(rewards))
#                     print('Total avg:', np.average(allRewards))
#                     if np.average(allRewards) > 75:
#                         render = True
#
#                     break
#             episode += 1
#
#
#
#
#
