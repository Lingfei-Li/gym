
# Monte carlo policy gradient

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


#network structure
input = tf.placeholder(tf.float32, [None, D], name="input")
W1 = tf.get_variable("W1", shape=[D, H], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
h1 = tf.nn.relu(tf.matmul(input, W1))

W2 = tf.get_variable("W2", shape=[H, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
probability = tf.nn.sigmoid(tf.matmul(h1, W2))

#gradient calculation
#advantage: the Q value in Monte Carlo Policy Gradient
tvars = tf.trainable_variables()
advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
input_action = tf.placeholder(tf.float32, [None, 1], name="input_action")
#the same probability is applied to a series of actions, because we don't update the probability before updating
loglik = tf.log( (input_action)*(probability) + (1-input_action)*(1-probability)  )
# loglik = tf.log(input_action*(input_action- probability) + (1 - input_action)*(input_action+ probability))
loss = -tf.reduce_mean(loglik * advantage)  #minus sign because we're doing gradient ascent
newGrad = tf.gradients(loss, tvars)

#mini batch update
W1Grad = tf.placeholder(tf.float32, name="W1Grad")
W2Grad = tf.placeholder(tf.float32, name="W2Grad")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
batchGrads = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrads, tvars))



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

episode = 0
max_episode = 5000
render = False
with tf.Session() as sess:
    sess.run(init)

    allRewards = []

    while episode < max_episode:
        print('Episode#', episode)
        gradBuffer = sess.run(tvars)
        for idx, val in enumerate(gradBuffer):
            gradBuffer[idx] = 0


        observation = env.reset()

        observations = []
        rewards = []
        actions = []
        grads = []

        done = False
        episode_reward = 0
        while not done:
            if render:
                # env.render()
                pass

            x = np.reshape(observation, [1, D])
            prob = sess.run(probability, feed_dict={input: x})

            action = 0
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

                episodeGrad = sess.run(newGrad, feed_dict={input: observations,
                                                           input_action:np.vstack(actions),
                                                           advantage: discounted_rewards})
                for idx, val in enumerate(episodeGrad):
                    gradBuffer[idx] += val

                if episode % batch_size == 0:
                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],
                                                     W2Grad: gradBuffer[1]})
                allRewards.append(np.sum(rewards))
                print('Episode reward:', np.sum(rewards))
                print('Total avg:', np.average(allRewards))
                if np.average(allRewards) > 75:
                    render = True

                break

        episode += 1



plt.plot(allRewards)
plt.ylabel('some numbers')
plt.show()













