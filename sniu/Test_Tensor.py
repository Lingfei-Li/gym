import gym
import numpy as np
import tensorflow as tf


# Load the environment

print 1+5//2
env = gym.make('CartPole-v0')

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,4],dtype=tf.float32)

'''
One hidden layer
'''
hidden = tf.contrib.layers.fully_connected(inputs=inputs1,
                                       num_outputs=10,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.contrib.layers.xavier_initializer() )

#Since there are only two actions, number of outputs must be 2

Q_estimate = tf.contrib.layers.fully_connected(inputs=hidden,
                                       num_outputs=10,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.contrib.layers.xavier_initializer() )

test1,test2 = tf.split(Q_estimate, num_or_size_splits=2, axis=1)

# shape1 = tf.shape(Q_estimate)
part1 = tf.slice(Q_estimate,[0,0],[0,2])
# part2 = tf.slice(Q_estimate,[0,2],[0,3])
#
# test_p1 = tf.placeholder(shape=[1,2],dtype= tf.float32)
# FW = tf.Variable(tf.random_normal([3,1]))
# AW = tf.Variable(tf.random_normal([2,1]))
#
# test_mul_1 = tf.matmul(Q_estimate,FW)
# test_mul = tf.matmul(test_p1,AW)
# test_v = tf.contrib.slim.flatten(Q_estimate)
#

actions_sapce = np.array([0,1])
action = tf.placeholder(shape = [1],dtype=tf.int32)

actions_onehot = tf.one_hot(action, 2, dtype=tf.float32)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    print env.action_space
    test3 = sess.run([actions_onehot],feed_dict={action:[0]})
    print test3

