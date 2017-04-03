import gym
import numpy as np
import tensorflow as tf


# Load the environment

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
                                       num_outputs=2,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.contrib.layers.xavier_initializer() )

# '''
# Perceptron
# '''
#
# Q_estimate = tf.contrib.layers.fully_connected(inputs=inputs1,
#                                        num_outputs=2,
#                                        activation_fn=tf.nn.relu,
#                                        weights_initializer=tf.contrib.layers.xavier_initializer() )
#

# '''
# manual version
# '''
#
# W = tf.Variable(tf.random_uniform([4,2],0,0.01))
# Q_estimate = tf.matmul(inputs1, W)


predict = tf.argmax(Q_estimate,1)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.

nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Q_estimate))

adam = tf.train.AdamOptimizer(learning_rate=0.01)
update = adam.minimize(loss)

init = tf.global_variables_initializer()

# Set learning parameters
disconunt = .9
e = 0.1
num_episodes = 500
#create lists to contain total rewards and steps per episode
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network

            a,allQ = sess.run([predict,Q_estimate],feed_dict={inputs1:s.reshape((1, 4))})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            #Get new state and reward from environment
            new_s,reward,d,_ = env.step(a[0])

            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Q_estimate,feed_dict={inputs1:s.reshape((1, 4))})

            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = reward + disconunt*maxQ1


            #Train our network using target and predicted Q values
            _,W1 = sess.run([update,Q_estimate],feed_dict={inputs1:s.reshape((1, 4)),nextQ:targetQ})
            # _, W1 = sess.run([update, W], feed_dict={inputs1: s.reshape((1, 4)), nextQ: targetQ})

            rAll += reward
            s = new_s

            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)

                break

        jList.append(j)
        rList.append(rAll)
        print("Number of total reward : ", rAll)
        jList.append(j)
        rList.append(rAll)
        print(sum(rList) / (i + 1))
