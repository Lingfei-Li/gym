import gym
import numpy as np
import random
import tensorflow as tf

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,4],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4,2],0,0.01))
Wu = W
#W = tf.Variable(tf.random_normal([4,2],stddev=0.01))
Qout = tf.matmul(inputs1, W)
Qoutu = tf.matmul(inputs1, Wu)


#One-hidden
# W = tf.Variable(tf.random_uniform([4,10],0,0.01))
# b = tf.Variable(tf.random_uniform([1, 10], 0, 0.01))
# Wo = tf.Variable(tf.random_uniform([10,2],0,0.01))
# bo = tf.Variable(tf.random_uniform([1, 2], 0, 0.01))
# Q_hidden = tf.nn.sigmoid(tf.matmul(inputs1, W) + b)
# Qout = tf.matmul(Q_hidden, Wo)

predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()



class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)



# Set learning parameters
y = 0.99
e = 0.5
num_episodes = 1000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    memory = Memory(10000)
    for i in range(num_episodes):
        sess.run(Wu)
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 300:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s.reshape((1, 4))})

            #Get new state and reward from environment
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            memory.add((s, a[0], r, s1)) #add state to memory
            rAll += r
            s = s1

            samplestates = memory.sample(1)
            for sample in samplestates:
                ss, sa, sr, ss1 = sample

                #Obtain the Q' values by feeding the new state through our network
                allQ = sess.run(Qout, feed_dict={inputs1: ss.reshape((1, 4))})
                ia = sess.run(predict, feed_dict={inputs1: ss.reshape((1, 4))})
                Q1 = sess.run(Qoutu,feed_dict={inputs1:ss1.reshape((1, 4))})
                #Obtain maxQ' and set our target value for chosen action.
                #maxQ1 = np.max(Q1)

                maxQ1 = Q1[0, ia]

                targetQ = allQ
                targetQ[0,sa] = sr + y*maxQ1
                #Train our network using target and predicted Q values
                _,W1 = sess.run([updateModel,W],feed_dict={inputs1:ss.reshape((1, 4)),nextQ:targetQ})


            #s1, r, d, _ = env.step(a[0])
            #print(s1)

            if d == True:
                #Reduce chance of random action as we train the model.
                e = 3./(i/2 + 10)
                break
        print("Number of total reward : ", rAll)
        jList.append(j)
        rList.append(rAll)
        print(sum(rList)/(i+1))

plt.plot(rList)
plt.show()