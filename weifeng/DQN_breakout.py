import gym
import numpy as np
import random
import tensorflow as tf

import matplotlib.pyplot as plt

env = gym.make("Breakout-v0")


tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions


def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 210, 160, 3])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=5)

    #conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #conv3 = maxpool2d(conv3, k=5)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
'wc1': tf.Variable(tf.random_normal([7, 7, 3, 32])),
'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
#'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
'wd1': tf.Variable(tf.random_normal([21*16*64, 1024])),
'out': tf.Variable(tf.random_normal([1024, 6])) }
biases = {
'bc1': tf.Variable(tf.random_normal([32])),
'bc2': tf.Variable(tf.random_normal([64])),
#'bc3': tf.Variable(tf.random_normal([128])),
'bd1': tf.Variable(tf.random_normal([1024])),
'out': tf.Variable(tf.random_normal([6])) }

inputs = tf.placeholder(shape=[210, 160, 3],dtype=tf.float32)

Wu = weights
Bu = biases
Qoutbefore = conv_net(inputs, Wu, Bu, 1) #Q-value for target
Qout = conv_net(inputs, weights, biases, 1) #get the up-to-date Q-value
Qoutupdate = conv_net(inputs, weights, biases, 0.5) #with dropout for updating
# Define loss and optimizer
nextQ = tf.placeholder(shape=[1,6],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Qoutupdate - nextQ))
updateModel = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


predict = tf.argmax(Qout,1)
predictbefore = tf.argmax(Qoutbefore, 1)

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
e =E = 0.5
num_episodes = 500
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    memory = Memory(10000)
    for i in range(num_episodes):
        #give value to updata parameters
        sess.run(Wu)
        sess.run(Bu)
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while True:
            j+=1
            print(j)
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs:s})

            #Get new state and reward from environment
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            print(r)
            memory.add((s, a[0], r, s1)) #add state to memory
            rAll += r
            s = s1

            samplestates = memory.sample(1)
            for sample in samplestates:
                ss, sa, sr, ss1 = sample

                #allQ is the new Q-values for update
                allQ = sess.run(Qout, feed_dict={inputs: ss})
                ia = sess.run(predict, feed_dict={inputs: ss1})
                #ia = sess.run(predict, feed_dict={inputs1: ss.reshape((1, 4))})

                #use the old parameters
                Q1 = sess.run(Qoutbefore,feed_dict={inputs:ss1})
                #Obtain maxQ' and set our target value for chosen action.
                #maxQ1 = np.max(Q1)

                maxQ1 = Q1[0, ia]

                targetQ = allQ
                targetQ[0,sa] = sr + y*maxQ1
                #Train our network using target and predicted Q values
                _,W1 = sess.run([updateModel,weights],feed_dict={inputs:ss, nextQ:targetQ})


            #s1, r, d, _ = env.step(a[0])
            #print(s1)

            if d == True:
                #Reduce chance of random action as we train the model.
                e = (10*E)/(i/2 + 10)
                break
        print("Number of total reward : ", rAll)
        jList.append(j)
        rList.append(rAll)
        print(sum(rList)/(i+1))

plt.plot(rList)
plt.show()