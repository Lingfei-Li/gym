import gym
import numpy as np
import random
import tensorflow as tf

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

tf.reset_default_graph()




#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[1,4],dtype=tf.float32)

W = tf.Variable(tf.random_uniform([4,2],0,0.01))
v = tf.placeholder(dtype=tf.float32)
p = tf.nn.softmax(tf.matmul(inputs, W))
J = p*v

prediction = tf.argmax(p, 1)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
al = tf.placeholder(dtype=tf.int32)
loss = -J[0, al]
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

    def empty(self):
        self.samples = []

    def getarray(self):
        return self.samples



# Set learning parameters
y = 0.99
e = E = 0.5
num_episodes = 1000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    memory = Memory(10000)
    for i in range(num_episodes):
        memory.empty()

        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #sample the path by p
        while True:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            allQ = sess.run(p,feed_dict={inputs:s.reshape((1, 4))})
            p1 = allQ.tolist()[0]
            p1 = [a/sum(p1) for a in p1]
            a = np.random.choice(a=2, size=1, p=p1)
            #print(a)

            #Get new state and reward from environment
            #if np.random.rand(1) < e:
            #    a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            memory.add((s, a[0], r)) #add state to memory
            rAll += r
            s = s1

            if d == True:
                #Reduce chance of random action as we train the model.
                e = (E*10)/(i/2 + 10)
                break

        samplestates = memory.getarray()
        vt = 0
        for index in range(len(samplestates)-1, -1, -1):
            ss, sa, sr = samplestates[index]
            vt = sr + y*vt

            _ = sess.run([updateModel],feed_dict={inputs:ss.reshape((1, 4)), v:vt, al:sa})



        print("Number of total reward : ", rAll)
        jList.append(j)
        rList.append(rAll)
        print(sum(rList)/(i+1))

plt.plot(rList)
plt.show()