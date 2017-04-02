import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('CartPole-v0')

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,4],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4,2],0,0.01))
#W = tf.Variable(tf.random_normal([4,2],stddev=0.01))
Qout = tf.matmul(inputs1, W)


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

# Set learning parameters
y = 0.99
e = 0.5
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
        while j < 300:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s.reshape((1, 4))})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1.reshape((1, 4))})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:s.reshape((1, 4)),nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 3./(i/2 + 10)
                break
        print("Number of total reward : ", rAll)
        jList.append(j)
        rList.append(rAll)
        print(sum(rList)/(i+1))