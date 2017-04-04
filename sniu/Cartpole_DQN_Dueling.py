import gym
import numpy as np
import tensorflow as tf
import random
# Load the environment

env = gym.make('CartPole-v0')


class Qnetwork():
    def __init__(self, ):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        self.inputs1 = tf.placeholder(shape=[None, 4], dtype=tf.float32)

        self.hidden = tf.contrib.layers.fully_connected(inputs=self.inputs1,
                                                   num_outputs=30,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer())

        # Since there are only two actions, number of outputs must be 2

        self.out_layer = tf.contrib.layers.fully_connected(inputs=self.hidden,
                                                       num_outputs=10,
                                                       activation_fn=tf.nn.relu,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer())
        '''
        Dueling
        Divide the final output into value part and action part
        Set number of outputs in final layer as 3, first for value , the other for advantage
        '''
        self.AW = tf.Variable(tf.random_normal([5,1]))
        self.VW = tf.Variable(tf.random_normal([5,2]))

        self.V_holder,self.A_holder  = tf.split(self.out_layer, num_or_size_splits=2, axis=1)

        self.Advantage = tf.matmul(self.A_holder, self.AW)
        self.Value = tf.matmul(self.V_holder, self.VW)

        # Subtract the mean of advantage from each advantage
        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_sapce = tf.constant(2)

        self.actions_onehot = tf.one_hot(self.actions, self.actions_sapce, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.updateModel = self.trainer.minimize(self.loss)


'''
Experience Reply (No priority)
'''
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])



def updateTargetGraph(tfVars,target_rate):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*target_rate) + ((1-target_rate)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def processState(states):
    return np.reshape(states,[21168])

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values


num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.

h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.

target_rate = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork()
targetQN = Qnetwork()

init = tf.global_variables_initializer()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, target_rate)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = 0.05

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.

with tf.Session() as sess:
    sess.run(init)
    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        s = env.reset()
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 1)
            else:

                a = sess.run(mainQN.predict, feed_dict={mainQN.inputs1: s.reshape(1,4)})[0]

            s1, r, d,_ = env.step(a)

            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

            if total_steps > pre_train_steps:

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values

                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.inputs1: np.vstack(trainBatch[:, 3])})

                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.inputs1: np.vstack(trainBatch[:, 3])})

                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]

                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.inputs1: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:,1]})

                    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
            rAll += r
            s = s1

            if d == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)

        jList.append(j)
        rList.append(rAll)
        print("Number of total reward : ", rAll)
        jList.append(j)
        rList.append(rAll)
        print(sum(rList) / (i + 1))

        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)

print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")