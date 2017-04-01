


# by the author: https://medium.com/emergent-future/simple-reinforcement-learning-with-
# tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# You are correct to observe that using a simple Q-learning algorithm on CartPole will fail.
# Due to the nature of the state space in CartPole it is very difficult
# for a basic Q algorithm to solve it. In fact, the Q-learning algorithm described
# here is almost never used for large or continuous state/action spaces.
# Instead DQN, with it’s augmentations to improve robustness is used.
# Or a policy gradient method as you mentioned.




import gym
import tensorflow as tf
import numpy as np


env = gym.make('CartPole-v0')


#setting up network
D = 4
H = 10
learning_rate = 0.01
discount = 0.99
epsilon = 0.1

input = tf.placeholder(tf.float32, [None, D])

fc1 = tf.contrib.layers.fully_connected(inputs=input,
                                        num_outputs=H,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                        num_outputs=2,
                                        activation_fn=tf.nn.sigmoid,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
predict = tf.arg_max(fc2, 1)



#mini-batch update
q_teacher = tf.placeholder(tf.float32, [None, 2])
loss = tf.reduce_mean(tf.square(q_teacher - fc2))
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update = adam.minimize(loss)


init = tf.global_variables_initializer()
episode = 0
max_episode = 10000
batch_size = 10
with tf.Session() as sess:
    sess.run(init)

    episode_rewards = []
    states = []
    rewards = []
    actions = []
    q_targets = []

    while episode < max_episode:
        state = env.reset()
        state = np.reshape(state, (-1, 4))
        episode_reward = 0

        while True:
            #calculate Q values, and choose the action with the max Q value
            action, q = sess.run([predict, fc2], feed_dict={input: state})
            action = action[0]
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()

            #step the environment with the action
            state1, reward, done, info = env.step(action)
            state1 = np.reshape(state1, (-1, 4))
            episode_reward += reward

            #get the max new Q value with the stepped environment.
            # set target q value to update the network
            q1= sess.run(fc2, feed_dict={input: state1})
            q1max= np.max(q1)
            q_target = q
            q_target[0, action] = discount*q1max + reward

            q_targets.append(q_target[0])
            states.append(state[0])

            state = state1

            if done:
                episode_rewards.append(episode_reward)
                print('episode', episode)
                print('reward', episode_reward)
                print('avg', np.mean(episode_rewards))
                episode += 1
                if episode % batch_size == 0:
                    sess.run(update, feed_dict={input:states,
                                                q_teacher: q_targets})
                    states = []
                    actions = []
                    rewards = []
                    q_targets = []

                break
















