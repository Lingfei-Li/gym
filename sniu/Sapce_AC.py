import numpy as np
import tensorflow as tf
import gym


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [1, 80, 80, 3], "state")
        self.state_fc = tf.reshape(self.state,[1,-1])
        self.action = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.actor_lrate = lr

        with tf.variable_scope('Actor'):
            # '''
            # CNN for actor
            # C_P_C_P_C_P_F_Out
            # '''
            # self.conv1 = tf.layers.conv2d(inputs=self.state, filters=10, kernel_size=3, strides=(1, 1),
            #                               padding="same", activation=tf.nn.relu,
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            # self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=2, strides=2)
            #
            # self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=20, kernel_size=3, strides=(1, 1),
            #                               padding="same", activation=tf.nn.relu,
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #
            # self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=2, strides=2)
            #
            # self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=30, kernel_size=3, strides=(1, 1),
            #                               padding="same", activation=tf.nn.relu,
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #
            # self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=2, strides=2, padding="same")
            #
            # self.falt_d = tf.reshape(self.pool3,[-1])
            #
            # self.acts_prob = tf.layers.dense(
            #     inputs=self.hidden1,
            #     units=n_actions,  # output units
            #     activation=tf.nn.softmax,  # get action probabilities
            #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='acts_prob'
            # )

            '''
            Fully connected network for actor
            '''

            self.hidden1 = tf.layers.dense(
                inputs=self.state_fc,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=self.hidden1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.actor_lrate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, state, action, td):
        feed_dict = {self.state: state, self.action: action, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, state):
        probs = self.sess.run(self.acts_prob, {self.state: state})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [1, 80, 80, 3], "state")
        self.state_fc = tf.reshape(self.state, [1, -1])
        self. v_next = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.reward = tf.placeholder(tf.float32, None, 'reward')
        self.critic_lrate = lr

        with tf.variable_scope('Critic'):
            self.hidden1 = tf.layers.dense(
                inputs=self.state_fc,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='hidden1'
            )

            self.v = tf.layers.dense(
                inputs=self.hidden1,
                units=1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.reward + discount * self. v_next - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+discount*V_next) -  v_nexteval

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.critic_lrate).minimize(self.loss)

    def learn(self, state, reward, state_):
        v_next = self.sess.run(self.v, {self.state: state_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.state: state, self. v_next:  v_next, self.reward: reward})
        return td_error

def resize(I):
    """ prepro 210x160x3 state into 1x80x80x3 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, :]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[np.newaxis,:]
    return I.astype(np.float)


if __name__ == "__main__":
    np.random.seed(2)
    tf.set_random_seed(2)  # reproducible
    MAX_EPISODE = 3000
    MAX_EP_STEPS = 1000  # maximum time step in one episode
    discount = 0.9  # reward discount in TD error

    # Set the environment
    env = gym.make('SpaceInvaders-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    N_Feature = env.observation_space.shape[0]
    N_Action = env.action_space.n

    sess = tf.Session()
    actor = Actor(sess, n_features=N_Feature, n_actions=N_Action)
    critic = Critic(sess, n_features=N_Feature,)
    sess.run(tf.global_variables_initializer())

    for i_episode in range(MAX_EPISODE):
        state = env.reset()
        state = resize(state)
        t = 0
        track_r = []
        while True:

            action = actor.choose_action(state)
            state_, r, done, info = env.step(action)
            state_ = resize(state_)

            if done:
                r = -20
            track_r.append(r)

            td_error = critic.learn(state, r, state_)  # gradient = grad[r + discount * V(s_) - V(s)]
            actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            state = state_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                print("episode:",i_episode, "  reward:", int(running_reward))
                break
