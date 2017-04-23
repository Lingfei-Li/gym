import numpy as np
import tensorflow as tf
import gym
import random
import warnings

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [None, 80, 80], "state")
        self.batch_size = tf.shape(self.state)[0]
        self.state_fc = tf.reshape(self.state,[self.batch_size,80*80*1])
        self.state_cnn = tf.reshape(self.state, [self.batch_size, 80,80,1])

        self.action = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.actor_lrate = lr

        with tf.variable_scope('Actor'):
            #
            # '''
            # CNN for actor
            # C_P_C_P_C_P_F_Out
            # '''
            # self.conv1 = tf.layers.conv2d(inputs=self.state_cnn, filters=10, kernel_size=3, strides=(1, 1),
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
            # self.falt_d = tf.reshape(self.pool3,[-1, 10*10*30])
            #
            # self.hidden1 = tf.layers.dense(
            #     inputs=self.falt_d,
            #     units=20,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='hidden1'
            # )
            # self.acts_prob = tf.layers.dense(
            #     inputs=self.hidden1,
            #     units=n_actions,  # output units
            #     activation=tf.nn.sigmoid,  # get action probabilities
            #     kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
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
                name='hidden1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=self.hidden1,
                units=n_actions,  # output units
                activation=tf.nn.sigmoid,  # get action probabilities
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):

            self.log_prob = tf.reshape(tf.log(self.acts_prob),[self.batch_size,n_actions])
            self.act_one_hot = tf.one_hot(self.action, n_actions)
            self.log_prob2 = tf.reduce_sum(self.log_prob*self.act_one_hot,1)
            self.exp_v = tf.reduce_mean(self.log_prob2 * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.actor_lrate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, state, action, td):
        action2 = np.reshape(action,[-1])
        td2 = np.reshape(td,[-1])
        feed_dict = {self.state: state, self.action: action2, self.td_error: td2}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, state):
        probs = self.sess.run(self.acts_prob, {self.state: state})  # get probabilities for all actions
        print probs
        choice = np.random.choice(np.arange(6), p= probs.ravel()/np.sum( probs.ravel()))

        return choice# return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [None, 80, 80], "state")
        self.batch_size = tf.shape(self.state)[0]
        self.state_fc = tf.reshape(self.state, [self.batch_size, 80 * 80 * 1])
        self.v_next = tf.placeholder(tf.float32, None, "v_next")
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
        v_next = np.reshape(v_next,[5,1])
        reward2 = np.reshape(reward,[5,1])
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.state: state, self. v_next:  v_next, self.reward: reward2})

        return td_error

def resize(I):
    """ prepro 210x160x3 state into 1x80x80x3 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[np.newaxis,:]
    return I.astype(np.float)

def sample(batch,size):

    return np.random.randint(0,size-1,batch)



if __name__ == "__main__":

    # warnings.simplefilter("error", "RuntimeWarning")

    np.random.seed(2)
    tf.set_random_seed(2)  # reproducible
    MAX_EPISODE = 3000
    MAX_EP_STEPS = 2000  # maximum time step in one episode
    discount = 0.9  # reward discount in TD error
    batch_size = 5
    memory_size = 1000

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

    m_state = []
    m_state_next = []
    m_action = []
    m_reward = []
    m_advantage = []
    episode_sum = []
    render = False
    for i_episode in range(MAX_EPISODE):
        state = env.reset()
        state = resize(state)
        t = 0
        track_r = []
        while True:

            action = actor.choose_action(state)
            state_, r, done, info = env.step(action)
            state_ = resize(state_)
            track_r.append(r)


            m_state.append(state[0])
            m_action.append(action)
            m_state_next.append(state_[0])
            m_reward.append(r)

            state = state_
            t += 1

            if len(m_state) > 1000:
                del m_state[:500]
                del m_action[:500]
                del m_state_next[:500]
                del m_reward[:500]

            if len(m_state) % batch_size == 0:

                sample_state ,sample_next_state ,sample_r, sample_action = \
                    zip(*random.sample(list(zip(m_state, m_state_next,m_reward,m_action)), batch_size))

                td_error = critic.learn(sample_state, sample_r, sample_next_state)
                actor.learn(sample_state, sample_action, td_error)

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                episode_sum.append(ep_rs_sum)
                render = ep_rs_sum == episode_sum[len(episode_sum)-2]
                print("episode:",i_episode, "  reward:", sum(track_r), "steps:" , t,"mean reward",
                      0.95* sum(episode_sum)/len(episode_sum) + 0.05 * sum(track_r))
                break
