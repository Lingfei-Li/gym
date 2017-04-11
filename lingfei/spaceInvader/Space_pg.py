import numpy as np
import tensorflow as tf
import gym


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [None, 80*80], "state")
        self.action = tf.placeholder(tf.float32, [None, n_actions], "act")  #one-hot array
        self.td_error = tf.placeholder(tf.float32, [None, 1], "td_error")  # advantage
        self.actor_lrate = lr

        with tf.variable_scope('Actor'):
            self.hidden1 = tf.layers.dense(
                inputs=self.state,
                units=100,  # number of hidden units
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
            tmp = self.acts_prob * self.action
            tmp = tf.clip_by_value(tmp, 1e-10, 1.0)
            log_prob = tf.log(tmp)

            # log_prob = tf.log(self.acts_prob[0, self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.actor_lrate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

        tf.add_to_collection('vars', self.hidden1)
        tf.add_to_collection('vars', self.acts_prob)

        self.saver = tf.train.Saver()

    def learn(self, state, action, td):
        feed_dict = {self.state: state, self.action: action, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def policy_forward(self, state):
        return self.sess.run(self.acts_prob, {self.state: state})  # get probabilities for all actions

    def save_model(self):
        self.saver.save(self.sess, 'my-model')

    def restore(self):
        saver = tf.train.import_meta_graph('my-model.meta')
        saver.restore(self.sess, "my-model")


def discount_rewards(r, discount=0.9):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * discount + r[t]
        discounted_r[t] = running_add
    return discounted_r

def preproc(I):
    """ prepro 210x160x3 state into 1x80x80x3 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, :]  # downsample by factor of 2
    # I[I == 144] = 0  # erase background (background type 1)
    # I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[:, :, 0]
    # I = I[np.newaxis,:]
    return I.astype(np.float).ravel()


if __name__ == "__main__":
    np.random.seed(2)
    tf.set_random_seed(2)  # reproducible
    MAX_EPISODE = 300000
    MAX_EP_STEPS = 1000  # maximum time step in one episode
    save_frequency = 20
    discount = 0.9  # reward discount in TD error
    D = 80*80
    batch_size = 10
    running_reward = None
    f = open('output.txt', 'w')

    # Set the environment
    env = gym.make('SpaceInvaders-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    N_Feature = env.observation_space.shape[0]
    N_Action = env.action_space.n

    sess = tf.Session()
    actor = Actor(sess, n_features=N_Feature, n_actions=N_Action)
    sess.run(tf.global_variables_initializer())

    # actor.restore()

    for episode_number in range(MAX_EPISODE):
        observation = env.reset()
        t = 0
        track_r = []
        advantages = []
        observations = []
        actions = []
        rewards = []
        grads = []
        prev_x = None
        reward_sum = 0

        while True:
            cur_x = preproc(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x

            # print(x.shape)

            prob = actor.policy_forward(x.reshape((-1, D)))
            action = np.random.choice(np.arange(N_Action), p=prob.ravel())

            observation, r, done, info = env.step(action)
            if done: r = -20

            action_onehot = np.zeros((N_Action), dtype=float)
            action_onehot[action] = 1.0
            actions.append(action_onehot)
            observations.append(x)
            rewards.append(r)
            reward_sum += r

            # print(observations)
            #update network
            if done:
                discounted_epr = discount_rewards(np.vstack(rewards), discount=discount)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                if episode_number % save_frequency == 0:
                    # actor.save_model()
                    f.flush()

                # print(discounted_epr)
                # print(np.vstack(actions))

                actor.learn(state=observations, action=np.vstack(actions), td=discounted_epr)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print("episode:",episode_number, "sum reward", reward_sum, "running reward:", int(running_reward))
                f.write("episode: "+str(episode_number) + " sum reward"+ str(reward_sum)+ "running reward: " + str(int(running_reward)))
                f.write('\n')
                break
    f.close()
