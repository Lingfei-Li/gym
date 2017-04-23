import numpy as np
import gym
import tensorflow as tf



class Actor:
    def __init__(self, env, ):
        self.env = env
        self.observation_space = I
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        # Learning parameters
        self.learning_rate = 0.0001
        # Declare tf graph
        self.graph = tf.Graph()

        # Build the graph when instantiated
        with self.graph.as_default():
            #     self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
            #     self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

            self.x = tf.placeholder("float", [None, self.observation_space])  # State input

            self.x2 = tf.reshape(self.x, [-1, 80, 80, 1])

            '''
            CNN for actor
            C_P_C_P_C_P_F_Out
            '''
            self.conv1 = tf.layers.conv2d(inputs=self.x2, filters=10, kernel_size=3, strides=(1, 1),
                                          padding="same", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=2, strides=2)

            self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=20, kernel_size=3, strides=(1, 1),
                                          padding="same", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=2, strides=2)

            self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=30, kernel_size=3, strides=(1, 1),
                                          padding="same", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=2, strides=2, padding="same")

            self.falt_d = tf.reshape(self.pool3, [-1, 10 * 10 * 30])

            self.hidden1 = tf.layers.dense(
                inputs=self.falt_d,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='hidden1'
            )
            self.policy = tf.layers.dense(
                inputs=self.hidden1,
                units=self.action_space_n,  # output units
                activation=tf.nn.softmax,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='policy'
            )

            self.y = tf.placeholder("float")  # Advantage input
            self.action_input = tf.placeholder("float", [None,
                                                         self.action_space_n])  # Input action to return the probability associated with that action
            self.log_action_probability = tf.reduce_sum(self.action_input * tf.log(self.policy))
            self.loss = -self.log_action_probability * self.y  # Loss is score function times advantage
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # Initializing all variables
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            print("Policy Graph Constructed")

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)



    def save(self):
        self.saver.save(self.sess, './actor_model')

    def restore(self):
        self.saver.restore(self.sess, './actor_model')

    def rollout_policy(self, timeSteps, episodeNumber):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        total_reward = 0
        curr_state = self.env.reset()
        prev_x = None
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_return_from_states = []
        # hyperparameters
        H = 200  # number of hidden layer neurons
        batch_size = 10  # every how many episodes to do a param update?
        learning_rate = 1e-4
        gamma = 0.99  # discount factor for reward
        I = 80 * 80
        H = 200

        def prepro(S):
            """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
            S = S[35:195]  # crop
            S = S[::2, ::2, 0]  # downsample by factor of 2
            S[S == 144] = 0  # erase background (background type 1)
            S[S == 109] = 0  # erase background (background type 2)
            S[S != 0] = 1  # everything else (paddles, ball) just set to 1
            return S.astype(np.float).ravel()

        # Replay memory consists of multiple lists of state, action, next state, reward, return from state
        replay_states = []
        replay_actions = []
        replay_rewards = []
        replay_next_states = []
        replay_return_from_states = []

        for time in range(timeSteps):
            # print(time)
            cur_x = prepro(curr_state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(I)
            prev_x = cur_x

            action = self.choose_action(cur_x)
            next_state, reward, done, info = self.env.step(action)
            next_x = prepro(next_state)
            # Update the total reward
            total_reward += reward
            if done or time >= self.env.spec.timestep_limit:
                break
            # Updating the memory
            curr_state_l = cur_x.tolist()
            next_state_l = next_x.tolist()
            #          if curr_state_l not in episode_states:
            episode_states.append(curr_state_l)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_next_states.append(next_state_l)
            episode_return_from_states.append(reward)
            #     for i in range(len(episode_return_from_states) - 1):
            #         episode_return_from_states[i] += reward * tf.pow(gamma, len(episode_return_from_states) - 1 - i)
            # else:
            #     # Iterate through the replay memory  and update the final return for all states
            #     for i in range(len(episode_return_from_states)):
            #         episode_return_from_states[i] += reward * tf.pow(gamma, len(episode_return_from_states) - i)
            curr_state = next_state

            #       episode_np = np.array(episode_return_from_states)
            #       episode_np -= np.mean(episode_np)
            #       episode_np /= np.std(episode_np)
            #       episode_return_from_states = episode_np.tolist()
        running_add = 0
        for t in reversed(range(0, len(episode_return_from_states))):
            #           if episode_return_from_states[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + episode_return_from_states[t]
            episode_return_from_states[t] = running_add

        self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states,
                           episode_return_from_states)
        return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward

    def update_policy(self, advantage_vectors):
        # Update the weights by running gradient descent on graph with loss function defined

        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        for i in range(len(replay_states)):

            states = replay_states[i]
            actions = replay_actions[i]
            advantage_vector = advantage_vectors[i]
            for j in range(len(states) - 1, len(states)):
                action = self.to_action_input(actions[j])

                state = np.asarray(states[j])
                state = state.reshape(1, I)

                _, error_value = self.sess.run([self.optim, self.loss],
                                               feed_dict={self.x: state, self.action_input: action,
                                                          self.y: advantage_vector[j]})

    def softmax_policy(self, state, weights):
        hidden = tf.nn.relu(tf.matmul(state, weights['W1']))
        policy = tf.nn.softmax(tf.matmul(hidden, weights['out']))
        return policy

    def choose_action(self, state):
        # Use softmax policy to sample
        state = np.asarray(state)
        state = state.reshape(1, I)
        softmax_out = self.sess.run(self.policy, feed_dict={self.x: state})
        action = np.random.choice([0, 1, 2, 3, 4, 5], 1, replace=True, p=softmax_out[0])[0]
        return action

    def update_memory(self, episode_states, episode_actions, episode_rewards, episode_next_states,
                      episode_return_from_states):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Using first visit Monte Carlo so total return from a state is calculated from first time it is visited

        replay_states.append(episode_states)
        replay_actions.append(episode_actions)
        replay_rewards.append(episode_rewards)
        replay_next_states.append(episode_next_states)
        replay_return_from_states.append(episode_return_from_states)

    def reset_memory(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]

    def to_action_input(self, action):
        action_input = [0] * self.action_space_n
        action_input[action] = 1
        action_input = np.asarray(action_input)
        action_input = action_input.reshape(1, self.action_space_n)
        return action_input


class Critic:
    def __init__(self, env):
        self.env = env
        #  self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        self.n_input = I
        self.n_hidden_1 = 200
        # Learning Parameters
        self.learning_rate = 0.0001
        # self.learning_rate = 0.1
        self.num_epochs = 10
        self.batch_size = 200
        # Discount factor
        self.discount = 0.99
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.state_input = self.x = tf.placeholder("float", [None, self.n_input])  # State input
            self.state_input2 = tf.reshape(self.state_input,[-1,80,80,1])

            '''
            CNN for actor
            C_P_C_P_C_P_F_Out
            '''
            self.conv1 = tf.layers.conv2d(inputs=self.state_input2, filters=10, kernel_size=3, strides=(1, 1),
                                          padding="same", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=2, strides=2)

            self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=20, kernel_size=3, strides=(1, 1),
                                          padding="same", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=2, strides=2)

            self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=30, kernel_size=3, strides=(1, 1),
                                          padding="same", activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=2, strides=2, padding="same")

            self.falt_d = tf.reshape(self.pool3,[-1, 10*10*30])

            self.hidden1 = tf.layers.dense(
                inputs=self.falt_d,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='hidden1'
            )
            self.value_pred = tf.layers.dense(
                inputs = self.hidden1,
                units = 1,  # output units
                kernel_initializer=tf.contrib.layers.xavier_initializer(),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
            self.return_input = tf.placeholder("float")  # Target return
            self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.return_input, 2))
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        print("Value Graph Constructed")
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)



    def save(self):
        self.saver.save(self.sess, './critic_model')

    def restore(self):
        self.saver.restore(self.sess, './critic_model')


    def update_value_estimate(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Monte Carlo prediction
        batch_size = self.batch_size
        if np.ma.size(replay_states) < batch_size:
            batch_size = np.ma.size(replay_states)

        for epoch in range(self.num_epochs):
            total_batch = min(divmod(np.ma.size(replay_states), batch_size)[0], 10)
            # Loop over all batches
            for i in range(total_batch):
                b_size = min(batch_size, 100)
                batch_state_input, batch_return_input = self.get_next_batch(batch_size, replay_states,
                                                                            replay_return_from_states)
                # Fit training data using batch
                self.sess.run(self.optim,
                              feed_dict={self.state_input: batch_state_input, self.return_input: batch_return_input})

    def get_advantage_vector(self, states, rewards, next_states):
        # Return TD(0) Advantage for particular state and action
        # Get value of current state
        advantage_vector = []
        for i in range(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1, I)
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1, I)
            reward = rewards[i]
            state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: state})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: next_state})
            # Current implementation uses TD(0) advantage
            advantage = reward + self.discount * next_state_value - state_value
            advantage_vector.append(advantage)

        ad_np = np.array(advantage_vector)
        ad_np -= np.mean(ad_np)
        ad_np /= np.std(ad_np)
        result_vector = ad_np.tolist()

        return result_vector

    def get_next_batch(self, batch_size, states_data, returns_data):
        # Return mini-batch of transitions from replay data
        all_states = []
        all_returns = []
        for i in range(len(states_data)):
            episode_states = states_data[i]
            episode_returns = returns_data[i]
            for j in range(len(episode_states)):
                all_states.append(episode_states[j])
                all_returns.append(episode_returns[j])
        all_states = np.asarray(all_states)
        all_returns = np.asarray(all_returns)
        randidx = np.random.randint(all_states.shape[0], size=batch_size)
        batch_states = all_states[randidx, :]
        batch_returns = all_returns[randidx]
        return batch_states, batch_returns


class ActorCriticLearner:
    def __init__(self, env, max_episodes, episodes_before_update):
        self.env = env
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)

        # Learner parameters
        self.max_episodes = max_episodes
        self.episodes_before_update = episodes_before_update
        self.save_rate = 1

    def learn(self):

        advantage_vectors = []
        sum_reward = 0
        tr = 0
        update = True
        for i in range(self.max_episodes):
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(
                10000, i + 1)
            # print("finish rolling ", i)
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            tr += episode_total_reward
            print("episode ", i, " reward is : ", episode_total_reward)
            print("average  : ", tr / (i + 1))
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                # print("Current {} episode average reward: {}, episode sum reward {}".format(self.episodes_before_update, avg_reward,sum_reward))
                # In this part of the code I try to reduce the effects of randomness leading to oscillations in my
                # network by sticking to a solution if it is close to final solution.
                # If the average reward for past batch of episodes exceeds that for solving the environment, continue with it
                #              if avg_reward >= 195:  # This is the criteria for having solved the environment by Open-AI Gym
                #                  update = False
                #              else:
                #                  update = True

                update = True

                if update:
                    print("Updating")
                    self.actor.update_policy(advantage_vectors)
                    self.critic.update_value_estimate()
                else:
                    print("Good Solution, not updating")
                # Delete the data collected so far
                del advantage_vectors[:]
                self.actor.reset_memory()
                sum_reward = 0

            if (i + 1) % self.save_rate == 0:
                self.actor.save()
                self.critic.save()



if __name__ == "__main__":
    # hyperparameters
    H = 200  # number of hidden layer neurons
    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 1e-4
    save_rate = 10
    gamma = 0.99  # discount factor for reward
    I = 80 * 80
    H = 200


    def prepro(S):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        S = S[35:195]  # crop
        S = S[::2, ::2, 0]  # downsample by factor of 2
        S[S == 144] = 0  # erase background (background type 1)
        S[S == 109] = 0  # erase background (background type 2)
        S[S != 0] = 1  # everything else (paddles, ball) just set to 1
        return S.astype(np.float).ravel()


    # Replay memory consists of multiple lists of state, action, next state, reward, return from state
    replay_states = []
    replay_actions = []
    replay_rewards = []
    replay_next_states = []
    replay_return_from_states = []


    if __name__ == "__main__":
        env = gym.make('SpaceInvaders-v0')
        env.seed(1234)
        np.random.seed(1234)
        # env.monitor.start('./cartpole-pg-experiment-15')
        # Learning Parameters
        max_episodes = 20000
        episodes_before_update = 2

        ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update)
        ac_learner.learn()
        env.monitor.close()