
import gym
import tensorflow as tf
import numpy as np


# Replay memory consists of multiple lists of state, action, next state, reward, return from state
replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []


class Actor:
    def __init__(self, env, ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        # Learning parameters
        self.learning_rate = 0.01
        # Declare tf graph
        self.graph = tf.Graph()

        # Build the graph when instantiated
        with self.graph.as_default():

            self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
            self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

            # Inputs
            self.x = tf.placeholder("float", [None, len(self.observation_space.high)])  # State input
            self.y = tf.placeholder("float")  # Advantage input
            self.action_input = tf.placeholder("float", [None,
                                                         self.action_space_n])  # Input action to return the probability associated with that action

            self.policy = self.softmax_policy(self.x, self.weights, self.biases)  # Softmax policy

            self.log_action_probability = tf.reduce_sum(self.action_input * tf.log(self.policy))
            self.loss = -self.log_action_probability * self.y  # Loss is score function times advantage
            # self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            # Initializing all variables
            self.init = tf.global_variables_initializer()



        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def rollout_policy(self, timeSteps, episodeNumber):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        total_reward = 0
        curr_state = self.env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_return_from_states = []

        for time in xrange(timeSteps):
            action = self.choose_action(curr_state)
            next_state, reward, done, info = self.env.step(action)
            # Update the total reward
            total_reward += reward
            if done or time >= self.env.spec.timestep_limit:
                break
            # Updating the memory
            curr_state_l = curr_state.tolist()
            next_state_l = next_state.tolist()
            if curr_state_l not in episode_states:
                episode_states.append(curr_state_l)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state_l)
                episode_return_from_states.append(reward)
                for i in xrange(len(episode_return_from_states) - 1):
                    episode_return_from_states[i] += reward
            else:
                # Iterate through the replay memory  and update the final return for all states
                for i in xrange(len(episode_return_from_states)):
                    episode_return_from_states[i] += reward
            curr_state = next_state
        self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states,
                           episode_return_from_states)
        return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward

    def update_policy(self, advantage_vectors):
        # Update the weights by running gradient descent on graph with loss function defined

        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        for i in xrange(len(replay_states)):

            states = replay_states[i]
            actions = replay_actions[i]
            advantage_vector = advantage_vectors[i]
            for j in xrange(len(states)):
                action = self.to_action_input(actions[j])

                state = np.asarray(states[j])
                state = state.reshape(1, 4)

                _, error_value = self.sess.run([self.optim, self.loss],
                                               feed_dict={self.x: state, self.action_input: action,
                                                          self.y: advantage_vector[j]})

    def softmax_policy(self, state, weights, biases):
        policy = tf.nn.softmax(tf.matmul(state, weights) + biases)
        return policy

    def choose_action(self, state):
        # Use softmax policy to sample
        state = np.asarray(state)
        state = state.reshape(1, 4)
        softmax_out = self.sess.run(self.policy, feed_dict={self.x: state})
        action = np.random.choice([0, 1], 1, replace=True, p=softmax_out[0])[0]  # Sample action from prob density
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
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        self.n_input = len(self.observation_space.high)
        self.n_hidden_1 = 20
        # Learning Parameters
        self.learning_rate = 0.005
        # self.learning_rate = 0.1
        self.num_epochs = 20
        self.batch_size = 170
        # Discount factor
        self.discount = 0.90
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
                'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'out': tf.Variable(tf.random_normal([1]))
            }
            self.state_input = self.x = tf.placeholder("float", [None, len(self.observation_space.high)])  # State input
            self.return_input = tf.placeholder("float")  # Target return
            self.value_pred = self.multilayer_perceptron(self.state_input, self.weights, self.biases)
            self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.return_input, 2))
            # self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def multilayer_perceptron(self, x, weights, biases):
        # First hidden layer
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)
        # Second hidden layer
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def update_value_estimate(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Monte Carlo prediction
        batch_size = self.batch_size
        if np.ma.size(replay_states) < batch_size:
            batch_size = np.ma.size(replay_states)

        for epoch in xrange(self.num_epochs):
            total_batch = np.ma.size(replay_states) / batch_size
            # Loop over all batches
            for i in xrange(total_batch):
                batch_state_input, batch_return_input = self.get_next_batch(batch_size, replay_states,
                                                                            replay_return_from_states)
                # Fit training data using batch
                self.sess.run(self.optim,
                              feed_dict={self.state_input: batch_state_input, self.return_input: batch_return_input})

    def get_advantage_vector(self, states, rewards, next_states):
        # Return TD(0) Advantage for particular state and action
        # Get value of current state
        advantage_vector = []
        for i in xrange(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1, 4)
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1, 4)
            reward = rewards[i]
            state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: state})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: next_state})
            # Current implementation uses TD(0) advantage
            advantage = reward + self.discount * next_state_value - state_value
            advantage_vector.append(advantage)

        return advantage_vector

    def get_next_batch(self, batch_size, states_data, returns_data):
        # Return mini-batch of transitions from replay data
        all_states = []
        all_returns = []
        for i in xrange(len(states_data)):
            episode_states = states_data[i]
            episode_returns = returns_data[i]
            for j in xrange(len(episode_states)):
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

    def learn(self):

        advantage_vectors = []
        sum_reward = 0
        update = True
        for i in xrange(self.max_episodes):
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(
                200, i + 1)
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                print "{}".format(avg_reward)
                # In this part of the code I try to reduce the effects of randomness leading to oscillations in my
                # network by sticking to a solution if it is close to final solution.
                # If the average reward for past batch of episodes exceeds that for solving the environment, continue with it
                if avg_reward >= 195:  # This is the criteria for having solved the environment by Open-AI Gym
                    update = False
                else:
                    update = True

                if update:

                    self.actor.update_policy(advantage_vectors)
                    self.critic.update_value_estimate()

                del advantage_vectors[:]
                self.actor.reset_memory()
                sum_reward = 0




if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.seed(1234)
    np.random.seed(1234)
    # env.monitor.start('./cartpole-pg-experiment-15')
    # Learning Parameters
    max_episodes = 1000
    episodes_before_update = 2

    ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update)
    ac_learner.learn()
