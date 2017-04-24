import numpy as np
import _pickle as pickle
import gym
import tensorflow as tf
import random
import tensorflow.contrib.slim as slim

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 3  # every how many episodes to do a param update?
trainlength = 6
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
I = 80 * 80

def prepro(S):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    S = S[35:195]  # crop
    S = S[::2, ::2, 0]  # downsample by factor of 2
    S[S != 0] = 1  # everything else (paddles, ball) just set to 1
    return S.astype(np.float).ravel()



# Replay memory consists of multiple lists of state, action, next state, reward, return from state
replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []


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

            self.weights = {
                'W1' : tf.Variable(np.random.randn(I, H) / np.sqrt(I), dtype=tf.float32),  # "Xavier" initialization
                'out' : tf.Variable(np.random.randn(H, 6) / np.sqrt(H), dtype=tf.float32)
            }

            # Inputs
            self.x = tf.placeholder("float", [None, self.observation_space])  # State input
            self.y = tf.placeholder("float")  # Advantage input
            self.action_input = tf.placeholder("float", [None,
                                                         self.action_space_n])  # Input action to return the probability associated with that action

            self.policy = self.softmax_policy(self.x, self.weights)  # Softmax policy

            self.log_action_probability = tf.reduce_sum(self.action_input * tf.log(self.policy))
            self.loss = -self.log_action_probability * self.y  # Loss is score function times advantage
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # Initializing all variables
            self.init = tf.global_variables_initializer()

            print("Policy Graph Constructed")

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def rollout_policy(self, timeSteps, episodeNumber):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        total_reward = 0
        curr_state = self.env.reset()
        prev_x = x = None
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_return_from_states = []

        for time in range(timeSteps):
            # print(time)
            #if (time>1): episode_next_states.append(x.tolist())
            cur_x = prepro(curr_state)

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
            #if curr_state_l not in episode_states:
            episode_states.append(x.tolist())
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
        # episode_next_states.append(x.tolist())

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
            for j in range(len(states)-1, len(states)):
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
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=H, state_is_tuple=True)

        self.h_size = 200
        self.env = env
        #self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        self.n_input = I
        self.n_hidden_1 = H
        # Learning Parameters
        self.learning_rate = 0.0001
        # self.learning_rate = 0.1
        self.num_epochs = 10

        # Discount factor
        self.discount = 0.99
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = {
                'h1': tf.Variable(np.random.randn(self.n_input, self.n_hidden_1) / np.sqrt(self.n_input), dtype=tf.float32),
                'out': tf.Variable(np.random.randn(self.n_hidden_1, 1) / np.sqrt(self.n_hidden_1), dtype=tf.float32)
            }
            self.biases = {
                'h1': tf.Variable(np.random.randn(self.n_hidden_1) / np.sqrt(self.n_input), dtype=tf.float32),
                'out': tf.Variable(np.random.randn(1) / np.sqrt(self.n_hidden_1), dtype=tf.float32)
            }

            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32)

            self.state_input = tf.placeholder("float", [None, self.n_input])  # State input
            self.return_input = tf.placeholder("float")  # Target return

            #self.rnn_input = self.multilayer_perceptron(self.state_input, self.weights, self.biases)
            self.rnn_input_2 = tf.add(tf.matmul(self.state_input, self.weights['h1']), self.biases['h1'])
            #print(self.rnn_input_2)
            self.rnn_input = tf.reshape(self.rnn_input_2, [self.batch_size, self.trainLength, H])
            self.state_in = self.rnn_cell.zero_state(self.batch_size, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=self.rnn_input, cell=self.rnn_cell, dtype=tf.float32, initial_state=self.state_in)
            self.rnn = tf.reshape(self.rnn, shape=[-1, H])
            self.value_pred = self.full_connect_out(self.rnn, self.weights, self.biases)

#            self.value_pred_slow = self.multilayer_perceptron(self.state_input, self.weights_slow, self.biases_slow)
            self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.return_input, 2))
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            init = tf.global_variables_initializer()
        print("Value Graph Constructed")
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def multilayer_perceptron(self, x, weights, biases):
        # First hidden layer
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
        #layer_1 = tf.nn.relu(layer_1)
        return layer_1

    def full_connect_out(self, state, weights, biases):
        out = tf.add(tf.matmul(state, weights['out']), biases['out'])
        return out

    def update_value_estimate(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states, batch_size, trainlength
        # Monte Carlo prediction
        batch_size_1 = batch_size
        if np.ma.size(replay_states) < batch_size:
            batch_size_1 = np.ma.size(replay_states)

        for epoch in range(self.num_epochs):
            #total_batch = min(divmod(np.ma.size(replay_states) ,batch_size)[0], 10)
            # Loop over all batches
            for i in range(50):
                batch_state_input, batch_return_input = self.get_next_batch(batch_size_1, trainlength, replay_states,
                                                                            replay_return_from_states)
                # Fit training data using batch
                state_train = (np.zeros([batch_size_1, H]), np.zeros([batch_size_1, H]))
                self.sess.run(self.optim,
                              feed_dict={self.state_input: batch_state_input, self.return_input: batch_return_input,
                                         self.state_in:state_train, self.batch_size:batch_size_1, self.trainLength:trainlength})

    def get_advantage_vector(self, states, rewards, next_states):
        # Return TD(0) Advantage for particular state and action
        # Get value of current state
        advantage_vector = []
        rstate = (np.zeros([1, H]), np.zeros([1, H]))
        for i in range(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1, I)
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1, I)
            reward = rewards[i]
            state_value, rstate1 = self.sess.run([self.value_pred, self.rnn_state], feed_dict={self.state_input: state, self.trainLength:1, self.batch_size:1, self.state_in:rstate})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: next_state, self.trainLength:1, self.batch_size:1, self.state_in:rstate1})
            rstate = rstate1
            # Current implementation uses TD(0) advantage
            advantage = reward + self.discount * next_state_value - state_value
            advantage_vector.append(advantage)

        ad_np = np.array(advantage_vector)
        ad_np -= np.mean(ad_np)
        ad_np /= np.std(ad_np)
        result_vector = ad_np.tolist()

        return result_vector

    def get_next_batch(self, batch_size, trace_length, states_data, returns_data):
        sampledTraces = []
        sampledTraces_r = []
        l = len(states_data)
        for i in range(batch_size):
            index = random.randint(0, l-1)
            episode = states_data[index]
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])

            episode = returns_data[index]
            sampledTraces_r.append(episode[point:point + trace_length])


        sampledTraces = np.array(sampledTraces)
        sampledTraces_r = np.array(sampledTraces_r)
        return np.reshape(sampledTraces, [batch_size * trace_length, I]), np.reshape(sampledTraces_r, [batch_size * trace_length, 1])


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
        tr = 0
        update = True
        for i in range(self.max_episodes):
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(
                10000, i + 1)
            #print("finish rolling ", i)
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            tr += episode_total_reward
            print("episode ", i, " reward is : ", episode_total_reward )
            print("average  : ", tr/(i+1))
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                #print("Current {} episode average reward: {}, episode sum reward {}".format(self.episodes_before_update, avg_reward,sum_reward))
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
