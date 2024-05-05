import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 

class Sarsa():
    def __init__(self, env):
        self.env = env 
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n 
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Hyperparameters 
        self.epsilon = 0.95
        self.epsilon_decay = 0.999  # Encourage exploration in beginning and exploitation towards the end 
        self.epsilon_min = 0.01
        self.discount_factor = 0.95
        self.learning_rate = 0.1
        self.total_episodes = 20000

        self.rewards = []
        self.test_episodes_range = 100 
        self.test_episodes = []
        self.test_frequency = 1000

    def run(self):
        print(f"Running SARSA algorithm...")

        reward = 0
        for episode in range(self.total_episodes):
            state_current = self.env.reset()[0]
            action_current = self._choose_action(state_current)
            truncated = False   # Keeps track if episode goes over timelimit 
            terminated = False

            while not(terminated or truncated):
                state_next, reward, terminated, truncated, info = self.env.step(action_current)
                action_next = self._choose_action(state_next)
                self._update_Q_value(state_current, action_current, state_next, action_next, reward)

                state_current = state_next
                action_current = action_next

            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

            if episode % self.test_frequency == 0:
                print(f"Evaluating episode {episode}")
                total_rewards = 0

                for _ in range(self.test_episodes_range):
                    state = self.env.reset()[0]
                    terminated, truncated = False, False
                    while not(terminated or truncated):
                        action = np.argmax(self.Q[state])
                        state, reward, terminated, truncated, _ = self.env.step(action)
                        total_rewards += reward
                
                self.rewards.append(total_rewards / self.test_episodes_range)
                self.test_episodes.append(episode)

        self._plot_results()

    def _choose_action(self, state):
        '''
        Chooses an action using epsilon-greedy policy 
        '''
        action = 0

        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])

        return action


    def _update_Q_value(self, state, action, state_next, action_next, reward):
        '''
        Updates the Q-value function 
        '''
        Q_current = self.Q[state, action]
        target = reward + self.discount_factor*self.Q[state_next, action_next]
        self.Q[state, action] = Q_current + self.learning_rate*(target - Q_current)


    def _plot_results(self):
        fig, ax = plt.subplots()
        ax.plot(self.test_episodes, self.rewards)
        ax.set_title('Episodes vs average rewards')
        ax.set_xlabel('Episode')
        _ = ax.set_ylabel('Average reward')

        plt.show()