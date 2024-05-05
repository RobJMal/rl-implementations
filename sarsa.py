import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 

class Sarsa():
    def __init__(self, env):
        self.env = env 
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n 
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Parameters
        self.epsilon = 0.9
        self.discount_factor = 0.95
        self.learning_rate = 0.9
        self.total_episodes = 10000
        self.max_steps = 90    # Number of steps allowed in environment 

        self.rewards = []

    def run(self):
        reward = 0

        for episode in range(self.total_episodes):
            state_current = self.env.reset()[0]
            action_current = self._choose_action(state_current)
            truncated = False   # Keeps track if episode goes over timelimit 
            terminated = False

            while (not terminated) or (not truncated):
                state_next, reward, terminated, truncated, info = self.env.step(action_current)
                action_next = self._choose_action(state_next)

                self._update_Q_value(state_current, action_current, state_next, action_next, reward)

                state_current = state_next
                action_current = action_next

                reward += 1
            
            print(f"Completed episode {episode}")
            print(self.Q)
        
    
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