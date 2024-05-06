import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 

from moviepy.editor import ImageSequenceClip
import seaborn as sns

class QLearning():
    def __init__(self, env):
        self.env = env 
        self.map_size = 4
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n 
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Hyperparameters 
        # Modified and obtained from: https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187
        self.epsilon = 1
        self.epsilon_decay = 0.999  # Encourage exploration in beginning and exploitation towards the end 
        self.epsilon_min = 0.01
        self.discount_factor = 0.99
        self.learning_rate = 0.1
        self.total_episodes = 10000

        # For evaluation and plotting 
        self.rewards = []
        self.test_episodes_range = 50 
        self.test_episodes = []
        self.test_frequency = 100


    def run(self):
        '''
        Runs Q-Learning. 
        '''
        print(f"Running Q-Learning training...")

        reward = 0
        for episode in range(self.total_episodes):
            state_current = self.env.reset()[0]
            truncated = False   # Keeps track if episode goes over timelimit 
            terminated = False

            while not(terminated or truncated):
                action_current = self._choose_action(state_current)
                state_next, reward, terminated, truncated, _ = self.env.step(action_current)

                # Crux of Q-Learning Algorithm 
                Q_current = self.Q[state_current, action_current]
                target = reward + self.discount_factor * np.max(self.Q[state_next])
                self.Q[state_current, action_current] = Q_current + self.learning_rate*(target - Q_current)

                self._update_Q_value(state_current, action_current, state_next, reward)

                state_current = state_next

            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

            if episode % self.test_frequency == 0:
                self._evaluate_policy(episode)

        print(f"Q-Learning training COMPLETED")


    def plot_results(self, filename=None):
        '''
        Plots results of the episodes vs the average rewards. 
        '''
        fig, ax = plt.subplots()
        ax.plot(self.test_episodes, self.rewards)
        ax.set_title('Episodes vs average rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average reward')
        ax.grid(True)

        if filename:
            output_directory = 'media'
            os.makedirs(output_directory, exist_ok=True)

            file_path = os.path.join(output_directory, filename)
            plt.savefig(file_path)

        plt.show()


    def record_policy(self, output_directory="media", video_filename="q-learning_policy-0.mp4"):
        '''
        Saves the policy execution as a video
        '''
        os.makedirs(output_directory, exist_ok=True)

        frames = []
        state = self.env.reset()[0]
        terminated, truncated = False, False

        while not (terminated or truncated):
            frames.append(self.env.render())

            action = np.argmax(self.Q[state, :])
            state, _, terminated, truncated, _ = self.env.step(action)
        frames.append(self.env.render())    # Render the last frame before end of episode

        clip = ImageSequenceClip(frames, fps=10)
        video_path = os.path.join(output_directory, video_filename)
        clip.write_videofile(video_path)
        print(f"Video saved to {output_directory}/{video_filename}")


    def plot_q_values_map(self, filename="q-learning_policy-0.png"):
        '''
        Plots the policy and the last frame as a heatmap of Q-values over the grid.

        Source based off of: https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#sphx-glr-tutorials-training-agents-frozenlake-tuto-py
        '''

        def generate_qtable_with_arrows_and_values(qtable, map_size):
            action_symbols = ['←', '↓', '→', '↑']
            qtable_directions = np.empty((map_size, map_size), dtype=object)

            # Prepare Q-values and best actions
            best_actions = np.argmax(qtable, axis=1).reshape(map_size, map_size)
            max_q_values = np.max(qtable, axis=1).reshape(map_size, map_size)

            # Annotate each square with the arrow and Q-values
            for i in range(map_size):
                for j in range(map_size):
                    arrow = action_symbols[best_actions[i, j]]
                    max_q_value = max_q_values[i, j]
                    qtable_directions[i, j] = f"{arrow}\n{max_q_value:6.2f}"

            return max_q_values, qtable_directions

        # Prepare data for the heatmap
        max_q_values, qtable_directions = generate_qtable_with_arrows_and_values(self.Q, self.map_size)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            max_q_values,
            annot=qtable_directions,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title="Learned Q-values\nArrows represent best action")

        # Save the plot
        output_dir = 'media'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.show()    


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


    def _update_Q_value(self, state_current, action_current, state_next, reward):
        '''
        Updates the Q-value function based on the Q-Learning algorithm. 
        '''
        Q_current = self.Q[state_current, action_current]
        target = reward + self.discount_factor * np.max(self.Q[state_next])
        self.Q[state_current, action_current] = Q_current + self.learning_rate*(target - Q_current)


    def _evaluate_policy(self, episode):
        '''
        Evaluates the policy based on a certain frequency and averaging among certain
        amount of episodes. 
        '''
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

        print(f"Evaluating episode {episode} | Average Rewards: {total_rewards/self.test_episodes_range}")

