import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 

from moviepy.editor import ImageSequenceClip
import seaborn as sns

class Sarsa():
    def __init__(self, env):
        self.env = env 
        self.map_size = 4
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
        '''
        Runs SARSA. 
        '''
        print(f"Running SARSA training...")

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

        print(f"SARSA training COMPLETED")


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


    def record_policy(self, output_directory="media", video_filename="sarsa_policy-0.mp4"):
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


    def plot_q_values_map(self, filename="sarsa_policy-0.png"):
        '''
        Plots the policy and the last frame as a heatmap of Q-values over the grid.
        '''

        def qtable_directions_map(qtable, map_size):
            # Define action symbols corresponding to (left, down, right, up)
            action_symbols = ['←', '↓', '→', '↑']

            qtable_val_max = np.max(qtable, axis=1).reshape(map_size, map_size)
            best_actions = np.argmax(qtable, axis=1).reshape(map_size, map_size)

            qtable_directions = np.empty(best_actions.shape, dtype=str)
            for i in range(map_size):
                for j in range(map_size):
                    qtable_directions[i, j] = action_symbols[best_actions[i, j]]

            return qtable_val_max, qtable_directions

        # Get the best Q-values and directions
        qtable_val_max, qtable_directions = qtable_directions_map(self.Q, self.map_size)

        # Plot the last rendered frame
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        ax[0].imshow(self.env.render())
        ax[0].axis("off")
        ax[0].set_title("Last frame")

        # Plot the Q-value heatmap with arrows
        sns.heatmap(
            qtable_val_max,
            annot=qtable_directions,
            fmt="",
            ax=ax[1],
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title="Learned Q-values\nArrows represent best action")

        for _, spine in ax[1].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

        # Ensure the output directory exists
        output_dir = 'media'
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.show()    