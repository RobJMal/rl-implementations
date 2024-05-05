import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 
import time

from sarsa import Sarsa

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode='rgb_array')
    seed_value = 42
    env.reset(seed=seed_value)
    env.action_space.seed(seed_value)

    sarsa = Sarsa(env=env)
    sarsa.run()
    sarsa.plot_results(filename='SARSA_test-0.png')
    sarsa.record_policy(output_directory="media", video_filename="sarsa_policy-0.mp4")

    env.close()
