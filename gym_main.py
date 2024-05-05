import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 
import time

from sarsa import Sarsa

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

sarsa = Sarsa(env=env)
sarsa.run()
sarsa.plot_results(filename='SARSA_test-0.png')

env.close()
