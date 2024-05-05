import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env.action_space.seed(42)

action_space = env.action_space
num_actions = action_space.n
print(f"Action space: {action_space} (Total actions: {num_actions})")

observation_space = env.observation_space
num_observations = observation_space.n
print(f"Observation space: {observation_space} (Total states: {num_observations})")

# print("Reward range", env.reward_range)
