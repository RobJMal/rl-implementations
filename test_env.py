import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 

env = gym.make("CartPole-v1")
env.action_space.seed(42)

# action_space = env.action_space
# num_actions = action_space.n
# print(f"Action space: {action_space} (Total actions: {num_actions})")
# print("")

state, _ = env.reset()
num_observations = len(state)
observation_space = env.observation_space.shape[0]
print(observation_space)

# num_observations = observation_space.n
# print(f"Observation space: {observation_space} (Total states: {num_observations})")
print("")

# state_current = env.reset()
# action_current = env.action_space.sample()
# state_next, reward, _, _, _ = env.step(action_current)
# print("Next state: ", state_next)
