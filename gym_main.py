import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 
import time

# env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    print("Action: ", action)
    print("Observation: ", observation)
    print("Rewards: ", reward)

    if terminated or truncated:
        observation, info = env.reset()
    
    # time.sleep(0.1)
    
env.close()
