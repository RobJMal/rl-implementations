'''
Main source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm 
'''

import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 
import random
from collections import namedtuple, deque
import torch 

from moviepy.editor import ImageSequenceClip
import seaborn as sns

class DQN():
    def __init__(self, env):
        self.env = env 
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n 

        # Hyperparameters 
        # Modified and obtained from: https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67 
        self.epsilon = 1
        self.epsilon_decay = 0.999  # Encourage exploration in beginning and exploitation towards the end 
        self.epsilon_min = 0.01
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.total_episodes = 10000
        self.batch_size = 32            # Size of experiences we sample to train DNN

        # For evaluation and plotting 
        self.rewards = []
        self.test_episodes_range = 50 
        self.test_episodes = []
        self.test_frequency = 100

# Represents transition in environment 
Transition = namedtuple('Transition', ('state', 'action', 'state_next', 'reward'))

class ReplayMemory():
    def __init__(self, max_capacity: int) -> None:
        self.memory_buffer = deque([], maxlen=max_capacity)

    def push(self, *args) -> None:
        '''
        Adds a new transition into the memory buffer 
        '''
        self.memory_buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        '''
        Returns a number of samples from buffer based on the batch_size
        '''
        return random.sample(self.memory_buffer, batch_size)
    
    def __len__(self) -> int:
        '''
        Returns the length of the memory_buffer
        '''
        return len(self.memory_buffer)