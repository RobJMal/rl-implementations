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
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from moviepy.editor import ImageSequenceClip
import seaborn as sns

class DQN():
    def __init__(self, env, device):
        self.device = device
        self.env = env 
        self.num_actions = self.env.action_space.n
        self.num_states = env.observation_space.shape[0]

        # Hyperparameters 
        # Modified and obtained from: https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67 
        self.epsilon = 1
        self.epsilon_decay = 0.999  # Encourage exploration in beginning and exploitation towards the end 
        self.epsilon_min = 0.01
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.total_episodes = 1000
        self.batch_size = 32            # Size of experiences we sample to train DNN
        self.replay_memory_max_capacity = 5000

        self.tau = 0.005   # Update rate of the target model 
        self.replay_memory = ReplayMemory(max_capacity=self.replay_memory_max_capacity,
                                          device=self.device)
        self.policy_model = DQNModel(num_states=self.num_states, 
                                     num_actions=self.num_actions).to(self.device)
        self.target_model = DQNModel(num_states=self.num_states,
                                     num_actions=self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())   # Synchronize the weights of the networks
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)

        # For evaluation and plotting 
        self.rewards = []
        self.test_episodes_range = 50 
        self.test_episodes = []
        self.test_frequency = 100


    def run(self) -> None:
        '''
        Runs the learning of the agent. 
        '''
        for episode in range(self.total_episodes):
            state_current = self.env.reset()[0]
            terminated, truncated = False, False

            while not(terminated or truncated):
                action_current = self._choose_action(state_current)
                state_next, reward, terminated, truncated, _ = self.env.step(action_current.item())
                reward = torch.tensor([reward], device=self.device)

                if terminated:
                    state_next = None
                else:
                    state_next = torch.tensor(state_next, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                self.replay_memory.push(state_current, action_current, state_next, reward)

                state_current = state_next
                self._optimize_model()
                self._soft_update_target_network_weights()

            # Reduce the epsilon after every episode
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

            print(f"Finished episode {episode}")

            # if episode % self.test_frequency == 0:
            #         self._evaluate_policy(episode)

    def _choose_action(self, state):
        '''
        Chooses an action using epsilon-greedy policy 
        '''
        if np.random.uniform(0, 1) < self.epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_model(state).max(1).indices.view(1, 1)


    def _optimize_model(self) -> None:
        '''
        Performs optimization on the model 
        '''
        if len(self.replay_memory) < self.batch_size:
            return 
        
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Filtering out the termination state so model doesn't learn future reward
        # calculations and for stability
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                                batch.state_next)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.state_next if s is not None])
        
        state_batch = torch.cat([s for s in batch.state if s is not None])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Ensure tensors are on the correct device
        state_batch, action_batch, reward_batch = state_batch.to(self.device), action_batch.to(self.device), reward_batch.to(self.device)

        Q_values = self.policy_model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1).values

        Q_values_expected = (next_state_values * self.discount_factor) + reward_batch

        loss_function = nn.SmoothL1Loss()   # Huber loss 
        loss = loss_function(Q_values, Q_values_expected.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

    
    def _soft_update_target_network_weights(self) -> None:
        '''
        Performs a soft update of the target network's weights
        '''
        target_model_state_dict = self.target_model.state_dict()
        policy_model_state_dict = self.policy_model.state_dict()

        for key in policy_model_state_dict:
            target_model_state_dict[key] = policy_model_state_dict[key]*self.tau + target_model_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_model_state_dict)


# Represents transition in environment 
Transition = namedtuple('Transition', ('state', 'action', 'state_next', 'reward'))

class ReplayMemory():
    def __init__(self, max_capacity: int, device) -> None:
        self.memory_buffer = deque([], maxlen=max_capacity)
        self.device = device

    def push(self, state, action, state_next, reward) -> None:
        '''
        Adds a new transition into the memory buffer 
        '''
        # print("State before conversion: ", state)
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0) if not isinstance(state, torch.Tensor) else state.to(self.device)
        action = torch.tensor([action], device=self.device, dtype=torch.long) if not isinstance(action, torch.Tensor) else action.to(self.device)
        if state_next is not None:
            state_next = torch.tensor(state_next, device=self.device, dtype=torch.float32).unsqueeze(0) if not isinstance(state_next, torch.Tensor) else state_next.to(self.device)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float) if not isinstance(reward, torch.Tensor) else reward.to(self.device)
        self.memory_buffer.append(Transition(state, action, state_next, reward))
    
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
    

class DQNModel(nn.Module):
    '''
    Constructs model used for DQN agent for Q-function approximation. 
    '''
    def __init__(self, num_states, num_actions) -> None:
        super(DQNModel, self).__init__()
        self.fc_layer1 = nn.Linear(num_states, 128)
        self.fc_layer2 = nn.Linear(128, 128)
        self.fc_layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc_layer1(x))
        x = F.relu(self.fc_layer2(x))
        return self.fc_layer3(x)