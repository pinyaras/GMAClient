# A modified Replay buffer with a save option to save all state-action pairs inside the Buffer to a .h5 file
# The buffer needs to store data for RL purpose with observations, actions, rewards, dones and next_observations.


import numpy as np
import pandas as pd
import h5py
import torch

class ReplayBuffer:
    def __init__(self, max_size, obs_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, obs_shape))
        self.new_state_memory = np.zeros((self.mem_size, obs_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = torch.Tensor(self.state_memory[batch])
        states_ = torch.Tensor(self.new_state_memory[batch])
        actions = torch.Tensor(self.action_memory[batch])
        rewards = torch.Tensor(self.reward_memory[batch])
        dones = torch.Tensor(self.terminal_memory[batch])

        return states, actions, rewards, states_, dones

    def save_buffer(self, file_name):
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('states', data=self.state_memory[:min(self.mem_cntr, self.mem_size)])
            f.create_dataset('actions', data=self.action_memory[:min(self.mem_cntr, self.mem_size)])
            f.create_dataset('rewards', data=self.reward_memory[:min(self.mem_cntr, self.mem_size)])
            f.create_dataset('next_states', data=self.new_state_memory[:min(self.mem_cntr, self.mem_size)])
            f.create_dataset('dones', data=self.terminal_memory[:min(self.mem_cntr, self.mem_size)])

    def load_buffer(self, file_name):
        with h5py.File(file_name, 'r') as f:
            self.state_memory[:self.mem_size] = f['states']
            self.action_memory[:self.mem_size] = f['actions']
            self.reward_memory[:self.mem_size] = f['rewards']
            self.new_state_memory[:self.mem_size] = f['next_states']
            self.terminal_memory[:self.mem_size] = f['dones']
            self.mem_cntr = min(self.mem_size, len(f['states']))
