import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, lr):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.action_std = action_std
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class SACAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, action_std=0.1):
        self.actor = Actor(state_dim, action_dim, action_std, actor_lr)
        self.critic_1 = Critic(state_dim, action_dim, critic_lr)
        self.critic_2 = Critic(state_dim, action_dim, critic_lr)

    def predict(self, state):
        with torch.no_grad():
            action_mean = self.actor(state)
            action = Normal(action_mean, self.actor.action_std).sample()
            return action.numpy()

    def learn(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()

        # Update Q-functions
        next_actions = self.actor(next_states)
        next_q1 = self.critic_1(next_states, next_actions.detach())
        next_q2 = self.critic_2(next_states, next_actions.detach())
        next_q_target = torch.min(next_q1, next_q2)
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, expected_q.detach())
        q2_loss = nn.functional.mse_loss(q2, expected_q.detach())

        self.critic_1.optimizer.zero_grad()
        q1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        q2_loss.backward()
        self.critic_2.optimizer.step()

        # Update policy
        new_actions = self.actor(states)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        policy_loss = -torch.min(q1_new, q2_new).mean()

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filepath):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
