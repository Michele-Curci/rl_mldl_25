"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from env.custom_hopper import *
#from agent import Agent, Policy

class Policy(nn.Module):
	def __init__(self, obs_dim, action_dim):
		super(Policy, self).__init__()
		self.fc1 = nn.Linear(obs_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		self.output = nn.Linear(128, action_dim)
		self.log_std = nn.Parameter(torch.zeros(action_dim))

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		mean = self.output(x)
		std = torch.exp(self.log_std)
		return mean, std

class Agent:
    def __init__(self, policy, lr=1e-3, gamma=0.99, device='cpu'):
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

        self.log_probs = []
        self.rewards = []

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        self.log_probs.append(log_prob)
        return action, log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        self.rewards.append(reward)

    def finish_episode(self):
        R = 0
        returns = []

        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Path to file')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	agent = Agent(policy, device=args.device)

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state) #add evaluation=True

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")
	

if __name__ == '__main__':
	main()