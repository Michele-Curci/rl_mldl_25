"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
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
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		agent.finish_episode()
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model.mdl")

	

if __name__ == '__main__':
	main()