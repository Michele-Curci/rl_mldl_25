"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import torch
import gym
from agent import Agent, Policy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Path to file')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--actor-critic', action='store_true', help='Use Actor-Critic instead of REINFORCE')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	agent = Agent(policy, device=args.device, use_actor_critic=args.actor_critic)

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, *_ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")

	env.close()
	

if __name__ == '__main__':
	main()