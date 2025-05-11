"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')
    parser.add_argument('--actor-critic', action='store_true', help='ActoriCritic if true else REINFORCE')

    return parser.parse_args()

args = parse_args()

def analyze_termination(observation):
    height = observation[1]
    angle = observation[2]
    
    if not (0.7 <= height <= 9999999999):
        return "Height out of healthy range"
    elif not (-0.2 <= angle <= 0.2):
        return "Angle out of healthy range"
    else:
        return "General unhealthy state"


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

	agent = Agent(policy, device=args.device, actor_critic=args.actor_critic)

	total_reward = 0
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
			total_reward += reward
		reason = analyze_termination(state)

		print(f"Episode: {episode} | Return: {test_reward}")
		print(f"[Episode {episode + 1}] terminated: {reason}, height = {state[1]}, angle = {state[2]}")

	
	print(f"Total Return: {total_reward}")
	print(f"Average Return: {total_reward/args.episodes}")


if __name__ == '__main__':
	main()