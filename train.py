"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--actor-critic', action='store_true', help='Use Actor-Critic instead of REINFORCE')

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
    agent = Agent(policy, device=args.device, use_actor_critic=args.actor_critic)

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:
            if args.actor_critic:
                action, log_prob, value = agent.get_action(state)
            else:
                action, log_prob = agent.get_action(state)
                value = None  # Unused in REINFORCE

            next_state, reward, done, info = env.step(action.detach().cpu().numpy())

            if agent.actor_critic and not done:
                _, _, next_value = agent.get_action(next_state, evaluation=True)
                agent.store_outcome(state, next_state, log_prob, reward, done, value, next_value)
            else:
                agent.store_outcome(state, next_state, log_prob, reward, done)

            state = next_state
            train_reward += reward

        agent.update_policy()

        if (episode + 1) % args.print_every == 0:
            print(f'Training episode: {episode + 1}')
            print(f'Episode return: {train_reward:.2f}')

    torch.save(agent.policy.state_dict(), "model.mdl")


if __name__ == '__main__':
	main()