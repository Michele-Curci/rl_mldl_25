"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import gym
from agent import Agent, Policy
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=20000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=10000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--actor-critic', action='store_true', help='Use Actor-Critic instead of REINFORCE')
    parser.add_argument('--baseline', default=0, type=int, help='Value of the baseline used in REINFORCE')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Learning rate (for wandb tracking)')
    parser.add_argument('--project', default='rl-hopper', type=str, help='wandb project name')

    return parser.parse_args()

args = parse_args()


def main():

    wandb.init(
        project=args.project,
        config={
            "n_episodes": args.n_episodes,
            "baseline": args.baseline,
            "actor_critic": args.actor_critic,
            "device": args.device,
            "learning_rate": args.learning_rate
        },
        name=f"{'ActorCritic' if args.actor_critic else f'REINFORCE_baseline{args.baseline}'}"
    )

    env = gym.make('CustomHopper-source-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device, use_actor_critic=args.actor_critic, baseline=args.baseline)

    episodes_returns = []

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:
            if args.actor_critic:
                action, log_prob, value = agent.get_action(state)
            else:
                action, log_prob, _ = agent.get_action(state)
                value = None  # Unused in REINFORCE

            next_state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.actor_critic:
                if not done:
                    _, _, next_value = agent.get_action(next_state, evaluation=True)
                else:
                    next_value = torch.tensor(0.0, device=args.device).unsqueeze(0)
                agent.store_outcome(state, next_state, log_prob, reward, done, value, next_value)
            else:
                agent.store_outcome(state, next_state, log_prob, reward, done)

            state = next_state
            train_reward += reward

        episodes_returns.append(train_reward)

        wandb.log({"episode": episode, "return": train_reward})

        agent.update_policy()

        if (episode + 1) % args.print_every == 0:
            print(f'Training episode: {episode + 1}')
            print(f'Episode return: {train_reward:.2f}')

    model_name = "ActorCritic.mdl" if args.actor_critic else f"Reinforce{'Baseline' if args.baseline != 0 else ''}.mdl"
    torch.save(agent.policy.state_dict(), model_name)
    wandb.save(model_name)
    env.close()
    wandb.finish()


if __name__ == '__main__':
	main()