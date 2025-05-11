"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import gym
import pickle
from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=20000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=500, type=int, help='Print info every <> episodes')
	parser.add_argument('--update-every', default=10, type=int, help='Update policy every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--actor-critic', action='store_true', help='ActoriCritic if true else REINFORCE')
	parser.add_argument('--baseline', default = 0, type=int, help='S net fixed baseline')

	return parser.parse_args()


def main():

	args = parse_args()
	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	print(args.actor_critic)
	print(args.baseline)
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device, baseline = args.baseline, actor_critic=args.actor_critic)
	episodes_returns = []


	for episode in range(args.n_episodes):
		done = False
		train_reward = 0 #New reward
		state = env.reset()  # Reset the environment to S_0 and observe the initial state

		while not done:  # Loop until the episode is over
			if args.actor_critic:
				action, action_probabilities, value = agent.get_action(state)
			else:
				action, action_probabilities, _ = agent.get_action(state)
				value = None
			
			new_state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.actor_critic:
				if not done:
					_, _, next_value = agent.get_action(new_state, evaluation=True)
				else:
					next_value = torch.tensor(0.0, device=args.device).unsqueeze(0)
				agent.store_outcome(state, new_state, action_probabilities, reward, done, value, next_value)
			else:
				agent.store_outcome(state, new_state, action_probabilities, reward, done)			
			
			state = new_state
			train_reward += reward
			

		episodes_returns.append(train_reward)
		agent.update_policy()
		
		if (episode + 1) % args.print_every == 0:
			print('Training episode:', episode + 1)
			print('Episode return:', train_reward)
		if (episode+1) % 100 == 0:
			print("fatto")
		
		"""	
		Intermediate model saving
			checkpoint_filename = f"modelnobs_checkpoint_ep{episode + 1}.mdl"
			torch.save(agent.policy.state_dict(), checkpoint_filename)
			print(f"Modello salvato in {checkpoint_filename}")
		"""

	with open("training_statsBaseline.pkl", "wb") as f:
		pickle.dump({"returns": episodes_returns,}, f)

	torch.save(agent.policy.state_dict(), "modelBaseline.mdl")
	env.close()
	

if __name__ == '__main__':
	main()