import gym
import torch
from agent import PPO
import pickle
import os
from env.custom_hopper import *

def train():
    # === Config ===
    total_episodes = 10000
    max_steps_per_episode = 1000

    # === Set up environment and agent ===
    env = gym.make('CustomHopper-source-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "/content/drive/MyDrive/rl_logs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "reward_history.pkl")

    rewards_history = []

    # === Training loop ===
    for episode in range(total_episodes):
        episode_reward = PPO(env)
        rewards_history.append(episode_reward)
        print(f"Episode {episode+1} | Reward: {episode_reward}")  
         
        rewards_history.append(episode_reward)
        with open(save_path, 'wb') as f:
            pickle.dump(rewards_history, f)

    env.close()

if __name__ == "__main__":
    train()
