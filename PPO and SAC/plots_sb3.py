import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

def plot_rewards(log_dir1, log_dir2, window=50):
    def load_rewards(log_dir, window):
        monitor_file = None
        # Find the monitor CSV (can be inside a subdirectory like "CustomHopper-source-v0.monitor.csv")
        for root, _, files in os.walk(log_dir):
            for f in files:
                if f.endswith("monitor.csv"):
                    monitor_file = os.path.join(root, f)
                    break

        if monitor_file is None:
            raise FileNotFoundError("Monitor CSV file not found in log directory.")

        # Skip first two rows: header metadata from Monitor wrapper
        data = pd.read_csv(monitor_file, skiprows=1)
        timesteps = data['l'].cumsum()
        rewards = data['r']
        smoothed_rewards = rewards.rolling(window, min_periods=1).mean()
        return timesteps, smoothed_rewards


    timesteps_1, rewards_1 = load_rewards(log_dir1, window)
    timesteps_2, rewards_2 = load_rewards(log_dir2, window)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps_1, rewards_1, label='PPO')
    plt.plot(timesteps_2, rewards_2, label='SAC')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('PPO vs SAC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/PPO_SAC.png")
    plt.close()

def plot_comparison(source, target, udr, ndr):
    source_env = Monitor(gym.make('CustomHopper-source-v0'))
    source_env = DummyVecEnv([lambda: source_env])

    target_env = Monitor(gym.make('CustomHopper-target-v0'))
    target_env = DummyVecEnv([lambda: target_env])

    all_results = []

    #Source->Target
    rewards_st, _ = evaluate_policy(source, target_env, n_eval_episodes=50, deterministic=True, return_episode_rewards=True)
    for r in rewards_st:
        all_results.append({
            "Model": 'Source->Target',
            "Reward": r
        })


    rewards_udr, _ = evaluate_policy(udr, target_env, n_eval_episodes=50, deterministic=True, return_episode_rewards=True)
    for r in rewards_udr:
        all_results.append({
            "Model": 'Reptile->Target',
            "Reward": r
        })

    rewards_ndr, _ = evaluate_policy(ndr, target_env, n_eval_episodes=50, deterministic=True, return_episode_rewards=True)
    for r in rewards_ndr:
        all_results.append({
            "Model": 'ADR->Target',
            "Reward": r
        })

    #Target->Target
    rewards_tt, _ = evaluate_policy(target, target_env, n_eval_episodes=50, deterministic=True, return_episode_rewards=True)
    for r in rewards_tt:
        all_results.append({
            "Model": 'Target->Target',
            "Reward": r
        })

    df = pd.DataFrame(all_results)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Reward", data=df)
    sns.stripplot(x="Model", y="Reward", data=df, color='black', alpha=0.3, jitter=True)
    plt.title(f"Reptile vs ADR (50 Episodes)")
    plt.ylabel("Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/Reptile-ADR.png")
    plt.close()

if __name__ == "__main__":
    log_dir1 = "./ppo_source_hopper/"
    log_dir2 = "./sac_source_hopper/"
    source_PPO = PPO.load("./ppo_source_hopper/best_model.zip")
    target_PPO = PPO.load("./ppo_target_hopper/best_model.zip")
    source_SAC = SAC.load("./sac_source_hopper/best_model.zip")
    target_SAC = SAC.load("./sac_target_hopper/best_model.zip")
    udr_PPO = PPO.load("./models/UDR_PPO_Source_model.zip")
    ndr_PPO = PPO.load("./ndr_ppo_source_hopper/best_model.zip")
    udr_SAC = SAC.load("./udr_sac_custom_hopper/best_model.zip")
    ndr_SAC = SAC.load("./ndr_sac_source_hopper/best_model.zip")
    reptile = PPO.load("./ppo_reptile_final.zip")
    adr = PPO.load("./models/ADR_PPO_model.zip")
    #plot_rewards(log_dir1, log_dir2)
    plot_comparison(source_SAC, target_SAC, reptile, adr)