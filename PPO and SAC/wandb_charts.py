import pickle
import wandb
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Initialize Weights & Biases
wandb.init(project="ppo_vs_sac_comparison", name="PPO vs SAC Line Plot")

# Load PPO results
with open("ppo_training_stats.pkl", "rb") as f:
    ppo_data = pickle.load(f)

# Load SAC results
with open("sac_training_stats.pkl", "rb") as f:
    sac_data = pickle.load(f)

# Extract timesteps and rewards (modify keys if needed)
ppo_timesteps = ppo_data.get("timesteps") or ppo_data.get("steps")
ppo_rewards = ppo_data.get("rewards") or ppo_data.get("episode_rewards")

sac_timesteps = sac_data.get("timesteps") or sac_data.get("steps")
sac_rewards = sac_data.get("rewards") or sac_data.get("episode_rewards")

# # Limit PPO to first 20,000 episodes (make sure timesteps match)
# limit = 20000
# ppo_timesteps = ppo_timesteps[:limit]
# ppo_rewards = ppo_rewards[:limit]

# Apply moving average smoothing
window_size = 50
ppo_rewards_smooth = moving_average(ppo_rewards, window_size)
sac_rewards_smooth = moving_average(sac_rewards, window_size)

# Adjust x-axis (timesteps) accordingly after smoothing
ppo_timesteps_smooth = ppo_timesteps[window_size-1:]
sac_timesteps_smooth = sac_timesteps[window_size-1:]

# Plot smoothed rewards
plt.figure(figsize=(10, 6))
plt.plot(ppo_timesteps_smooth, ppo_rewards_smooth, label="PPO", color="blue")
plt.plot(sac_timesteps_smooth, sac_rewards_smooth, label="SAC", color="green")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("PPO vs SAC Performance")
plt.legend()
plt.grid(True)

# Save figure locally and log to wandb
plt.savefig("ppo_vs_sac_smoothed.png")
wandb.log({"PPO vs SAC": wandb.Image("ppo_vs_sac_smoothed.png")})

# Finish wandb run
wandb.finish()

