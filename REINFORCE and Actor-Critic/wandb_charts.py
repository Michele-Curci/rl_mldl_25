import pickle
import wandb
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Initialize Weights & Biases
wandb.init(project="reinforce_vs_actor_critic", name="REINFORCE vs Actor-Critic Line Plot")

# Load REINFORCE results
with open("training_stats_reinforce.pkl", "rb") as f:
    reinforce_data = pickle.load(f)

# Load Actor-Critic results
with open("training_stats_actor_critic.pkl", "rb") as f:
    actor_critic_data = pickle.load(f)

# Extract episode returns
# reinforce_returns = reinforce_data["returns"]
# actor_critic_returns = actor_critic_data["returns"]

# Extract and smooth returns
reinforce_returns = moving_average(reinforce_data["returns"], window_size=100)
actor_critic_returns = moving_average(actor_critic_data["returns"], window_size=100)

# Create x-axis as episode indices
episodes_reinforce = list(range(len(reinforce_returns)))
episodes_ac = list(range(len(actor_critic_returns)))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(episodes_reinforce, reinforce_returns, label="REINFORCE", color="red")
plt.plot(episodes_ac, actor_critic_returns, label="Actor-Critic", color="purple")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("REINFORCE vs Actor-Critic Performance")
plt.legend()
plt.grid(True)

# Save and log to wandb
plt.savefig("reinforce_vs_actor_critic.png")
wandb.log({"REINFORCE vs Actor-Critic": wandb.Image("reinforce_vs_actor_critic.png")})

wandb.finish()
