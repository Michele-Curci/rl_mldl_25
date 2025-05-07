import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_rewards(log_dir):
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

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data['l'].cumsum(), data['r'], label='Episode Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/SAC_reward_plot.png")
    plt.close()

def plot_comparison():
    df = pd.read_csv('./statistics/ppo_metrics.csv')

    model_names = [f"Model {i+1}" for i in range(len(df))]
    src_mean = df["Source Env Mean Reward"].values
    src_std = df["Source Env Std Reward"].values
    tgt_mean = df["Target Env Mean Reward"].values
    tgt_std = df["Target Env Std Reward"].values

    width = 0.6
    std_scale = 1  # Adjust the scale factor for std (this will modify how big the std bars are)

    # ===== Plot 1: Source Env =====
    fig, ax1 = plt.subplots()

    x1 = np.arange(len(src_mean))                      # Positions for mean bars
    x2 = np.arange(len(src_std)) + len(src_mean) + 1   # Positions for std bars, with a gap

    # Color matching for each model
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

    bars1 = ax1.bar(x1, src_mean, width, label='Mean Reward', color=colors)
    ax2 = ax1.twinx()  # Create a second y-axis for std scaling
    bars2 = ax2.bar(x2, src_std * std_scale, width, label='Std Reward', color=colors)

    # Set x-ticks to show 'reward' and 'std' labels
    ax2.set_xticks([0.5, 3.5])
    ax2.set_xticklabels(['reward', 'std'], rotation=0)

    # Axis labels and titles
    ax1.set_ylabel('Mean Reward')
    ax2.set_ylabel('Std Reward (scaled)')
    ax1.set_title('Source Environment Performance')
    
    # Legends
    legend = fig.legend([bars1, bars2[1]], ['Source -> Source', 'Target -> Source'], loc='upper right', bbox_to_anchor=(0.95, 0.95))
    legend.set_zorder(100)

    plt.tight_layout()
    plt.savefig('source_env_comparison.png')
    plt.close()

    # ===== Plot 2: Target Env =====
    fig, ax1 = plt.subplots()

    x1 = np.arange(len(tgt_mean))
    x2 = np.arange(len(tgt_std)) + len(tgt_mean) + 1

    bars1 = ax1.bar(x1, tgt_mean, width, label='Mean Reward', color=colors)
    ax2 = ax1.twinx()  # Create a second y-axis for std scaling
    bars2 = ax2.bar(x2, tgt_std * std_scale, width, label='Std Reward', color=colors)

    ax2.set_xticks([0.5, 3.5])
    ax2.set_xticklabels(['reward', 'std'], rotation=0)

    ax1.set_ylabel('Mean Reward')
    ax2.set_ylabel('Std Reward (scaled)')
    ax1.set_title('Target Environment Performance')

    legend = fig.legend([bars1, bars2[1]], ['Target -> Target', 'Source -> Target'], loc='upper right', bbox_to_anchor=(0.95, 0.95))
    legend.set_zorder(100)

    plt.tight_layout()
    plt.savefig('target_env_comparison.png')
    plt.close()

if __name__ == "__main__":
    log_dir = "./sac_custom_hopper/"
    plot_rewards(log_dir)
    #plot_comparison()