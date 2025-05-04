import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import sem

def load_stats(path="training_stats.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def smooth(data, window=10):
    """Compute moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_smoothed_returns(filename, reinforce_returns, actor_critic_returns, smoothing_window=10):
    smoothed_reinforce = smooth(reinforce_returns, smoothing_window)
    smoothed_ac = smooth(actor_critic_returns, smoothing_window)
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_reinforce, label='REINFORCE (Smoothed)')
    plt.plot(smoothed_ac, label='Actor-Critic (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Return')
    plt.title(f'Smoothed Returns (Window = {smoothing_window})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_std_returns(filename, reinforce_returns, actor_critic_returns, window=10):
    def moving_std(data, window):
        return [np.std(data[i:i+window]) for i in range(len(data) - window + 1)]

    std_reinforce = moving_std(reinforce_returns, window)
    std_ac = moving_std(actor_critic_returns, window)

    plt.figure(figsize=(10, 5))
    plt.plot(std_reinforce, label='REINFORCE (STD)', color='orange')
    plt.plot(std_ac, label='Actor-Critic (STD)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Return Std Deviation')
    plt.title(f'Return Variability (Window = {window})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    reinforce_stats = load_stats("reinforce_stats.pkl")
    actor_critic_stats = load_stats("actor_critic_stats.pkl")
    reinforce_returns = reinforce_stats["returns"]
    actor_critic_returns = actor_critic_stats["returns"]

    print("REINFORCE returns:", len(reinforce_returns))
    print("Actor-Critic returns:", len(actor_critic_returns))


    plot_smoothed_returns("smoothed_returns.png", reinforce_returns, actor_critic_returns)
    plot_std_returns("std_returns.png", reinforce_returns, actor_critic_returns)

if __name__ == '__main__':
	main()