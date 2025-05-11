#python plot_stats.py reinforce_stats.pkl --model-name "REINFORCE"
#python plot_stats.py actor_critic_stats.pkl --model-name "Actor-Critic" --window 20

import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

def load_stats(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def smooth(data, window=10):
    """Compute moving average"""
    return np.convolve(data, np.ones(window) / window, mode='valid')

def moving_std(data, window):
    return [np.std(data[i:i+window]) for i in range(len(data) - window + 1)]

def plot_smoothed_returns(returns, model_name, window=10, output_dir = "."):
    smoothed = smooth(returns, window)
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed, label=f'{model_name} (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Return')
    plt.title(f'{model_name} - Smoothed Returns (Window = {window})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_smoothed_returns.png")
    plt.savefig(filename)
    plt.close()

def plot_std_returns(returns, model_name, window=10, output_dir = "."):
    std_dev = moving_std(returns, window)
    plt.figure(figsize=(10, 5))
    plt.plot(std_dev, label=f'{model_name} (STD)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Return Std Deviation')
    plt.title(f'{model_name} - Return Variability (Window = {window})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_std_returns.png")
    plt.savefig(filename)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", type=str, help="Path to the .pkl stats file")
    parser.add_argument("--model-name", type=str, required=True, help="Model name (e.g. REINFORCE, Actor-Critic)")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window size")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the plots")
    return parser.parse_args()

def main():
    args = parse_args()
    stats = load_stats(args.stats_file)
    returns = stats["returns"]

    print(f"{args.model_name} - Number of episodes: {len(returns)}")

    plot_smoothed_returns(returns, args.model_name, window=args.window, output_dir=args.output_dir)
    plot_std_returns(returns, args.model_name, window=args.window, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
