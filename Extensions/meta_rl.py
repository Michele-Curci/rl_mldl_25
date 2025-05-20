import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import sample_task, clone_model_weights, set_model_weights

def reptile_meta_train(meta_iters=200, n_tasks=5, inner_steps=5000, meta_lr=0.1, save_path="reptile_meta_model"):
    os.makedirs(save_path, exist_ok=True)

    # Initial policy
    base_env = DummyVecEnv([lambda: sample_task()])
    model = PPO("MlpPolicy", base_env, verbose=0)

    for meta_iter in range(meta_iters):
        initial_weights = clone_model_weights(model)
        weight_deltas = []

        for _ in range(n_tasks):
            # Sample new task
            task_env = DummyVecEnv([lambda: sample_task()])
            task_model = PPO("MlpPolicy", task_env, verbose=0)
            set_model_weights(task_model, initial_weights)

            # Inner loop training on the task
            task_model.learn(total_timesteps=inner_steps)

            # Compute weight difference
            updated_weights = clone_model_weights(task_model)
            delta = [uw - iw for uw, iw in zip(updated_weights, initial_weights)]
            weight_deltas.append(delta)

        # Average weight deltas
        avg_delta = [sum(deltas) / len(deltas) for deltas in zip(*weight_deltas)]

        # Update base model weights using meta-gradient step
        new_weights = [iw + meta_lr * d for iw, d in zip(initial_weights, avg_delta)]
        set_model_weights(model, new_weights)

        print(f"[Meta Iter {meta_iter}] Meta-update completed.")

        if (meta_iter + 1) % 20 == 0:
            model.save(os.path.join(save_path, f"ppo_reptile_iter{meta_iter+1}"))

    model.save(os.path.join(save_path, "ppo_reptile_final"))

if __name__ == "__main__":
    reptile_meta_train()
