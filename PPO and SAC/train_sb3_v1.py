"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)
   This script focuses on using the PPO algorithm with gym.make() for env instantiation.
"""
import gym

from env.custom_hopper import *

import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

def main():
    ALGORITHM_NAME = "PPO"
    TOTAL_TIMESTEPS = 1_000_000
    MODEL_FILENAME_PREFIX = f"hopper_{ALGORITHM_NAME.lower()}_gymmake"

    SOURCE_ENV_ID = 'CustomHopper-source-v0'
    TARGET_ENV_ID = 'CustomHopper-target-v0'

    # Directories for saving models and logs
    SAVE_DIR = "trained_models_ppo_gymmake"
    BEST_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"{MODEL_FILENAME_PREFIX}_best")
    FINAL_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"{MODEL_FILENAME_PREFIX}_final")
    LOG_DIR = f"./sb3_hopper_logs_{ALGORITHM_NAME.lower()}_gymmake/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    #Initialize Training Environment (Source Domain) using gym.make()
    train_env = gym.make(SOURCE_ENV_ID)

    print(f"--- Training Environment: {SOURCE_ENV_ID} (using gym.make()) ---")
    print('State space (source):', train_env.observation_space)
    print('Action space (source):', train_env.action_space)
    if hasattr(train_env.unwrapped, 'get_parameters'): # Access unwrapped env for custom methods
        print('Dynamics parameters (source):', train_env.unwrapped.get_parameters())
    elif hasattr(train_env, 'get_parameters'):
        print('Dynamics parameters (source):', train_env.get_parameters())
    else:
        print("Source environment does not have a 'get_parameters' method directly or on unwrapped.")


    #Initialize Target Environment (Target Domain) for Evaluation
    # For EvalCallback
    eval_env_callback = gym.make(TARGET_ENV_ID)
    print(f"\n--- Evaluation Callback Environment: {TARGET_ENV_ID} (using gym.make()) ---")
    # You might want to print its details too if needed for debugging callbacks

    # For final testing
    test_env = gym.make(TARGET_ENV_ID)
    print(f"\n--- Target Test Environment: {TARGET_ENV_ID} (using gym.make()) ---")
    print('State space (target):', test_env.observation_space)
    print('Action space (target):', test_env.action_space)
    if hasattr(test_env.unwrapped, 'get_parameters'):
        print('Dynamics parameters (target):', test_env.unwrapped.get_parameters())
    elif hasattr(test_env, 'get_parameters'):
        print('Dynamics parameters (target):', test_env.get_parameters())
    else:
        print("Target environment does not have a 'get_parameters' method directly or on unwrapped.")


    eval_callback = EvalCallback(eval_env_callback, # Pass the single gym.Env instance
                                 best_model_save_path=BEST_MODEL_SAVE_PATH,
                                 log_path=LOG_DIR,
                                 eval_freq=10000,
                                 n_eval_episodes=5,
                                 deterministic=True,
                                 render=False)

    # SB3 PPO will internally wrap train_env with a DummyVecEnv if it's not already a VecEnv.
    model = PPO("MlpPolicy",
                train_env, # Pass the single gym.Env instance
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=3e-4,
                n_steps=2048, # For PPO, this is per environment. SB3 handles if it's a single env.
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                )

    print(f"\nStarting training with {ALGORITHM_NAME} on {SOURCE_ENV_ID} (using gym.make()) for {TOTAL_TIMESTEPS} timesteps...")
    print(f"TensorBoard logs will be saved to: {LOG_DIR}")
    print(f"Best models during training will be saved to: {BEST_MODEL_SAVE_PATH}")

    # --- Train the Agent

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                tb_log_name=MODEL_FILENAME_PREFIX)

    # --- Save the Final Model ---
    model.save(FINAL_MODEL_SAVE_PATH)
    print(f"Final PPO model saved to {FINAL_MODEL_SAVE_PATH}.zip")

    # --- Evaluate the Trained Policy on the Target Environment ---
    print("\n--- Evaluating Trained PPO Policy on Target Domain ---")

    best_model_full_path = os.path.join(BEST_MODEL_SAVE_PATH, "best_model.zip")

    if os.path.exists(best_model_full_path):
        print(f"Loading best PPO model from {best_model_full_path} for evaluation...")
        # When loading, you can provide the 'env' argument to set the environment for the loaded model.
        # This is useful if the environment has changed or if you want to ensure it's set correctly.
        loaded_model = PPO.load(best_model_full_path, env=test_env)
    else:
        print(f"No best model found at {best_model_full_path}. Using the final model for evaluation.")
        print(f"Loading final PPO model from {FINAL_MODEL_SAVE_PATH}.zip for evaluation...")
        loaded_model = PPO.load(FINAL_MODEL_SAVE_PATH, env=test_env)

    print(f"Evaluating the loaded PPO model on the target environment: {TARGET_ENV_ID}")
    mean_reward, std_reward = evaluate_policy(loaded_model,
                                              test_env, # test_env is already a gym.make() instance
                                              n_eval_episodes=10,
                                              deterministic=True)
    print(f"Evaluation on {TARGET_ENV_ID}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if train_env:
        train_env.close()
    if eval_env_callback:
        eval_env_callback.close()
    if test_env:
        test_env.close()
    print("\nScript finished.")

if __name__ == '__main__':
    main()
