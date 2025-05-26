"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
from env.custom_hopper import *
from pickle_callback import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


#env_name = 'CustomHopper-source-v0'
env_name = 'CustomHopper-target-v0'

def main():
    log_dir = "./ppo_custom_hopper/"
    os.makedirs(log_dir, exist_ok=True)

    train_env = gym.make(env_name)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "monitor.csv"))
    train_env = DummyVecEnv([lambda: train_env])
    #train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    #train_env.training = True

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.envs[0].get_parameters())  # masses of each link of the Hopper


    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_hopper")
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    #eval_env.training = False
    #eval_env.norm_reward = False
    training_stats_path = os.path.join(log_dir, "ppo_training_stats.pkl")
    stats_callback = SaveTrainingStatsCallback(training_stats_path)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=5000, deterministic=True, render=False)

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        verbose=1,
        n_steps=4096,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        tensorboard_log=log_dir
    )

    model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, eval_callback, stats_callback])

    model.save(os.path.join(log_dir, "final_model"))
    # train_env.save(os.path.join(log_dir, "ppo.pkl"))


if __name__ == '__main__':
    main()