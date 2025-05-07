import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *


env_name = 'CustomHopper-source-v0'
#env_name = 'CustomHopper-target-v0'

def main():
    log_dir = "./sac_custom_hopper/"
    os.makedirs(log_dir, exist_ok=True)
    print("Created or found log_dir at:", os.path.abspath(log_dir))


    train_env = gym.make(env_name)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "monitor.csv"))
    train_env = DummyVecEnv([lambda: train_env])

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.envs[0].get_parameters())  # masses of each link of the Hopper


    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_hopper")
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=5000, deterministic=True, render=False)

    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        ent_coef='auto',
        learning_starts=10000,
        target_update_interval=1,
        tensorboard_log=log_dir
    )

    model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, eval_callback])

    model.save(os.path.join(log_dir, "final_model"))


if __name__ == '__main__':
    main()