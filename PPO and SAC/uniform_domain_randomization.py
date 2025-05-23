
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
import mujoco_py
import numpy as np
import gym


class HopperGaussianMassWrapper(gym.Wrapper):
    def __init__(self, env, mean=1.0, std=0.3):
        super(HopperGaussianMassWrapper, self).__init__(env)
        self.mean = mean
        self.std = std
        # we want to randomize tight, leg and foot (not torse!)
        self.original_masses = self.env.model.body_mass.copy()

    def reset(self, **kwargs):
        self.randomize_masses()
        return self.env.reset(**kwargs)

    def randomize_masses(self):
        new_masses = self.original_masses.copy()

        # body names: ('world', 'torso', 'thigh', 'leg', 'foot')
        # Randomize thigh, leg, and foot (index 2, 3, 4)
        for i in range(2, 5):
            noise = np.random.normal(loc=self.mean, scale=self.std)
            noise = np.clip(noise, 0.5, 1.5)  # avoid crazy values
            new_masses[i] = self.original_masses[i] * noise

        self.env.model.body_mass[:] = new_masses



log_dir = "./ppo_custom_hopper/"
os.makedirs(log_dir, exist_ok=True)
env_name_source = 'CustomHopper-source-v0'
env_name_target = 'CustomHopper-target-v0'
# env = gym.make(env_name_source)
# print(env.sim.model.body_mass)
# print( env.sim.model.body_names)
env = HopperGaussianMassWrapper(gym.make(env_name_source))
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_hopper")
eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_name_source))])
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=5000, deterministic=True, render=False)
model = PPO(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    n_steps=4096,
    batch_size=128,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
)
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])
model.save(os.path.join(log_dir, "final_model"))
