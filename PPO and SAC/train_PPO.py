import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *


env_name = 'CustomHopper-source-v0'
#env_name = 'CustomHopper-target-v0'

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, variation_ratio=0.3):
        super().__init__(env)
        self.variation_ratio = variation_ratio
        self.body_names = ["thigh", "leg", "foot"]

        # Original body masses
        self.body_masses = {
            name: self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] for name in self.body_names
        }

    def reset(self, **kwargs):
        for name, mass in self.body_masses.items():
            # Bounds for uniform distribution
            low = (1.0 - self.variation_ratio) * mass
            high = (1.0 + self.variation_ratio) * mass

            # Choose a distribution 
            #new_mass = np.random.uniform(low, high)
            new_mass = np.random.normal(mass)

            # Apply it
            self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] = new_mass

        return self.env.reset(**kwargs)

def main():
    log_dir = "./TEST_ppo_source_hopper/"
    os.makedirs(log_dir, exist_ok=True)

    train_env = gym.make(env_name)
    train_env = DomainRandomizationWrapper(train_env, variation_ratio=0.3)
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "monitor.csv"))
    train_env = DummyVecEnv([lambda: train_env])

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_hopper")
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
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

    model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, eval_callback])

    model.save(os.path.join(log_dir, "final_model"))


if __name__ == '__main__':
    main()