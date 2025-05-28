"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import math

#env_name = 'CustomHopper-source-v0'
env_name = 'CustomHopper-target-v0'

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, variation_ratio=0.3):
        super().__init__(env)
        self.variation_ratio = variation_ratio
        self.body_names = ["thigh", "leg", "foot"]

        # Precompute original masses for each body part
        self.body_masses = {
            name: self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] for name in self.body_names
        }

    def reset(self, **kwargs):
        for name, original_mass in self.body_masses.items():

            std_dev = self.variation_ratio * original_mass
            # Sample from a normal distribution centered at original_mass
            new_mass = np.random.normal(loc=original_mass, scale=std_dev)
            new_mass = np.clip(new_mass, 0.1 * original_mass, 2.0 * original_mass)
            self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] = new_mass

        return self.env.reset(**kwargs)


class WandbEpisodeRewardCallback(BaseCallback):
    """
    Custom callback for logging episode rewards to wandb.
    """
    def __init__(self, verbose=0):
        super(WandbEpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_count += 1
                    # Log to wandb
                    wandb.log({
                        "Episode Reward": info['episode']['r'],
                        "Episode": self.episode_count
                    })
        return True

def linear_decay_lr(initial_lr, final_lr, total_timesteps):
    """ Linear decay of learning rate from initial_lr to final_lr over total_timesteps """
    def schedule(timestep):
        if timestep >= total_timesteps:
            return final_lr
        return initial_lr + (final_lr - initial_lr) * (timestep / total_timesteps)
    return schedule

def light_exponential_decay_lr(initial_lr, final_lr, total_timesteps, decay_rate=2):
    """
    Light exponential decay of learning rate from initial_lr to final_lr over total_timesteps.
    decay_rate > 1: The higher the value, the faster the decay.
    """
    def schedule(timestep):
        if timestep >= total_timesteps:
            return final_lr
        # Normalized step (0 to 1)
        progress = timestep / total_timesteps
        
        # Light exponential decay: we apply a (1 - progress) raised to a power
        decay_factor = (1 - progress) ** decay_rate
        
        # Compute decayed learning rate
        return final_lr + (initial_lr - final_lr) * decay_factor
    
    return schedule

def main():
    timesteps = 1000000
    learning_rate_schedule = linear_decay_lr(0.0003, 0.00001, timesteps)
    #Initialize wandb run
    wandb.init(
    project="hopper-rl",
    name="PPO-Hopper_Target",
    config={
        "env_name": env_name,
        "algorithm": "PPO",
        "total_timesteps": timesteps,
        "gamma": 0.99,
        "n_steps": 4096,
        "batch_size": 128,
        "learning_rate": 0.0003
    }
    )

    train_env = gym.make(env_name)
    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #  
    model = PPO("MlpPolicy", train_env, verbose = 1, gamma=0.99, n_steps=4096, batch_size=128, learning_rate=0.0003)
    model.learn(
        total_timesteps=timesteps,
        callback=[WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models_PPO_Hopper_Target/{wandb.run.id}",
            verbose=1
        ), WandbEpisodeRewardCallback()]
    )
    model.save("ppo_hopper_Target")
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    wandb.log({
        "Mean Reward": mean_reward,
        "Std Reward": std_reward
    })
    wandb.finish() 
if __name__ == '__main__':
    main()
