import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import math
import numpy as np

env_name = 'CustomHopper-source-v0'

class ADRWrapper(gym.Wrapper):
    """Wrap the base environment, appling perturbations to body masses before each episode.
    params: env: The environment to wrap
            adversary_policy: The policy that decides how to perturb the body masses
            body_names: Parts of the leg to modify
            variation_ratio: Maximum perturbation (30% of original mass)"""
    def __init__(self, env, adversary_policy, body_names=["thigh", "leg", "foot"], variation_ratio=0.3):
        super().__init__(env)
        self.adversary = adversary_policy
        self.body_names = body_names
        self.variation_ratio = variation_ratio
        self.original_masses = {
            name: self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] for name in self.body_names
        } #Get all the hopper original masses

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs) #Reset the environment
        obs_tensor = np.array(obs, dtype=np.float32) 
        perturbation = self.adversary.predict(obs_tensor, deterministic=True)[0] #Predict the extent of the perturbation from adversary policy

        for i, name in enumerate(self.body_names):
            """For each part of the hopper (except the torso) we modify the mass based on the predicted perturbation 
            that is in [-1,1] continuous range (see AdversarialEnv class). We also clip it so that the new mass is between 10% and 2x of the original mass"""
            original_mass = self.original_masses[name]
            delta = perturbation[i] * self.variation_ratio * original_mass
            new_mass = np.clip(original_mass + delta, 0.1 * original_mass, 2.0 * original_mass)
            self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] = new_mass

        return self.env.reset(**kwargs)



class AdversarialEnv(gym.Env):
    """
    The custom environment where advesarial is trained. The adversarial chooses mass perturbations and a random action agent tries to maximize its reward.
    params: env_name: Name of the environment to use
            body_names: Parts of the leg to modify
            variation_ratio: Maximum perturbation (30% of original mass)
    """
    def __init__(self, env_name, body_names=["thigh", "leg", "foot"], variation_ratio=0.3):
        super().__init__()
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.body_names = body_names
        self.variation_ratio = variation_ratio
        self.original_masses = {
            name: self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] for name in self.body_names
        }
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.body_names),), dtype=np.float32)
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        return self.obs

    #When the adversarial chooses an action it applies the perturbation to the hopper's leg
    def step(self, action):
        for i, name in enumerate(self.body_names):
            original_mass = self.original_masses[name]
            delta = action[i] * self.variation_ratio * original_mass
            new_mass = np.clip(original_mass + delta, 0.1 * original_mass, 2.0 * original_mass)
            self.env.sim.model.body_mass[self.env.sim.model.body_name2id(name)] = new_mass

        #Simulates an episode with a random policy. The adversary receives reward = negative of total agent reward (goal is to make it fail)
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 1000:
            a = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(a)
            total_reward += reward
            steps += 1
        reward_adv = -total_reward
        return self.obs, reward_adv, True, {}


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

#Trains the adversarial policy (PPO) in the AdversarialEnv 100k episodes
def train_adversary():
    adv_env = DummyVecEnv([lambda: AdversarialEnv('CustomHopper-source-v0')])
    model = PPO("MlpPolicy", adv_env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("adr_adversary")
    return model

#Train the protagonist policy (PPO) using the adversarial policy as wrapper around the source environment
def train_protagonist(adversary):
    train_env = gym.make('CustomHopper-source-v0')
    train_env = ADRWrapper(train_env, adversary)
    print('State space:', train_env.observation_space)
    print('Action space:', train_env.action_space)
    print('Dynamics parameters:', train_env.get_parameters())

    timesteps = 1000000
    wandb.init(
        project="hopper-rl",
        name="PPO-Hopper_Source_with_ADR",
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
    model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.99, n_steps=4096, batch_size=128, learning_rate=0.0003)
    model.learn(
        total_timesteps=timesteps,
        callback=[WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models_PPO_Hopper_Source_with_ADR/{wandb.run.id}",
            verbose=1
        ), WandbEpisodeRewardCallback()]
    )
    model.save("ppo_hopper_Source_with_ADR")
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    wandb.log({
        "Mean Reward": mean_reward,
        "Std Reward": std_reward
    })
    wandb.finish()

def evaluate_on_target(model):
    test_env = gym.make("CustomHopper-target-v0")
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=100, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    return mean_reward, std_reward

def main():
    adversary = train_adversary()
    train_protagonist(adversary)
    test_env = gym.make("CustomHopper-source-v0")       
    model = PPO.load("ppo_hopper_Source_with_ADR", env=test_env)
    mean_reward, std_reward = evaluate_on_target(model)
    print(f"Mean reward on target: {mean_reward} +/- {std_reward}")

if __name__ == '__main__':
    main()