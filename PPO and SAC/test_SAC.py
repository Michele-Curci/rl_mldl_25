import gym
import sys
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

def render_test(env_name, model_path):
    env = Monitor(gym.make(env_name))
    env = DummyVecEnv([lambda: env])

    model = SAC.load(model_path)

    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward[0]
        time.sleep(0.001)  # slow down to view
    print("Total reward:", total_reward)

    env.close()

def test_policy(env_name, model_path):
    env = Monitor(gym.make(env_name))
    env = DummyVecEnv([lambda: env])

    model = SAC.load(model_path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward over 10 episodes: {mean_reward}")
    print(f"Standard deviation of reward: {std_reward}")


def main():
    env_name = 'CustomHopper-target-v0'
    model_path = "./models/SAC_Source.zip"  #other models: SAC_Target, SAC_UDR, SAC_NDR

    render_test(env_name, model_path)
    test_policy(env_name, model_path)


if __name__ == '__main__':
    main()