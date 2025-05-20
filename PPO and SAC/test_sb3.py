import gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

def render_test(env_name, vecnorm_path, model_path):
    # Load environment and wrap in VecEnv
    env = Monitor(gym.make(env_name))
    env = DummyVecEnv([lambda: env])
    #env = VecNormalize.load(vecnorm_path, env)
    #env.training = False
    #env.norm_reward = False

    # Load trained model
    model = PPO.load(model_path)

    # Run evaluation
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

def test_policy(env_name, vecnorm_path, model_path):
    # Load environment and wrap in VecEnv
    env = Monitor(gym.make(env_name))
    env = DummyVecEnv([lambda: env])
    #env = VecNormalize.load(vecnorm_path, env)
    #env.training = False
    #env.norm_reward = False

    # Load trained model
    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward over 10 episodes: {mean_reward}")
    print(f"Standard deviation of reward: {std_reward}")

def evaluate_model_on_env(model, env, n_eval_episodes=10):
    # Wrap the environment with Monitor to track rewards and lengths
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    return mean_reward, std_reward

def save_metrics():
    # Load environments
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    # Load trained models
    model_source = PPO.load("./models/PPO_Source_model.zip")
    model_target = PPO.load("./models/PPO_Target_model.zip")

    # Evaluate models on both environments
    ss_model_mean_reward, ss_model_std_reward = evaluate_model_on_env(model_source, source_env)
    st_model_mean_reward, st_model_std_reward = evaluate_model_on_env(model_source, target_env)
    tt_model_mean_reward, tt_model_std_reward = evaluate_model_on_env(model_target, target_env)
    ts_model_mean_reward, ts_model_std_reward = evaluate_model_on_env(model_target, source_env)

    # Prepare data for CSV
    data = {
        'Source Env Mean Reward': [ss_model_mean_reward, ts_model_mean_reward],
        'Source Env Std Reward': [ss_model_std_reward, ts_model_std_reward],
        'Target Env Mean Reward': [tt_model_mean_reward, st_model_mean_reward],
        'Target Env Std Reward': [tt_model_std_reward, st_model_std_reward],
    }

    # Save data to CSV
    df = pd.DataFrame(data)
    df.to_csv('./statistics/ppo_metrics.csv', index=False)

def main():
    #env_name, vecnorm_path, model_path = 'CustomHopper-source-v0', "./models/PPO_Source_vecnormalize.pkl", "./models/UDR_PPO_Source_model.zip"
    env_name, vecnorm_path, model_path = 'CustomHopper-target-v0', "./models/PPO_Target_vecnormalize.pkl", "./models/PPO_Target_model.zip"
    model_path = "./ppo_target_hopper/best_model.zip"
    env_name = 'CustomHopper-target-v0'
    #Use these only for single model test
    render_test(env_name, vecnorm_path, model_path)
    test_policy(env_name, vecnorm_path, model_path)

    #Use this to compare the two models
    #save_metrics()

if __name__ == '__main__':
    main()