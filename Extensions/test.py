from stable_baselines3 import PPO
from utils import sample_task
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import sample_task, clone_model_weights, set_model_weights
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

def evaluate_policy(model, env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]  # use [0] because DummyVecEnv wraps single env
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def test_reptile_meta_model(model_path="reptile_normal_model0.5/ppo_reptile_final.zip", 
                            inner_steps=5000, 
                            n_test_tasks=5, 
                            n_eval_episodes=5):
    
    # Step 1: Load the meta-trained model
    env = DummyVecEnv([lambda: gym.make('CustomHopper-target-v0')])
    meta_model = PPO.load(model_path, env=env)

    pre_adapt_rewards = []
    post_adapt_rewards = []

    for task_idx in range(n_test_tasks):
        print(f"\n[Testing on Task {task_idx+1}]")
        
        # Step 2: Sample a new test task
        task_env = DummyVecEnv([lambda: sample_task()])
        
        # Step 3: Clone the model with meta weights
        test_model = PPO("MlpPolicy", task_env, verbose=0)
        meta_weights = clone_model_weights(meta_model)
        set_model_weights(test_model, meta_weights)

        # Step 4: Evaluate before adaptation
        mean_reward_pre, _ = evaluate_policy(test_model, task_env, n_eval_episodes)
        print(f"Mean reward before adaptation: {mean_reward_pre:.2f}")
        pre_adapt_rewards.append(mean_reward_pre)

        # Step 5: Fine-tune (adapt) on the task
        test_model.learn(total_timesteps=inner_steps)

        # Step 6: Evaluate after adaptation
        mean_reward_post, _ = evaluate_policy(test_model, task_env, n_eval_episodes)
        print(f"Mean reward after adaptation: {mean_reward_post:.2f}")
        post_adapt_rewards.append(mean_reward_post)

    # Summary
    print("\n=== Meta-Test Summary ===")
    print(f"Avg reward before adaptation: {np.mean(pre_adapt_rewards):.2f}")
    print(f"Avg reward after adaptation:  {np.mean(post_adapt_rewards):.2f}")

if __name__ == "__main__":
    test_reptile_meta_model()