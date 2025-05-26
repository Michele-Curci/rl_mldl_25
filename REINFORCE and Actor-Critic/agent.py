import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(rewards, gamma):
    discounted = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    return discounted


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = 128
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, action_dim)
        )
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_dim)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1)
        )

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
                torch.nn.init.zeros_(m.bias)


    def forward(self, state):
        mean = self.actor(state)
        std = self.sigma_activation(self.sigma)
        dist = Normal(mean, std)
        value = self.critic(state)
        return dist, value


class Agent(object):
    def __init__(self, policy, device='cpu', use_actor_critic=False, baseline=0, gamma=0.99):
        self.device = device
        self.policy = policy.to(self.device)
        self.gamma = gamma
        self.use_actor_critic = use_actor_critic
        self.baseline = baseline

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=1e-3)

        self.state_mean = torch.zeros(11).to(self.device)
        self.state_var = torch.ones(11).to(self.device)
        self.state_count = 1e-5
        self.episode_count = 0

        self.reset_storage()
        

    def reset_storage(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.state_values = []
        self.next_state_values = []

    def normalize_state(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        self.state_var += delta * (state - self.state_mean)
        std = torch.sqrt(self.state_var / self.state_count)

        # Normalize the state
        normalized_state = (state - self.state_mean) / (std + 1e-8)
        return normalized_state

    def update_policy(self):
        if len(self.states) == 0:
            print("[Warning] No data collected, skipping update")
            return

        self.episode_count += 1

        action_log_probs = torch.stack(self.action_log_probs).to(self.device)
        states = torch.stack(self.states).to(self.device)
        next_states = torch.stack(self.next_states).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device).squeeze()
        done = torch.Tensor(self.done).to(self.device)

        
        if self.use_actor_critic:
            values = torch.stack(self.state_values).squeeze().to(self.device)
            next_values = torch.stack(self.next_state_values).squeeze().to(self.device)

            returns = rewards + self.gamma * next_values * (1 - done)
            advantages = (returns - values).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            normal_dist, _ = self.policy(states)
            entropy = normal_dist.entropy().sum(dim=-1).mean()

            #Update actor
            actor_loss = -(action_log_probs * advantages).mean() - 0.05 * entropy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.actor_optimizer.step()

            #Update critic
            critic_loss = F.mse_loss(values, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.critic_optimizer.step()
            
        else:
            returns = discount_rewards(rewards, self.gamma)
            baseline = self.baseline
            advantage = returns - baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            normal_dist, _ = self.policy(states)
            entropy = normal_dist.entropy().sum(dim=-1).mean()
            loss = -(action_log_probs * advantage).mean() - 0.001 * entropy

            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.actor_optimizer.step()

        # === Debug Info ===
        if self.episode_count is not None and self.episode_count % 1000 == 0:
            print(f"\n[DEBUG] ====== Update Info — Episode {self.episode_count} ======")
            print("Mean reward:", rewards.mean().item())
            if self.use_actor_critic:
                print("Returns mean:", returns.mean().item())
                print("Returns std:", returns.std().item())
                print("Actor loss:", actor_loss.item())
                print("Critic loss:", critic_loss.item())
                print("[DEBUG] Sample value prediction:", values[0].item())
                print("[DEBUG] Sample target return:", returns[0].item())
                print("[DEBUG] Value prediction mean:", values.mean().item())
                print("[DEBUG] Target returns mean:", returns.mean().item())
                print("[DEBUG] Value prediction std:", values.std().item())
                print("[DEBUG] Target returns std:", returns.std().item())
            else:
                print("REINFORCE loss:", loss.item())
            print("=================================\n")


        self.reset_storage()

        return        


    def get_action(self, state, evaluation=False):
        state = self.normalize_state(state)
        x = state.to(self.device)

        dist, value = self.policy(x)
        value = value.squeeze()

        if evaluation:  # Return mean
            if self.use_actor_critic:
                return dist.mean, None, value
            else:
                return dist.mean, None, None

        else:   # Sample from the distribution
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum()
            if self.use_actor_critic:
                return action, action_log_prob, value
            else:
                return action, action_log_prob, None


    def store_outcome(self, state, next_state, action_log_prob, reward, done, value=None, next_value=None):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

        if self.use_actor_critic:
            self.state_values.append(value.view(1))
            self.next_state_values.append(next_value.view(1))