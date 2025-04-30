import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 128
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = -1.0
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.ln1_critic = torch.nn.LayerNorm(self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.ln2_critic = torch.nn.LayerNorm(self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.ln3_critic = torch.nn.LayerNorm(self.hidden)
        self.fc4_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.ln1_critic(self.fc1_critic(x)))
        x_critic = self.tanh(self.ln2_critic(self.fc2_critic(x_critic)))
        x_critic = self.tanh(self.ln3_critic(self.fc3_critic(x_critic)))
        value = self.fc4_critic(x_critic)

        return normal_dist, value


class Agent(object):
    def __init__(self, policy, device='cpu', use_actor_critic=False, baseline=0):
        self.train_device = device
        self.policy = policy.to(self.train_device)

        self.actor_params = list(self.policy.fc1_actor.parameters()) + \
                    list(self.policy.fc2_actor.parameters()) + \
                    list(self.policy.fc3_actor_mean.parameters()) + \
                    [self.policy.sigma]
        self.critic_params = list(self.policy.fc1_critic.parameters()) + \
                     list(self.policy.fc2_critic.parameters()) + \
                     list(self.policy.fc3_critic.parameters()) + \
                     list(self.policy.fc4_critic.parameters())

        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=1e-3)


        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.9)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=0.9)

        self.gamma = 0.99
        self.use_actor_critic = use_actor_critic
        self.baseline = baseline

        self.state_mean = torch.zeros(11).to(self.train_device)
        self.state_var = torch.ones(11).to(self.train_device)
        self.state_count = 1e-5

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
        state = torch.from_numpy(state).float().to(self.train_device)
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

        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device) #squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device)
        next_states = torch.stack(self.next_states).to(self.train_device)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze()
        done = torch.Tensor(self.done).to(self.train_device)

        
        if self.use_actor_critic:
            values = torch.stack(self.state_values).squeeze()
            next_values = torch.stack(self.next_state_values).squeeze()

            #Bootstrapped returns
            returns = []
            R = next_values[-1] * (1.0 - done[-1])
            for t in reversed(range(len(rewards))):
                R = rewards[t] + self.gamma * R * (1.0 - done[t])
                returns.insert(0, R)
            returns = torch.stack(returns).detach()
            returns = returns / 10.0

            # Clip advantages
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            normal_dist, _ = self.policy(states)
            entropy = normal_dist.entropy().sum(dim=-1).mean()

            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        else:
            returns = discount_rewards(rewards, self.gamma).detach()
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            baseline = self.baseline
            normal_dist, _ = self.policy(states)
            entropy = normal_dist.entropy().sum(dim=-1).mean()
            loss = -(action_log_probs * (returns - baseline)).mean()
            loss = loss - 0.01 * entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # === Debug Info ===
        print("\n[DEBUG] ====== Update Info ======")
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
        x = state.to(self.train_device)

        normal_dist, value = self.policy(x)
        value = value.squeeze()

        if evaluation:  # Return mean
            if self.use_actor_critic:
                return normal_dist.mean, None, value
            else:
                return normal_dist.mean, None, None

        else:   # Sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
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
            self.state_values.append(value.detach().view(1))
            self.next_state_values.append(next_value.detach().view(1))