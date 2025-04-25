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
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
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
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic(x_critic)

        return normal_dist, value


class Agent(object):
    def __init__(self, policy, device='cpu', use_actor_critic=False):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.use_actor_critic = use_actor_critic

        self.reset_storage()
        

    def reset_storage(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.state_values = []
        self.next_state_values = []

    def update_policy(self):
        if len(self.states) == 0:
            print("[Warning] No data collected, skipping update")
            return

        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device) #squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device)

        #debug
        for i, v in enumerate(self.next_state_values):
            if v.shape != self.next_state_values[0].shape:
                print(f'Incosistent tensor at index {i}: shape {v.shape}')
                print("All next_state_values shapes:", [v.shape for v in self.next_state_values])
                

        next_states = torch.stack(self.next_states).to(self.train_device)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze()
        done = torch.Tensor(self.done).to(self.train_device)

        

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #
        if self.use_actor_critic:
            values = torch.stack(self.state_values).squeeze()
            next_values = torch.stack(self.next_state_values).squeeze()

            #Bootstrapped returns
            returns = []
            R=0
            for r, d, nv, in zip(reversed(rewards), reversed(done), reversed(next_values)):
                R = r + self.gamma*R*(1. - d)
                returns.insert(0, R)
            returns = torch.stack(returns).detach()

            advantages = returns - values.detach()

            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + critic_loss
        else:
            returns = discount_rewards(rewards, self.gamma).detach()
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            baseline = 0
            loss = -(action_log_probs * (returns - baseline)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

        # ========== DEBUG INFO ==========
        print("\n[DEBUG] ====== Update Info ======")
        print("Mean reward:", rewards.mean().item())
        print("Std reward:", rewards.std().item())
        print("Sample action log prob:", action_log_probs[0].item())
        print("Sample return:", returns[0].item())

        if self.use_actor_critic:
            print("Mean state value:", values.mean().item())
            print("Std state value:", values.std().item())
            print("Mean advantage:", advantages.mean().item())
            print("Std advantage:", advantages.std().item())
            print("Actor loss:", actor_loss.item())
            print("Critic loss:", critic_loss.item())
        else:
            print("REINFORCE loss:", loss.item())

        total_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        print("Gradient norm (clipped):", total_norm.item())

        sigma_values = self.policy.sigma_activation(self.policy.sigma).detach().cpu().numpy()
        print("Sigma (std dev) values:", sigma_values)
        print("=================================\n")

        self.optimizer.step()

        self.reset_storage()

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, value = self.policy(x)
        value = value.squeeze()

        if evaluation:  # Return mean
            if self.use_actor_critic:
                return normal_dist.mean, None, value
            else:
                return normal_dist.mean, None, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
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