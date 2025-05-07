import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import gym


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
        # I decided to use action-value function (Q-function) as critic at first,
        # but it didn't work well, so I changed it to state-value function (V-function)
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)  
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)  


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

        return normal_dist
    

    def critic_forward(self, x):
        """
            Critic: now takes only state
        """
        v = self.tanh(self.fc1_critic(x))
        v = self.tanh(self.fc2_critic(v))
        v_value = self.fc3_critic_value(v)

        return v_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5)

        self.actor_params = list(self.policy.fc1_actor.parameters()) + \
            list(self.policy.fc2_actor.parameters()) + \
            list(self.policy.fc3_actor_mean.parameters()) + \
            [self.policy.sigma]  
        self.critic_params = list(self.policy.fc1_critic.parameters()) + \
                list(self.policy.fc2_critic.parameters()) + \
                list(self.policy.fc3_critic_value.parameters())
        
        self.optimizer_critic = torch.optim.Adam(self.critic_params, lr=3e-5)
        self.optimizer_actor = torch.optim.Adam(self.actor_params, lr=3e-5)
        self.optimizer_value = torch.optim.Adam(self.critic_params, lr=3e-5)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.values = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []



    def PPO(self, env, maxSteps=10000):
        done = False
        state = env.reset()
        steps = 0
        episode_reward = 0

        ### 1. collect trajectory (experience)
        for i in range(2048):
            while not done and steps < maxSteps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
                distribution = self.policy(state_tensor)
                action = distribution.sample()
                action_log_prob = distribution.log_prob(action).sum()

                V_s = self.policy.critic_forward(state_tensor)

                next_state, reward, done, _ = env.step(action.cpu().numpy())

                # Store the outcome of the stocastic astion and corresponding reward and value fnunction
                self.states.append(state_tensor)
                self.actions.append(action)
                self.action_log_probs.append(action_log_prob)
                self.rewards.append(torch.tensor([reward], device=self.train_device, dtype=torch.float32))

                self.done.append(done)
                self.values.append(V_s)

                state = next_state
                steps += 1
                episode_reward += reward

        ### 2. evaluate the trajectory (experience)
        #   -> we want to calculate 
        #       A_{t} = r + \gamma*V(s+1) - V(s)

        #   2.1. compute gamma*V(s+1)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.train_device)
        with torch.no_grad():
            V_s_next = self.policy.critic_forward(next_state_tensor)
        returns = []
        for r, d in zip(reversed(self.rewards), reversed(self.done)):
            V_s_next = r + self.gamma * V_s_next * (1-d)
            returns.insert(0, V_s_next)
            

        #   2.2. compute tensors
        returns_tensor = torch.cat(returns).detach()
        values_tensor = torch.cat(self.values).detach()

        #   2.3. compute advantage and normalize it
        advantage = returns_tensor - values_tensor
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = advantage.detach()


        ### 3. Calculate improvement with clipping (Heart of PPO <3)

        #   generate state,actions and log-prob actions tensors
        states_tensor = torch.cat(self.states)
        actions_tensor = torch.cat(self.actions)
        actions_log_prob_tensor = torch.stack(self.action_log_probs).detach()

        #   first we need an inner loop to define the best result for clipping (between 4-10)
        ppo_epochs = 10
        for i in range(ppo_epochs):
            distribution = self.policy(states_tensor)
            next_actions_log_prob_tensor = distribution.log_prob(actions_tensor).sum(axis=-1)

            # now we should compute the division result between old and new policy
            # to do so, we compute exponential of the difference between log prob,
            # based on this formula:  e^{ln(a) - (b)} = a/b
            ratio = torch.exp(next_actions_log_prob_tensor - actions_log_prob_tensor)

            # now, the actual clipping...
            clip_range = 0.2
            clipped_ratio = torch.clip(ratio, 1 - clip_range, 1 + clip_range)

            # Loss_{clip} = E[min(ratio * Advantage, clipped_ratio * Advantage)]
            clipped_loss = -torch.min(ratio * advantage.squeeze(), clipped_ratio * advantage.squeeze()).mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            clipped_loss = clipped_loss - 0.01 * entropy  # 0.01 is the entropy coefficient



            self.optimizer.zero_grad()
            clipped_loss.backward()
            self.optimizer.step() 

        
        ### update policy and value function by MSE of loss
        value_epochs = 10
        for i in range(value_epochs):
            value_predicted = self.policy.critic_forward(states_tensor)
            value_loss = F.mse_loss(value_predicted, returns_tensor)
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        self.states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()
        self.values.clear()

        return episode_reward

