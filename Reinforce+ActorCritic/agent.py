#Change Gamma
#Change NN architecture
#Try using a Relu

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

        """Initializes MLP NNs to map, using forward function, states into
        - actor: parameters of a normal distribution from which to sample the agent's actions N(mu(s(t)), sigma(s(t)))
        - critic: estimate of the state value function V(s(t))
               
        args:
            state_space: dimension of the observation space
            action_space: dim of the action space
            
        """
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
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
        args: 
            x: observation from the environment for the current state s(t)

        returns:
            normal_dist: parameters of the normal distribution from which to sample the agent's actions
            state_value: estimate of the state value function V(s(t))
        """
        
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
        state_value = self.fc3_critic(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', baseline = 0, actor_critic = False):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.actor_critic = actor_critic

        self.baseline = baseline
        self.gamma = 0.99

        self.state_mean = torch.zeros(11).to(self.train_device)
        self.state_var = torch.ones(11).to(self.train_device)
        self.state_count = 1e-5

        self.states = []
        self.next_states = []

        self.state_values = []
        self.next_state_values = []

        self.action_log_probs = []
        self.rewards = []
        self.done = []

    """
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
    """

    def update_policy(self): #Update neural newtworks parameters

        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device)
        states = torch.stack(self.states, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze()
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device)
        done = torch.Tensor(self.done).to(self.train_device)

        if self.actor_critic == False:
        #
        # TASK 2:
        # 1. Analyze the performance of the trained policies in terms of reward and time consumption.
            #With the same number of iterations model with baseline is being trained faster than the one without baseline.
        # 2. How would you choose a good value for the baseline?
            #The baseline should be chosen in a way that it is close to the expected return of the policy of last k episodes (use a moving average). 
            #If it's too low it's useless, if too high it will be too conservative and policy will not be able to learn enough.
        # 3. How does the baseline affect the training, and why?
            #It reduces the variance of the policy gradient estimates, making training faster.
            #It does not modify the expected value of the policy gradient (proof: grad(J) = E(grad(log(pi(a|s))) * (G_t - baseline)) = E(grad(log(pi(a|s)))*G_t)-b*E(grad(log(pi(a|s)))) = ... - 0 since expected value of the gradient of a probability is 0.
            #On the opposite it reduces the variance because it is centering all episodic returns around a value (e.g. 0 if baseline is the average
            #of the last k episodes) and thus it reduces the variance of the policy gradient estimates.

            #   - compute discounted returns
            returns = discount_rewards(rewards,self.gamma) #G_t
            returns = returns - self.baseline
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            #   - compute policy gradient loss function given actions and returns
            loss = -(action_log_probs*returns).mean()        

        

        # TASK 3:
        if self.actor_critic == True:
            state_values = torch.stack(self.state_values, dim=0).to(self.train_device).squeeze()
            future_state_values = torch.stack(self.next_state_values, dim=0).to(self.train_device).squeeze()

            td_target = rewards + self.gamma * future_state_values * (1 - done)
            advantages = (td_target - state_values).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            """ actor loss """
            actor_loss = - (action_log_probs * advantages).mean()
            """ critic loss """
            critic_loss = F.mse_loss(state_values, td_target.detach())
            loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        #Empty lists to prepare for next episode
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done, self.state_values, self.next_state_values = [], [], [], [], [], [], []

        return



    def get_action(self, state, evaluation=False):
        """ If normalization is used """
        #state = self.normalize_state(state)
        #x = state.to(self.train_device)

        """ If normalization is not used """
        x = torch.from_numpy(state).float().to(self.train_device)


        normal_dist, state_value = self.policy(x)
        value = state_value.squeeze()

        if evaluation:  # Return mean of the action distribution
            if self.actor_critic == True:
                return normal_dist.mean, None, value
            else:
                return normal_dist.mean, None, None

        else:   # Sample a "random" action from the distribution (exploration)
            action = normal_dist.sample() #A_t
            # Computes Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum() #log(pi(A_t|S_t))
            if self.actor_critic == True:
                return action, action_log_prob, value
            else:
                return action, action_log_prob, None


    def store_outcome(self, state, next_state, action_log_prob, reward, done, value = None, next_value = None):
        """ Store the outcome of the action taken in the environment """
        self.states.append(torch.from_numpy(state).float())                 #State S_t
        self.next_states.append(torch.from_numpy(next_state).float())       #State S_{t+1}
        self.action_log_probs.append(action_log_prob)                       #log(p(a|s))
        self.rewards.append(torch.Tensor([reward]))                         #R_t
        self.done.append(done)
        if self.actor_critic:
            self.state_values.append(value.view(1))                         #V(S_t)
            self.next_state_values.append(next_value.view(1))               #V(S_{t+1})


