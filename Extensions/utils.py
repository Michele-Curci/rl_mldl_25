import gym
import torch
import copy
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

def sample_task():
    env = gym.make('CustomHopper-source-v0')
    #env.sim.model.body_mass[env.sim.model.body_name2id("thigh")] *= np.random.uniform(0.7, 1.3)
    #env.sim.model.body_mass[env.sim.model.body_name2id("leg")] *= np.random.uniform(0.7, 1.3)
    #env.sim.model.body_mass[env.sim.model.body_name2id("foot")] *= np.random.uniform(0.7, 1.3)
    env.sim.model.body_mass[env.sim.model.body_name2id("thigh")] = np.random.normal(env.sim.model.body_mass[env.sim.model.body_name2id("thigh")])
    env.sim.model.body_mass[env.sim.model.body_name2id("leg")] = np.random.normal( env.sim.model.body_mass[env.sim.model.body_name2id("leg")])
    env.sim.model.body_mass[env.sim.model.body_name2id("foot")] = np.random.normal(env.sim.model.body_mass[env.sim.model.body_name2id("foot")])
    return env

def clone_model_weights(model):
    return [param.clone().detach() for param in model.policy.parameters()]

def set_model_weights(model, weights):
    with torch.no_grad():
        for param, w in zip(model.policy.parameters(), weights):
            param.copy_(w)
