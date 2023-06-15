import numpy as np
import gym
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
import collections
import threading as td
import time
import copy
import torch
import torch.nn as nn

from test_MAPPO.util import get_shape_from_obs_space, get_shape_from_act_space
from test_MAPPO.Layers import ACTLayer, MLPLayer


n_process = mp.cpu_count()

env = gym.make('CartPole-v1')

obs_space = env.observation_space
act_space = env.action_space
hidden_size = 256
layer_N = 2


class A2C(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size, layer_N):
        super(A2C, self).__init__()

        self.obs_shape = get_shape_from_obs_space(obs_space)
        self.act_shape = get_shape_from_act_space(act_space)
        print(self.obs_shape, self.act_shape)
        self.hidden_size = hidden_size
        self.layer_N = layer_N

        self.mlp = MLPLayer(self.obs_shape, self.hidden_size, self.layer_N)

        self.act = ACTLayer(act_space, self.hidden_size)

    def action(self, action):
        return self.act(action)

    def train(self, data):
        pass


a2c = A2C(obs_space, act_space, hidden_size, layer_N)

print(a2c)







