import os
import sys
import time
import enum
import math
import random
import collections
import statistics
import json
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

class SimpleTagNet(torch.nn.Module):
    """NN Model for the agents. Both good agents and adversaries use this model."""
        
    def __init__(self, config, agent_type, normalizer=None):
        super().__init__()
        self.device      = torch.device(config.device_id)
        self.observation_size = math.prod(config[agent_type].observation_shape)
        self.n_actions   = config[agent_type].n_actions
        self.hidden_size = config[agent_type].hidden_size
        self.n_rnn_layers = config[agent_type].n_rnn_layers
        self.enable_rnn = config[agent_type].enable_rnn
        self.normalizer = normalizer
        
        if self.enable_rnn:
            print(
                f"Creating baseline RNN net for {agent_type} "
                f"with {self.n_rnn_layers} layers"
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.observation_size, self.hidden_size),
                torch.nn.ReLU(inplace=True),
            )
            self.rnn = torch.nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_rnn_layers,
                batch_first=True
            )
            self.output_mlp = torch.nn.Linear(self.hidden_size, self.n_actions)
        else:
            print(f"Creating baseline MLP net for {agent_type}")
            self.output_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.observation_size, self.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.hidden_size, self.n_actions)
            )
    
    def forward(self, observation, hidden=False):
        """Apply DQN to episode step.
        
        Parameters
        ==========
        observation : ndarray
            The observation vector obtained from the environment.
        
        Returns
        =======
        torch.Tensor
            Vector of Q-value associated with each action.
        """
        if self.normalizer is not None:
            observation = self.normalizer(observation)
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        if self.enable_rnn:
            z = self.mlp(observation)
            z = z.unsqueeze(0).unsqueeze(0)
            z, hidden = self.rnn(z, hidden)
            z = z.squeeze(0).squeeze(0)
            Q = self.output_mlp(z)
            return Q, hidden
        else:
            Q = self.output_mlp(observation)
            return Q, None

def choose_action(config, agent_type, Q, epsilon=0.05, is_val=False):
    if not is_val and random.random() < epsilon:
        return random.randrange(config[agent_type].n_actions)
    else:
        return torch.argmax(Q).item()

def save_agent(savepath, net):
    torch.save(net.state_dict(), savepath)
