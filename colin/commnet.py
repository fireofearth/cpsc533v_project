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

class CommNet(torch.nn.Module):
    """NN Model for the agents. Both good agents and adversaries use this model.
    
    For example:
    - message from each agent has form [x, y] where x,y take on values {0,1}.
    - this message has 2 bits so message_size == 2.
    - there are 4 possible messages to send, namely [0,0], [0,1], [1,0], [1,1]
      so n_messages == 4
    """
        
    def __init__(self, config, agent_type, normalizer=None):
        super().__init__()
        self.device      = torch.device(config.device_id)

        # network hyperparams
        self.hidden_size = config[agent_type].hidden_size
        self.n_rnn_layers = config[agent_type].n_rnn_layers
        self.enable_rnn = config[agent_type].enable_rnn

        # message hyperparams
        self.enable_messaging = config[agent_type].enable_messaging
        self.send_message_size = config[agent_type].message_size
        self.n_messages = 2**self.send_message_size
        self.recv_message_size = config[agent_type].message_size*(config[agent_type].n_agents - 1)

        # env hyperparams
        self.observation_size = math.prod(config[agent_type].observation_shape)
        self.n_actions   = config[agent_type].n_actions

        # computed variables
        print(f"Creating CommNet net for agent of class {agent_type}")
        if self.enable_messaging:
            print(f"    using messsages with {self.send_message_size} bits")
            self.input_size = self.observation_size + self.recv_message_size
            self.output_size = self.n_actions + self.n_messages
        else:
            print("    not sending messsages")
            self.input_size = self.observation_size
            self.output_size = self.n_actions
        print(f"    input, output are resp. {self.input_size}, {self.output_size}")
        
        self.normalizer = normalizer
        if self.normalizer is None:
            print("    not normalizing observation")
        else:
            print(f"    using normalizer {self.normalizer.__class__.__name__}")
            
        if self.enable_rnn:
            print(f"    using RNN net with {self.n_rnn_layers} layers")
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.hidden_size),
                torch.nn.ReLU(inplace=True),
            )
            self.rnn = torch.nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_rnn_layers,
                batch_first=True
            )
            self.output_mlp = torch.nn.Linear(self.hidden_size, self.output_size)
        else:
            print(f"    using MLP net for {agent_type}")
            self.output_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.hidden_size, self.output_size)
            )
    
    def forward(self, observation, hidden=None, message=None, agent_name=None):
        """Apply DQN to episode step.
        
        Parameters
        ==========
        observation : ndarray
            The observation vector obtained from the environment.
        hidden : torch.Tensor or None
            Hidden state of RNN, used if enabled.
        message: torch.Tensor or None
            Messages from other agents, used if enabled.
        
        Returns
        =======
        torch.Tensor
            Vector of Q-value associated with each action.
        torch.Tensor or None
            Hidden state of RNN if enabled
        torch.Tensor or None
            Vector of Q-value associated with each message
        
        Raises
        ======
        TypeError
            When enable_messaging is False but message is passed.
        """
        if self.normalizer is not None:
            observation = self.normalizer(observation, agent_name)
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)

        if self.enable_messaging:
            z = torch.cat((observation, message))
        else:
            z = observation

        if self.enable_rnn:
            z = self.mlp(z)
            z = z.unsqueeze(0).unsqueeze(0)
            z, hidden = self.rnn(z, hidden)
            z = z.squeeze(0).squeeze(0)
            z = self.output_mlp(z)
        else:
            z = self.output_mlp(z)
        
        if self.enable_messaging:
            Q_u, Q_m = z[:self.n_actions], z[self.n_actions:]
        else:
            Q_u, Q_m = z, None

        return Q_u, Q_m, hidden

def choose_action(config, agent_type, Q_u, epsilon=0.05, is_val=False):
    if not is_val and random.random() < epsilon:
        return random.randrange(config[agent_type].n_actions)
    else:
        return torch.argmax(Q_u).item()

def choose_message(config, agent_type, Q_m, epsilon=0.05, is_val=False):
    """Convert message index into a message of form [x, y, ...]
    where x,y,... take on values {0., 1.}.
    
    Converts numbers to list of float. If message_size is 2 then
    - 3 => [1.0, 1.0]
    - 2 => [0.0, 1.0]
    - 1 => [1.0, 0.0]
    - 0 => [0.0, 0.0]
    """
    if not is_val and random.random() < epsilon:
        n_messages = 2**config[agent_type].message_size
        m_idx = random.randrange(n_messages)
        # return np.random.randint(2, size=config[agent_type].message_size)
    else:
        m_idx = torch.argmax(Q_m).item()

    bitstr = list(reversed(f"{m_idx:0b}"))
    def bitstr_to_floatint(i):
        try:
            return float(bitstr[i])
        except IndexError as e:
            return 0.
    m = list(map(bitstr_to_floatint, range(config[agent_type].message_size)))
    return m_idx, m

def save_agent(savepath, net):
    torch.save(net.state_dict(), savepath)
