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
import imageio
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.utils import random_demo

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TimeDelta(object):
    def __init__(self, delta_time):
        """Convert time difference in seconds to days, hours, minutes, seconds.
        
        Parameters
        ==========
        delta_time : float
            Time difference in seconds.
        """
        self.fractional, seconds = math.modf(delta_time)
        seconds = int(seconds)
        minutes, self.seconds = divmod(seconds, 60)
        hours, self.minutes = divmod(minutes, 60)
        self.days, self.hours = divmod(hours, 24)
    
    def __repr__(self):
        return f"{self.days}-{self.hours:02}:{self.minutes:02}:{self.seconds + self.fractional:02}"

class Normalizer(object):
    def __init__(self, env):
        self.n_landmarks = len(env.world.landmarks)
        self.n_allagents = len(env.world.agents)
        self.n_good = sum(map(lambda a: not a.adversary, env.world.agents))
    
    @staticmethod
    def normalize_abs_pos(s):
        """Clip absolute position and scale to [-1, 1]
        s is a scalar or an ndarray of one dimension."""
        return np.clip(s, -1.5, 1.5) / 1.5

    @staticmethod
    def normalize_rel_pos(s):
        """Clip relative position and scale to [-1, 1]
        s is a scalar or an ndarray of one dimension."""
        return np.clip(s, -3, 3) / 3

    def __call__(self, obs):
        # normalize and clip positions
        norm_obs = obs.copy()
        # normalize velocity of current entity
        norm_obs[:2] = norm_obs[:2] / 1.3
        # clip/scale abs. position of current entity
        norm_obs[2:4] = self.normalize_abs_pos(norm_obs[2:4])
        # clip/scale rel. position of other entities
        n_range = self.n_landmarks + self.n_allagents - 1
        for i in range(n_range):
            norm_obs[4 + (2*i):4 + (2*(i + 1))] = self.normalize_rel_pos(
                norm_obs[4 + (2*i):4 + (2*(i + 1))]
            )
        # normalize velocity of other entities
        norm_obs[4 + (2*n_range):] = norm_obs[4 + (2*n_range):] / 1.3
        return norm_obs

class RewardsShaper(object):
    def __init__(self, env):
        self.n_landmarks = len(env.world.landmarks)
        # self.n_allagents = len(env.world.agents)
        self.name_to_idx = {agent.name: i for i, agent in enumerate(env.world.agents)}
        self.idx_to_name = {i: agent.name for i, agent in enumerate(env.world.agents)}
        self.goodagent_indices = [
            i for i, agent in enumerate(env.world.agents) if agent.name.startswith("agent")
        ]
        self.adversary_indices = [
            i for i, agent in enumerate(env.world.agents) if agent.name.startswith("adversary")
        ]
        # rdist - distance between adversary-good agent to start computing rewards.
        self.rdist = 1
        # collision_dist - distance between adversary-good agent to count collision.
        #    Based on PettingZoo numbers. 
        self.collision_dist = 0.075 + 0.05

    @staticmethod
    def bound(x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)
        
    def __call__(self, agent_name, obs):
        """Compute reshaped rewards from observation for agent given agent name.
        Adversary: start gaining small rewards as it nears good agents.
        
        Good agent: starts gaining small penality as it nears bad agents.
        """
        _obs = obs[4 + (2*self.n_landmarks):]
        agent_idx = self.name_to_idx[agent_name]
        cum_r = 0.
        if agent_name.startswith("agent"):
            # penalty across all adversaries
            for adversary_idx in self.adversary_indices:
                # penalty from distance of adversary; penalty of collision
                other_idx = adversary_idx - 1 if agent_idx < adversary_idx else adversary_idx
                x, y = _obs[2*other_idx:(2*other_idx) + 2]
                d    = math.sqrt(x**2 + y**2)
                cum_r -= 1 - (1/self.rdist)*d
        
        elif agent_name.startswith("adversary"):
            # reward across all agents
            for goodagent_idx in self.goodagent_indices:
                # reward from distance to agent; reward of collision
                other_idx = goodagent_idx - 1 if agent_idx < goodagent_idx else goodagent_idx
                x, y = _obs[2*other_idx:(2*other_idx) + 2]
                d    = math.sqrt(x**2 + y**2)
                cum_r += 1 - (1/self.rdist)*d
        
        return cum_r

def get_agent_counts(env):
    all_agents = 0
    adversaries = 0
    for agent in env.world.agents:
        all_agents += 1
        adversaries += 1 if agent.adversary else 0
    good_agents = all_agents - adversaries
    return (adversaries, good_agents)

def get_landmark_count(env):
    return len(env.world.landmarks)

def process_config(config):
    for k, v in config.common.items():
        config.adversary[k] = v
        config.agent[k] = v

def pad_amt(w,  macro_block_size=16):
    amt = w % macro_block_size
    if amt > 0:
        return macro_block_size - amt
    else:
        return 0

def pad_image(img, macro_block_size=16):
    """Pad a image of shape (W, H, C)"""
    _pad_amt = lambda w: pad_amt(w)
    return np.pad(img, [(0, _pad_amt(img.shape[0])), (0, _pad_amt(img.shape[1])), (0, 0)])

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pad_values_front(a, n):
    """Zero out beginning of last axis"""
    pad_width = [(0, 0) for _ in range(a.ndim - 1)] + [(n, 0)]
    return np.pad(a, pad_width, mode='constant', constant_values=0)
