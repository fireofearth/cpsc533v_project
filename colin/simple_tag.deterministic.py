import os
import time
import enum
import math
import random
import collections
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pettingzoo.mpe import simple_tag_v2
from pettingzoo.utils import random_demo

from common import (AttrDict, TimeDelta, Normalizer, RewardsShaper, Container,
        get_agent_counts, get_landmark_count, process_config,
        pad_amt, pad_image, moving_average, pad_values_front)
from baseline import SimpleTagNet, choose_action, save_agent

max_cycles=30
env = simple_tag_v2.env(
    num_good=1,
    num_adversaries=1,
    num_obstacles=0,
    max_cycles=max_cycles,
    continuous_actions=False
).unwrapped

eps = 0.02
def deterministic_policy(observation, agent_name):
    """
    Parameters
    ==========
    observation : ndarray
    agent : str
    """
    if "adversary" in agent_name:
        pass
    elif "agent" in agent_name:
        pass
    if agent_name == "adversary_0":
        # get agent_0's
        x, y = observation[4:6]
        if x < -eps: # go left
            return 1
        elif x > eps: # go right
            return 2
        elif y < -eps: # go down
            return 3
        elif y > eps: # go up
            return 4
        else:
            return 0
            # return random.randint(0, 4)
    if agent_name == "agent_0":
        return 0
        # return random.randint(0, 4)
    return 0

env.reset()
print(*[landmark.name for landmark in env.world.landmarks])
print(*[agent.name for agent in env.world.agents])
agent_rewards = 0
reshaped_agent_rewards = 0
adversary_rewards = 0
reshaped_adversary_rewards = 0
rewardshaper = RewardsShaper(env)
normalize = Normalizer(env)
for agent_step_idx, agent_name in enumerate(env.agent_iter()):
    env.render()
    observation, reward, done, info = env.last()
    reshaped_reward = rewardshaper(agent_name, observation)
    # norm_obs = normalize(observation)

    if done:
        env.step(None)
    else:
        action = deterministic_policy(observation, agent_name)
        env.step(action)
        
    if "adversary" in agent_name:
        adversary_rewards += reward
        reshaped_adversary_rewards += reshaped_reward
    elif "agent" in agent_name:
        agent_rewards += reward
        reshaped_agent_rewards += reshaped_reward

    if agent_name == "agent_0":
        pass
    elif agent_name == "adversary_0":
        pass
    
    time.sleep(0.05)

print(f"episode ran for {max_cycles} cycles")
print("agent_rewards", agent_rewards)
print("adversary_rewards", adversary_rewards)
print("reshaped_agent_rewards", reshaped_agent_rewards)
print("reshaped_adversary_rewards", reshaped_adversary_rewards)