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
import pprint

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import imageio
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.utils import random_demo

from common import (AttrDict, TimeDelta, Normalizer, RewardsShaper, Container,
        get_agent_counts, get_landmark_count, process_config,
        pad_amt, pad_image, moving_average, pad_values_front)
# from baseline import SimpleTagNet, choose_action, save_agent
from commnet import CommNet, choose_action, choose_message, save_agent

pp = pprint.PrettyPrinter(indent=4)

def file_path(s):
    """Directory path type for argparse"""
    if os.path.isfile(s):
        return s
    else:
        raise argparse.ArgumentTypeError(
                f"{s} is not an existing file path")

def load_attr_dict(d):
    config = AttrDict()
    for key in d.keys():
        if isinstance(d[key], dict):
            config[key] = load_attr_dict(d[key])
        else:
            config[key] = d[key]
    return config

def make_args():
    parser = argparse.ArgumentParser("Evaluate training")
    parser.add_argument(
        "--config",
        type=file_path,
        help="path to config .json file"
    )
    parser.add_argument(
        "--adversary",
        type=file_path,
        help="path to saved adversary weights .pth file"
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    config = load_attr_dict(config)
    config.device_id = "cpu"
    
    env = simple_tag_v2.env(
        num_good=config.n_good_agents,
        num_adversaries=config.n_adversaries,
        num_obstacles=config.n_landmarks,
        max_cycles=config.n_cycles,
        continuous_actions=False
    ).unwrapped
    env.reset()

    print("Using config:")
    pp.pprint(config)
    print("Name of experiment:")
    print(config.exp_name)
    return env, config

def run_episode(
    config, container, adversary_net, epsilon=0.05,
    should_render=False, is_val=False, save_video=False, save_video_path=None
):
    """Run one episodes.
    
    inputs consist of observation, message (backprop), hidden (backprop) indexed by agent
    outputs consist of action, q-value of action (backprop), reward, done indexed by (step, agent)
    
    Returns
    =======
    AttrDict
        Contains episode metrics:
        - steps : number of steps. All agents take an action at each step.
        - reward : episodic rewards indexed by ('adversary', 'agent').
        - step_records : list of quantities produced indiced by step, ('adversary', 'agent'), agent index.
          Each step record has:
            + observation : ndarray of relevant agents' position and velocity.
            + reward : 
            + done
        - loss : contains episodic losses indexed by ('adversary', 'agent'). To be updated by train_agents()
    """
    rendered_video = []
    episode = AttrDict(
        steps=0,
        reward=AttrDict(adversary=0, agent=0),
        step_records=[],
    )
    container.reset()
    n_agents = config.adversary.n_agents + config.agent.n_agents
    step_record = None
    env.reset()
    for agent_step_idx, agent_name in enumerate(env.agent_iter()):
        if agent_step_idx % n_agents == 0:
            episode.steps += 1
            step_record = AttrDict(adversary={}, agent={})
            episode.step_records.append(step_record)
            
        obs_curr, reward, done, _ = env.last()
        if not is_val:
            reward += shapereward(agent_name, obs_curr)
        
        agent_type, agent_idx = agent_name.split("_")
        agent_idx = int(agent_idx)
        
        # hack to make agent stationary and skip network
        if agent_type == "agent":
            env.step(0)
            continue
        # else "adversary"

        if done:
            step_record[agent_type][agent_idx] = AttrDict(
                observation=obs_curr,
                message=None,
                action=None,
                reward=reward,
                done=done,
            )
            env.step(None)
            continue

        if config[agent_type].enable_rnn:
            hidden = container.get_hidden(agent_name)
        else:
            hidden = None

        if config[agent_type].enable_messaging:
            m_curr = container.get_message(agent_name).detach()
        else:
            m_curr = None

        if agent_type == "agent":
            Q_u, Q_m, hidden = agent_net(obs_curr, hidden, m_curr, agent_name)
        else:
            # agent_type == "adversary"
            Q_u, Q_m, hidden = adversary_net(obs_curr, hidden, m_curr, agent_name)

        action = choose_action(config, agent_type, Q_u, epsilon, is_val)

        if config[agent_type].enable_rnn:
            container.update_hidden(agent_name, hidden)

        if config[agent_type].enable_messaging:
            m_idx, m_up = choose_message(config, agent_type, Q_m, epsilon, is_val)
            m_up = torch.tensor(m_up, dtype=torch.float, device=container.device)
            container.update_message(agent_name, m_up)
        else:
            m_idx = None

        if should_render:
            if agent_name == "adversary_0":
                # print("rew, shaped rew", round(_reward, 2), round(reward, 2))
                # print("obs, normed obs", np.round(obs_curr, 2), np.round(normalize(obs_curr), 2))
                # print("obs, normed obs", np.round(obs_curr[4:6], 2), np.round(normalize(obs_curr[4:6]), 2))
                # print("obs, rew", np.round(normalize(obs_curr[4:6]), 2), reward)
                # print("message index, payload", m_idx, m_up)
                pass
            env.render()
            time.sleep(0.01)
        
        if save_video:
            rendered_image = env.render(mode='rgb_array')
            rendered_video.append(pad_image(rendered_image))
        
        env.step(action)
        step_record[agent_type][agent_idx] = AttrDict(
            observation=obs_curr,
            message=m_curr,
            action=action,
            msg_action=m_idx,
            reward=reward,
            done=done,
        )
        episode.reward[agent_type] += reward
    
    if should_render:
        env.close()
    if save_video:
        imageio.mimwrite(save_video_path, rendered_video, fps=30)
    return episode

def evaluate_agents(config, container, adversary_net):
    episodic_rewards=AttrDict(adversary=[])
    with torch.no_grad():
        for e in range(config.n_eval_episodes):
            should_render = e % 10 == 0
            episode = run_episode(
                config, container, adversary_net,
                should_render=should_render, is_val=True
            )
            episodic_rewards.adversary.append(episode.reward.adversary)
    min_adversary_rewards = min(episodic_rewards.adversary)
    avg_adversary_rewards = statistics.fmean(episodic_rewards.adversary)
    max_adversary_rewards = max(episodic_rewards.adversary)
    print(f"Evaluation reward at episode is: ")
    print(f"    min adversary {min_adversary_rewards:.2f}")
    print(f"    avg adversary {avg_adversary_rewards:.2f}")
    print(f"    max adversary {max_adversary_rewards:.2f}")

def evaluate(config, normalizer=None):
    device = torch.device(config.device_id)
    adversary_net = CommNet(config, "adversary", normalizer=normalizer).to(device)
    adversary_net.load_state_dict(
        torch.load(config.adversary)
    )
    adversary_net.eval()
    print("Initialized the agent nets.")
    container = Container(config)
    evaluate_agents(
        config, container, adversary_net
    )

if __name__ == "__main__":
    env, config = make_args()
    if config.enable_pomdp:
        normalizer = POMDPNormalizer(env)
    else:
        normalizer = Normalizer(env) # norm_obs = normalize(obs)
    shapereward = RewardsShaper(env) # reward = shapereward(agent_name, obs)
    criterion = torch.nn.MSELoss()
    evalutate(config, normalizer)
