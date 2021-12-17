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
from baseline import SimpleTagNet, choose_action, save_agent

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
        type=str,
        help="path to config file"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="path to saved adversary weights"
    )
    args = parser.parse_args()

    print('these are the args: ', args.config, args.path)
    with open(args.config) as f:
        config = json.load(f)
    config = load_attr_dict(config)
    config.device_id = "cpu"
    config.path = args.path
    
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
            + observation
            + Q
            + reward
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
    hidden = None
    env.reset()
    for agent_step_idx, agent_name in enumerate(env.agent_iter()):
        if agent_step_idx % n_agents == 0:
            episode.steps += 1
            step_record = AttrDict(adversary={}, agent={})
            episode.step_records.append(step_record)
            
        obs_curr, reward, done, _ = env.last()
        if not is_val:
            reward += shapereward(agent_name, obs_curr)
        if should_render:
            # env.render()
            if agent_name == "adversary_0":
                # print("rew, shaped rew", round(_reward, 2), round(reward, 2))
                # print("obs, normed obs", np.round(obs_curr, 2), np.round(normalize(obs_curr), 2))
                # print("obs, normed obs", np.round(obs_curr[4:6], 2), np.round(normalize(obs_curr[4:6]), 2))
                # print("obs, rew", np.round(normalize(obs_curr[4:6]), 2), reward)
                pass
            # time.sleep(0.05)
        
        # if save_video:
        rendered_image = env.render(mode='rgb_array')
        rendered_video.append(pad_image(rendered_image))

        agent_type, agent_idx = agent_name.split("_")
        agent_idx = int(agent_idx)
        if done:
            step_record[agent_type][agent_idx] = AttrDict(
                observation=obs_curr,
                action=None,
                reward=reward,
                done=done,
            )
            env.step(None)
            continue
        if agent_type == "agent":
            env.step(0)
            step_record[agent_type][agent_idx] = AttrDict(
                observation=obs_curr,
                action=0,
                reward=reward,
                done=done,
            )
        else:
            # agent_type == "adversary"
            hidden = container.get_hidden(agent_name)
            Q_curr, hidden = adversary_net(obs_curr, hidden)
            action = choose_action(config, agent_type, Q_curr, epsilon, is_val=is_val)
            env.step(action)
            step_record[agent_type][agent_idx] = AttrDict(
                observation=obs_curr,
                action=action,
                reward=reward,
                done=done,
            )
            container.update_hidden(agent_name, hidden)
        episode.reward[agent_type] += reward
    
    if should_render:
        env.close()
    # if save_video:
    #     imageio.mimwrite(save_video_path, rendered_video, fps=30)
    return episode, rendered_video

def evaluate_agents(config, container, adversary_net):
    all_video = []
    episodic_rewards=AttrDict(adversary=[])
    with torch.no_grad():
        for e in range(50):
            print(e)
            should_render = e % 1 == 0
            # should_render=0
            episode, rendered_video = run_episode(
                config, container, adversary_net,
                should_render=should_render, is_val=True
            )
            episodic_rewards.adversary.append(episode.reward.adversary)
            all_video += rendered_video
    min_adversary_rewards = min(episodic_rewards.adversary)
    avg_adversary_rewards = statistics.fmean(episodic_rewards.adversary)
    max_adversary_rewards = max(episodic_rewards.adversary)
    print(f"Evaluation reward at episode is: ")
    print(f"    min adversary {min_adversary_rewards:.2f}")
    print(f"    avg adversary {avg_adversary_rewards:.2f}")
    print(f"    max adversary {max_adversary_rewards:.2f}")

    imageio.mimwrite("/Users/frankyu/Documents/University/Fall2021/CPSC533V/cpsc533v_project/all_results/multiple_adversary.mp4", all_video, fps=60)


def evaluate(config, normalizer=None):
    device = torch.device('cpu')
    adversary_net = SimpleTagNet(config, "adversary", normalizer=normalizer).to(device)
    path = config.path
    print('here we go!: ', path)
    adversary_net.load_state_dict(
        torch.load(path, map_location=torch.device('cpu'))
    )
    adversary_net.eval()
    print("Initialized the agent nets.")
    container = Container(config)
    evaluate_agents(
        config, container, adversary_net
    )

if __name__ == "__main__":
    env, config = make_args()
    normalizer = Normalizer(env) # norm_obs = normalize(obs)
    shapereward = RewardsShaper(env) # reward = shapereward(agent_name, obs)
    criterion = torch.nn.MSELoss()
    evaluate(config, normalizer)
