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

from common import (AttrDict, TimeDelta, Normalizer, RewardsShaper, Container,
        get_agent_counts, get_landmark_count, process_config,
        pad_amt, pad_image, moving_average, pad_values_front)
from baseline import SimpleTagNet, choose_action, save_agent

# torch.autograd.set_detect_anomaly(True)

def make_args(config):
    env = simple_tag_v2.env(
        num_good=config.n_good_agents,
        num_adversaries=config.n_adversaries,
        num_obstacles=config.n_landmarks,
        max_cycles=30,
        continuous_actions=False
    ).unwrapped
    env.reset()

    config.common=AttrDict(
        hidden_size=64,
        enable_rnn=config.enable_rnn,
        enable_messaging=False,
        n_rnn_layers=1,
        n_actions=env.action_space(env.agent_selection).n,
    )
    config.adversary=AttrDict(
        n_agents=config.n_adversaries,
        observation_shape=env.observation_space("adversary_0").shape

    )
    config.agent=AttrDict(
        n_agents=config.n_good_agents,
        observation_shape=env.observation_space("agent_0").shape
    )
    model_tag = "_rnn" if config.enable_rnn else "_mlp"
    config.exp_name = (f"advonly{model_tag}_nadversaries{config.n_adversaries}"
                    f"_ngoodagents{config.n_good_agents}"
                    f"_landmarks{config.n_landmarks}")
    process_config(config)
    print("Using config:", config)
    return env

def plot_training_run(savedir, logger):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.ravel()

    axes[0].plot(logger.episodic_losses.adversary, label="adversary")
    axes[0].set_title("adversary loss")
    axes[0].legend()

    axes[1].plot(logger.episodic_rewards.adversary, label="adversary")
    axes[1].set_title("reward")
    axes[1].legend()

    _moving_average = lambda a, n=64: pad_values_front(moving_average(a, n=n), n=n)
    adversary_episodic_rewards = np.array(logger["episodic_rewards"]["adversary"])
    avg_adversary_episodic_rewards = _moving_average(adversary_episodic_rewards)
    axes[2].plot(avg_adversary_episodic_rewards, label="adversary mean")
    axes[2].set_title("reward")
    axes[2].legend()
    fig.savefig(os.path.join(savedir, "training_run.png"))

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
            env.render()
            if agent_name == "adversary_0":
                # print("rew, shaped rew", round(_reward, 2), round(reward, 2))
                # print("obs, normed obs", np.round(obs_curr, 2), np.round(normalize(obs_curr), 2))
                # print("obs, normed obs", np.round(obs_curr[4:6], 2), np.round(normalize(obs_curr[4:6]), 2))
                # print("obs, rew", np.round(normalize(obs_curr[4:6]), 2), reward)
                pass
            time.sleep(0.05)
        
        if save_video:
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
    if save_video:
        imageio.mimwrite(save_video_path, rendered_video, fps=30)
    return episode, rendered_video

def evaluate_agents(config, container, adversary_net, savedir):
    videodir = os.path.join(savedir, "evaluation")
    os.makedirs(videodir, exist_ok=True)
    # save_agent(os.path.join(savedir, f"adversary-net-{episode_idx}.pth"), adversary_net)
    adversary_net.eval()
    episodic_rewards=AttrDict(adversary=[])
    all_videos = []
    with torch.no_grad():
        for e in range(50):
            save_video = e % 1 == 0
            validation_save_path = None
            should_render = False
            if save_video:
                # validation_save_dir = os.path.join(videodir, f"test{e}")
                validation_save_path = os.path.join(videodir, f"eval{e}.mp4")
                should_render = config.visualize_on_evaluation
            episode, rendered_video = run_episode(
                config, container, adversary_net,
                should_render=should_render, save_video=save_video,
                save_video_path=validation_save_path, is_val=True
            )
            all_videos += rendered_video
            episodic_rewards.adversary.append(episode.reward.adversary)
    print('max at {} reward: {}'.format(torch.argmax(torch.Tensor(episodic_rewards.adversary)), max(episodic_rewards.adversary)))
    print('min at {} reward: {}'.format(torch.argmin(torch.Tensor(episodic_rewards.adversary)), min(episodic_rewards.adversary)))
    avg_adversary_rewards = statistics.fmean(episodic_rewards.adversary)
    print(f"Average evaluation reward after training is: ")
    print(f"    adversary {avg_adversary_rewards:.2f}")
    adversary_net.train()
    return episodic_rewards, all_videos

def evaluate(config, normalizer=None, path=None):
    # if episode_idx % config.evaluation_interval == 0:
    container = Container(config)
    
    device = 'cpu'
    adversary_net = SimpleTagNet(config, "adversary", normalizer=normalizer).to(device)
    adversary_net.eval()
    adversary_target_net = SimpleTagNet(config, "adversary", normalizer=normalizer).to(device)
    adversary_target_net.eval()

    # Load the models!
    # with reward shaping
    adversary_net.load_state_dict(torch.load(os.path.join(path, 'adversary-net-10000.pth'), map_location=torch.device('cpu')))

    eval_episodic_rewards, all_videos = evaluate_agents(
        config, container, adversary_net, path
    )
    # logger.eval_episodic_rewards.append(eval_episodic_rewards)
    
    # save_agent(os.path.join(savedir, "adversary-net-latest.pth"), adversary_net)
    # with open(os.path.join(savedir, "log.json"), "w") as f:
    #      json.dump(logger, f)
    # plot_training_run(savedir, logger)
    return all_videos

def collin(d):
    config = AttrDict()
    for key in d.keys():
        if isinstance(d[key], dict):
            config[key] = collin(d[key])
        else:
            config[key] = d[key]
    return config

import json
if __name__ == "__main__":
    # with reward shaping
    # path = "/Users/frankyu/Desktop/models/advonly_rnn_nadversaries1_ngoodagents1_landmarks0/15_Dec_2021_19_25_49/" 

    # no reward shaping
    path = "/Users/frankyu/Desktop/models/advonly_rnn_nadversaries1_ngoodagents1_landmarks0-noRewardShape/15_Dec_2021_19_25_48"
    
    json_path = os.path.join(path, "config.json")
    with open(json_path) as f:
        config = json.load(f)
    config['device_id'] = 'cpu'
    config = collin(config)
    env = make_args(config)
    normalizer = Normalizer(env) # norm_obs = normalize(obs)
    shapereward = RewardsShaper(env) # reward = shapereward(agent_name, obs)
    criterion = torch.nn.MSELoss()
    all_videos = evaluate(config, normalizer, path)

    imageio.mimwrite("/Users/frankyu/Documents/University/Fall2021/CPSC533V/cpsc533v_project/all_results/single_adversary_noRewardShaping_results.mp4", all_videos, fps=30)

