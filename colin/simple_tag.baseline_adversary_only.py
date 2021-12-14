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

from common import (AttrDict, TimeDelta, Normalizer, RewardsShaper,
        get_agent_counts, get_landmark_count, process_config,
        pad_amt, pad_image, moving_average, pad_values_front)
from baseline import SimpleTagNet, choose_action, save_agent

# torch.autograd.set_detect_anomaly(True)

def make_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--n-adversaries", type=int, default=3, help="number of adversary agents")
    parser.add_argument("--n-good-agents", type=int, default=1, help="number of good agents")
    parser.add_argument("--n-landmarks", type=int, default=0, help="number of landmarks")
    parser.add_argument("--n-cycles", type=int, default=30, help="number cycles in each episode")
    parser.add_argument("--discount", type=float, default=0.99, help="discount")
    parser.add_argument("--eps-start", type=float, default=0.99, help="start epsilon for epsilon-greedy strategy")
    parser.add_argument("--eps-end", type=float, default=0.05, help="end epsilon for epsilon-greedy strategy")
    parser.add_argument("--eps-decay", type=float, default=0.9996, help="epsilon decay for epsilon-greedy strategy")
    parser.add_argument("--n-episodes", type=int, default=20_000, help="number of episodes")
    parser.add_argument("--batch-size", type=int, default=1, help="number of batches of episodes when optimizing")
    parser.add_argument("--update-target-interval", type=int, default=32, help="interval to update target network")
    parser.add_argument("--report-interval", type=int, default=64, help="how frequently to compute running means of losses and rewards")
    parser.add_argument("--visualize", action='store_true', help="whether to visualize an episode in a window")
    parser.add_argument("--visualize-interval", type=int, default=64, help="how frequently to visualize an episode in a window")
    parser.add_argument("--evaluation-interval", type=int, default=500, help="how frequently to evaluate agents, and save the results")
    parser.add_argument("--n-eval-episodes", type=int, default=100, help="how many episodes to run for each evaluation")
    parser.add_argument("--visualize-on-evaluation", action='store_true', help="whether to visualize in a window when evaluating")
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device-id", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-rnn", action='store_true')
    config = parser.parse_args()
    config = AttrDict(**vars(config))

    env = simple_tag_v2.env(
        num_good=config.n_good_agents,
        num_adversaries=config.n_adversaries,
        num_obstacles=config.n_landmarks,
        max_cycles=config.n_cycles,
        continuous_actions=False
    ).unwrapped
    env.reset()

    config.common=AttrDict(
        hidden_size=32,
        enable_rnn=True,
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
    config.exp_name = (f"advonly_nadversaries{config.n_adversaries}"
                    f"_ngoodagents{config.n_good_agents}"
                    f"_landmarks{config.n_landmarks}")
    process_config(config)
    print("Using config:", config)
    return env, config

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
    config, adversary_net, epsilon=0.05,
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
            Q_curr, hidden = adversary_net(obs_curr, hidden)
            action = choose_action(config, agent_type, Q_curr, epsilon, is_val=is_val)
            env.step(action)
            step_record[agent_type][agent_idx] = AttrDict(
                observation=obs_curr,
                action=action,
                reward=reward,
                done=done,
            )
        episode.reward[agent_type] += reward
    
    if should_render:
        env.close()
    if save_video:
        imageio.mimwrite(save_video_path, rendered_video, fps=30)
    return episode

def train_agents(
    config, device, batch, adversary_net,
    adversary_target_net, adversary_optimizer
):
    """Compute loss of episode and update agent weights."""
    adversary_optimizer.zero_grad()
    discount = torch.tensor(config.discount, dtype=torch.float, device=device)
    adversary_losses = []
    
    for episode in batch:
        hidden = None
        target_hidden = None
        for step_idx in range(episode.steps):
            # Optimize adversary network
            for agent_idx in episode.step_records[step_idx].adversary.keys():
                curr_record = episode.step_records[step_idx].adversary[agent_idx]
                if curr_record.done:
                    # agent is done at this step
                    continue
                next_record = episode.step_records[step_idx + 1].adversary[agent_idx]
                r = torch.tensor(next_record.reward, dtype=torch.float, device=device)
                y = None
                if next_record.done:
                    # agent terminates at next step
                    y = r
                else:
                    with torch.no_grad():
                        next_o = next_record.observation
                        target_Q, target_hidden = adversary_target_net(next_o, target_hidden)
                        max_target_Q = torch.max(target_Q)
                        y = r + discount*max_target_Q
                curr_o = curr_record.observation
                u = curr_record.action
                Q, hidden = adversary_net(curr_o, hidden)
                Q_u = Q[u]
                adversary_losses.append(criterion(y, Q_u))

    show_norms = False
    adversary_loss = torch.mean(torch.stack(adversary_losses))
    adversary_loss.backward()

    if show_norms: # for debugging
        norms = [p.grad.detach().data.norm().item() for p in adversary_net.parameters()]
        print("norm of gradiants", *np.round(norms, 2))

    torch.nn.utils.clip_grad_norm_(adversary_net.parameters(), config.clip_grad_norm)
    adversary_optimizer.step()
    
    return AttrDict(
        adversary=[loss.item() for loss in adversary_losses]
    )

def evaluate_agents(config, savedir, episode_idx, adversary_net):
    videodir = os.path.join(savedir, "videos")
    save_agent(os.path.join(savedir, f"adversary-net-{episode_idx}.pth"), adversary_net)
    adversary_net.eval()
    episodic_rewards=AttrDict(adversary=[])
    with torch.no_grad():
        for e in range(config.n_eval_episodes):
            save_video = e % 10 == 0
            validation_save_path = None
            should_render = False
            if save_video:
                validation_save_dir = os.path.join(videodir, f"epoch{episode_idx}")
                os.makedirs(validation_save_dir, exist_ok=True)
                validation_save_path = os.path.join(validation_save_dir, f"eval{e}.mp4")
                should_render = config.visualize_on_evaluation
            episode = run_episode(
                config, adversary_net,
                should_render=should_render, save_video=save_video,
                save_video_path=validation_save_path, is_val=True
            )
            episodic_rewards.adversary.append(episode.reward.adversary)
    avg_adversary_rewards = statistics.fmean(episodic_rewards.adversary)
    print(f"Average evaluation reward at episode {episode_idx} is: ")
    print(f"    adversary {avg_adversary_rewards:.2f}")
    adversary_net.train()
    return episodic_rewards

def train(config, normalizer=None):
    """
    - Use parameter sharing between agents of the same class.
    - Good agents use one RL model, adversaries use another RL model.
      Train the agents side by side.
    - Separate, disjoint communication channels for two classes of agents,
      maintained by a container to store the messages.
    """
    datestamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())
    savedir = os.path.join("models", config.exp_name, datestamp)
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "config.json"), "w") as f:
         json.dump(config, f)
        
    print("Training the agents...")
    t0 = time.time()
    device = torch.device(config.device_id)
    adversary_net = SimpleTagNet(config, "adversary", normalizer=normalizer).to(device)
    adversary_target_net = SimpleTagNet(config, "adversary", normalizer=normalizer).to(device)
    adversary_target_net.eval()
    print("Created the agent nets.")
    adversary_optimizer = torch.optim.SGD(adversary_net.parameters(), lr=config.lr)
    logger = AttrDict(
        episodic_losses=AttrDict(adversary=[], agent=[]),
        episodic_rewards=AttrDict(adversary=[], agent=[]),
        eval_episodic_rewards=[]
    )
    
    print("Initial update of target nets")
    def update_targets():
        adversary_target_net.load_state_dict(adversary_net.state_dict())
    update_targets()
    
    epsilon = config.eps_start
    batch = []
    print("Beginning the episodes...")
    for episode_idx in range(1, config.n_episodes + 1):
        # Run an episode
        episode = run_episode(
            config, adversary_net, epsilon=epsilon,
            should_render=episode_idx % config.report_interval == 0
        )
        batch.append(episode)
        
        # update epsilon at the end of each episode
        epsilon = max(epsilon*config.eps_decay, config.eps_end)
        
        # Train on the episode
        if episode_idx % config.batch_size == 0:
            episodic_losses = train_agents(
                config, device, batch, adversary_net,
                adversary_target_net, adversary_optimizer
            )
            logger.episodic_losses.adversary.extend(episodic_losses.adversary)
            batch = []
        
        # Logging
        logger.episodic_rewards.adversary.append(episode.reward.adversary)
        logger.episodic_rewards.agent.append(episode.reward.agent)

        if episode_idx % config.update_target_interval == 0:
            # Update double network
            update_targets()
        
        if episode_idx % config.report_interval == 0:
            # Logging
            t1 = time.time()
            tdelta = TimeDelta(round(t1 - t0, 0))
            print(f"on episode {episode_idx}, curr epsilon {epsilon:.2f} (time taken so far: {tdelta})")
            mean_loss_adversary = statistics.fmean(logger.episodic_losses.adversary[-config.report_interval:])
            mean_reward_adversary = statistics.fmean(logger.episodic_rewards.adversary[-config.report_interval:])
            mean_reward_agent = statistics.fmean(logger.episodic_rewards.agent[-config.report_interval:])
            print(f"     mean loss: adversary {mean_loss_adversary:.5f}")
            print(f"     mean reward: adversary {mean_reward_adversary:.2f}, agent {mean_reward_agent:.2f}")

        if episode_idx % config.evaluation_interval == 0:
            eval_episodic_rewards = evaluate_agents(
                config, savedir, episode_idx, adversary_net
            )
            logger.eval_episodic_rewards.append(eval_episodic_rewards)
    
    save_agent(os.path.join(savedir, "adversary-net-latest.pth"), adversary_net)
    with open(os.path.join(savedir, "log.json"), "w") as f:
         json.dump(logger, f)
    plot_training_run(savedir, logger)
    return adversary_net, logger

if __name__ == "__main__":
    env, config = make_args()
    normalizer = Normalizer(env) # norm_obs = normalize(obs)
    shapereward = RewardsShaper(env) # reward = shapereward(agent_name, obs)
    criterion = torch.nn.MSELoss()
    adversary_net, logger = train(config, normalizer)
