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

from common import (AttrDict, TimeDelta, Normalizer, POMDPNormalizer, RewardsShaper, Container,
        get_agent_counts, get_landmark_count, process_config,
        pad_amt, pad_image, moving_average, pad_values_front)
# from baseline import SimpleTagNet, choose_action, save_agent
from commnet import CommNet, choose_action, choose_message, save_agent

torch.autograd.set_detect_anomaly(True)
pp = pprint.PrettyPrinter(indent=4)

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
    parser.add_argument("--disable-rnn", action='store_false', dest="enable_rnn", help="to disable RNN (ablation)")
    parser.add_argument(
        "--disable-messaging", action='store_false', dest="enable_messaging",
        help="to disable message passing (ablation)"
    )
    parser.add_argument(
        "--disable-hiding", action='store_false', dest="enable_pomdp",
        help="to disable POMDP and make other agents fully visible"
    )

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
        enable_rnn=config.enable_rnn,
        message_size=2,
        n_rnn_layers=1,
        n_actions=env.action_space(env.agent_selection).n,
    )
    adversary_enable_messaging = config.enable_messaging and config.n_adversaries > 1
    config.adversary=AttrDict(
        n_agents=config.n_adversaries,
        observation_shape=env.observation_space("adversary_0").shape,
        enable_messaging=adversary_enable_messaging,
    )
    agent_enable_messaging = config.enable_messaging and config.n_good_agents > 1
    config.agent=AttrDict(
        n_agents=config.n_good_agents,
        observation_shape=env.observation_space("agent_0").shape,
        enable_messaging=agent_enable_messaging,
    )
    model_tag = "_rnn" if config.enable_rnn else "_mlp"
    config.exp_name = (f"rail{model_tag}_nadversaries{config.n_adversaries}"
                    f"_ngoodagents{config.n_good_agents}"
                    f"_landmarks{config.n_landmarks}")
    process_config(config)
    print("Using config:")
    pp.pprint(config)
    return env, config


def plot_training_run(savedir, logger):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.ravel()

    axes[0].plot(logger.episodic_losses.adversary, label="adversary")
    axes[0].set_title("adversary loss")
    axes[0].legend()

    axes[1].plot(logger.episodic_losses.agent, label="good agent")
    axes[1].set_title("good agent loss")
    axes[1].legend()

    axes[2].plot(logger.episodic_rewards.adversary, label="adversary")
    axes[2].plot(logger.episodic_rewards.agent, label="good agent")
    axes[2].set_title("reward")
    axes[2].legend()

    _moving_average = lambda a, n=64: pad_values_front(moving_average(a, n=n), n=n)
    adversary_episodic_rewards = np.array(logger["episodic_rewards"]["adversary"])
    agent_episodic_rewards = np.array(logger["episodic_rewards"]["agent"])
    avg_adversary_episodic_rewards = _moving_average(adversary_episodic_rewards)
    avg_agent_episodic_rewards = _moving_average(agent_episodic_rewards)

    axes[3].plot(avg_adversary_episodic_rewards, label="adversary mean")
    axes[3].plot(avg_agent_episodic_rewards, label="agent mean")
    axes[3].set_title("reward")
    axes[3].legend()
    fig.savefig(os.path.join(savedir, "training_run.png"))


def run_episode(
    config, container, adversary_net, agent_net, epsilon=0.05,
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
                print("message index, payload", m_idx, m_up)
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


def train_agents(config, device, container, batch, adversary_net, agent_net,
                 adversary_target_net, agent_target_net,
                 adversary_optimizer, agent_optimizer):
    """Compute loss of episode and update agent weights."""
    adversary_optimizer.zero_grad()
    agent_optimizer.zero_grad()
    discount = torch.tensor(config.discount, dtype=torch.float, device=device)
    adversary_losses = []
    agent_losses = []
    
    for episode in batch:
        container.reset()
        for step_idx in range(episode.steps):
            # Optimize networks
            for agent_type in episode.step_records[step_idx].keys():
                for agent_idx in episode.step_records[step_idx][agent_type].keys():
                    curr_record = episode.step_records[step_idx][agent_type][agent_idx]
                    if curr_record.done:
                        # agent is done at this step
                        continue
                    next_record = episode.step_records[step_idx + 1][agent_type][agent_idx]
                    r = torch.tensor(next_record.reward, dtype=torch.float, device=device)
                    
                    # compute targets
                    y_u = None
                    y_m = None
                    if next_record.done:
                        # agent terminates at next step
                        y_u = r
                        y_m = r
                    else:
                        with torch.no_grad():
                            if config[agent_type].enable_rnn:
                                target_hidden =  container.get_target_hidden(
                                    f"{agent_type}_{agent_idx}"
                                )
                            else:
                                target_hidden = None

                            next_o = next_record.observation
                            next_m = next_record.message
                            if agent_type == "agent":
                                target_Q_u, target_Q_m, target_hidden = agent_target_net(
                                    next_o, target_hidden, next_m, f"{agent_type}_{agent_idx}"
                                )
                            else:
                                # agent_type == "adversary"
                                target_Q_u, target_Q_m, target_hidden = adversary_target_net(
                                    next_o, target_hidden, next_m, f"{agent_type}_{agent_idx}"
                                )

                            if config[agent_type].enable_rnn:
                                container.update_target_hidden(
                                    f"{agent_type}_{agent_idx}", target_hidden
                                )
                            
                            max_target_Q_u = torch.max(target_Q_u)
                            y_u = r + discount*max_target_Q_u

                            if config[agent_type].enable_messaging:
                                max_target_Q_m = torch.max(target_Q_m)
                                y_m = r + discount*max_target_Q_m
                    
                    if config[agent_type].enable_rnn:    
                        hidden =  container.get_hidden(f"{agent_type}_{agent_idx}")
                    else:
                        hidden = None
                    
                    curr_o = curr_record.observation
                    curr_m = curr_record.message
                    u = curr_record.action
                    if agent_type == "agent":
                        Q_u, Q_m, hidden = agent_net(curr_o, hidden, curr_m, f"{agent_type}_{agent_idx}")
                    else:
                        # agent_type == "adversary"
                        Q_u, Q_m, hidden = adversary_net(curr_o, hidden, curr_m, f"{agent_type}_{agent_idx}")
                    
                    if config[agent_type].enable_rnn:
                        container.update_hidden(f"{agent_type}_{agent_idx}", hidden)

                    # compute TD errors
                    Q_u = Q_u[u]
                    td_u = criterion(y_u, Q_u)
                    td_error = None
                    if config[agent_type].enable_messaging:
                        m_u = curr_record.msg_action
                        Q_m = Q_m[m_u]
                        td_m = criterion(y_m, Q_m)
                        td_error = td_u + td_m
                    else:
                        td_error = td_u
                    
                    if agent_type == "agent":
                        agent_losses.append(td_error)
                    else:
                        # agent_type == "adversary"
                        adversary_losses.append(td_error)
        
    show_norms = False
    adversary_loss = torch.mean(torch.stack(adversary_losses))
    adversary_loss.backward()
    
    if show_norms: # for debugging
        norms = [p.grad.detach().data.norm().item() for p in adversary_net.parameters()]
        print("norm of gradiants", *np.round(norms, 2))

    torch.nn.utils.clip_grad_norm_(adversary_net.parameters(), config.clip_grad_norm)
    adversary_optimizer.step()
    
    agent_loss = torch.mean(torch.stack(agent_losses))
    agent_loss.backward()
    if show_norms: # for debugging
        norms = [p.grad.detach().data.norm().item() for p in agent_net.parameters()]
        print("norm of gradiants", *np.round(norms, 2))

    torch.nn.utils.clip_grad_norm_(agent_net.parameters(), config.clip_grad_norm)
    agent_optimizer.step()
    
    return AttrDict(
        adversary=[loss.item() for loss in adversary_losses],
        agent=[loss.item() for loss in agent_losses]
    )


def evaluate_agents(config, container, savedir, episode_idx, adversary_net, agent_net):
    videodir = os.path.join(savedir, "videos")
    save_agent(os.path.join(savedir, f"adversary-net-{episode_idx}.pth"), adversary_net)
    save_agent(os.path.join(savedir, f"agent-net-{episode_idx}.pth"), agent_net)
    adversary_net.eval()
    agent_net.eval()
    episodic_rewards=AttrDict(adversary=[], agent=[])
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
                config, container, adversary_net, agent_net,
                should_render=should_render, save_video=save_video,
                save_video_path=validation_save_path, is_val=True
            )
            episodic_rewards.adversary.append(episode.reward.adversary)
            episodic_rewards.agent.append(episode.reward.agent)
    avg_adversary_rewards = statistics.fmean(episodic_rewards.adversary)
    avg_agent_rewards = statistics.fmean(episodic_rewards.agent)
    print(f"Average evaluation reward at episode {episode_idx} is: ")
    print(f"    adversary {avg_adversary_rewards:.2f}, agent {avg_agent_rewards:.2f}")
    adversary_net.train()
    agent_net.train()
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
    adversary_net = CommNet(config, "adversary", normalizer=normalizer).to(device)
    agent_net = CommNet(config, "agent", normalizer=normalizer).to(device)
    adversary_target_net = CommNet(config, "adversary", normalizer=normalizer).to(device)
    agent_target_net = CommNet(config, "agent", normalizer=normalizer).to(device)
    adversary_target_net.eval()
    agent_target_net.eval()
    print("Created the agent nets.")
    adversary_optimizer = torch.optim.SGD(adversary_net.parameters(), lr=config.lr)
    agent_optimizer = torch.optim.SGD(agent_net.parameters(), lr=config.lr)
    container = Container(config)
    logger = AttrDict(
        episodic_losses=AttrDict(adversary=[], agent=[]),
        episodic_rewards=AttrDict(adversary=[], agent=[]),
        eval_episodic_rewards=[]
    )
    
    print("Initial update of target nets")
    def update_targets():
        adversary_target_net.load_state_dict(adversary_net.state_dict())
        agent_target_net.load_state_dict(agent_net.state_dict())
    update_targets()
    
    epsilon = config.eps_start
    batch = []
    print("Beginning the episodes...")
    for episode_idx in range(1, config.n_episodes + 1):
        # Run an episode
        should_render = config.visualize and episode_idx % config.visualize_interval == 0
        config.visualize
        episode = run_episode(
            config, container, adversary_net, agent_net,
            epsilon=epsilon, should_render=should_render
        )
        batch.append(episode)
        
        # update epsilon at the end of each episode
        epsilon = max(epsilon*config.eps_decay, config.eps_end)
        
        # Train on the episode
        if episode_idx % config.batch_size == 0:
            episodic_losses = train_agents(
                config, device, container, batch, adversary_net, agent_net,
                adversary_target_net, agent_target_net, adversary_optimizer, agent_optimizer
            )
            logger.episodic_losses.adversary.extend(episodic_losses.adversary)
            logger.episodic_losses.agent.extend(episodic_losses.agent)
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
            print(f"on episode {episode_idx}, curr epsilon {epsilon} (time taken so far: {tdelta})")
            mean_loss_adversary = statistics.fmean(logger.episodic_losses.adversary[-config.report_interval:])
            mean_loss_agent = statistics.fmean(logger.episodic_losses.agent[-config.report_interval:])
            mean_reward_adversary = statistics.fmean(logger.episodic_rewards.adversary[-config.report_interval:])
            mean_reward_agent = statistics.fmean(logger.episodic_rewards.agent[-config.report_interval:])
            print(f"     mean loss: adversary {mean_loss_adversary:.5f}, agent {mean_loss_agent:.5f}")
            print(f"     mean reward: adversary {mean_reward_adversary:.2f}, agent {mean_reward_agent:.2f}")
        
        if episode_idx % config.evaluation_interval == 0:
            eval_episodic_rewards = evaluate_agents(
                config, container, savedir, episode_idx, adversary_net, agent_net
            )
            logger.eval_episodic_rewards.append(eval_episodic_rewards)
    
    save_agent(os.path.join(savedir, "adversary-net-latest.pth"), adversary_net)
    save_agent(os.path.join(savedir, "agent-net-latest.pth"), agent_net)
    with open(os.path.join(savedir, "log.json"), "w") as f:
         json.dump(logger, f)
    plot_training_run(savedir, logger)
    return adversary_net, agent_net, logger

if __name__ == "__main__":
    env, config = make_args()
    if config.enable_pomdp:
        normalizer = POMDPNormalizer(env)
    else:
        normalizer = Normalizer(env) # norm_obs = normalize(obs)
    shapereward = RewardsShaper(env) # reward = shapereward(agent_name, obs)
    criterion = torch.nn.MSELoss()
    adversary_net, agent_net, logger = train(config, normalizer)
