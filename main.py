import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import numpy as np

from torch.distributions import Categorical
from torch.optim import Adam

from args import args
from network import PolicyNetwork, Baseline

from utils import (
    get_discounted_rewards,
    evaluate_policy,
    log_training,
    plot_returns
)

from algorithms.reinforce import train_reinforce
from algorithms.mc_actor_critic import train_mc_actor_critic


algo = args.algo
n_episode = args.episodes
env_name = args.env
gamma = args.gamma

actor_lr = args.actor_lr
critic_lr = args.critic_lr

policy_hidden = args.policy_hidden
critic_hidden = args.nn_baseline_hidden
eval_freq = args.eval_freq


train_returns = []
train_avg_returns = []

eval_rewards = []
eval_episodes = []

log_file = open("training_log.txt", "w")


env = gym.make(
    env_name,
    render_mode="rgb_array"
)

env = RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda episode_id: episode_id % 500 == 0 or episode_id == n_episode - 1
)


policy = PolicyNetwork(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_size=policy_hidden
)

policy_optimizer = Adam(
    policy.parameters(),
    lr=actor_lr
)


baseline = None
baseline_optimizer = None
baseline_criterion = None

if algo in ["mc_ac", "td_ac", "ppo"]:
    baseline = Baseline(
        state_dim=env.observation_space.shape[0],
        hidden_size=critic_hidden
    )

    baseline_optimizer = Adam(
        baseline.parameters(),
        lr=critic_lr
    )

    baseline_criterion = torch.nn.MSELoss()


for episode in range(n_episode):

    states = []
    actions = []
    rewards = []
    log_probs = []

    state, info = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)

        probs = policy(state_tensor)
        distribution = Categorical(probs)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        next_state, reward, terminated, truncated, info = env.step(action.item())

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state
        done = terminated or truncated

    returns = get_discounted_rewards(
        rewards,
        gamma
    )

    if algo == "reinforce":
        train_reinforce(
            policy_optimizer,
            log_probs,
            returns
        )

    elif algo == "mc_ac":
        train_mc_actor_critic(
            baseline,
            baseline_optimizer,
            baseline_criterion,
            policy_optimizer,
            states,
            returns,
            log_probs
        )

    else:
        raise ValueError(f"Unknown or unimplemented algo: {algo}")

    episode_return = sum(rewards)
    train_returns.append(episode_return)

    train_avg = np.mean(train_returns[-50:])
    train_avg_returns.append(train_avg)

    avg_eval_return = None

    if (episode + 1) % eval_freq == 0:
        avg_eval_return = evaluate_policy(
            policy,
            env_name,
            n_eval_episodes=5
        )

        eval_rewards.append(avg_eval_return)
        eval_episodes.append(episode + 1)

    log_training(
        log_file=log_file,
        episode=episode + 1,
        episode_return=episode_return,
        train_avg=train_avg,
        avg_eval_return=avg_eval_return
    )


env.close()
log_file.close()


torch.save(
    policy.state_dict(),
    "policy_model.pth"
)

if baseline is not None:
    torch.save(
        baseline.state_dict(),
        "baseline_model.pth"
    )


plot_returns(
    train_avg_returns=train_avg_returns,
    eval_episodes=eval_episodes,
    eval_rewards=eval_rewards,
    n_episode=n_episode
)