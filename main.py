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
from algorithms.td_actor_critic import train_td_actor_critic

import os

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

actor_losses = []
critic_losses = []

best_eval_return = -float("inf")

log_file = open(args.log_file, "w")

os.makedirs(
    args.video_folder,
    exist_ok=True
)

if args.save_video:

    env = gym.make(
        args.env,
        render_mode="rgb_array"
    )

    env = RecordVideo(
        env,
        video_folder=args.video_folder,
        episode_trigger=lambda episode_id: True
    )

else:

    env = gym.make(
        args.env,
        render_mode="human"
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

        current_done = terminated or truncated

        if algo == "td_ac":
            actor_loss_value, critic_loss_value = train_td_actor_critic(
                baseline=baseline,
                baseline_optimizer=baseline_optimizer,
                baseline_criterion=baseline_criterion,
                optimizer=policy_optimizer,
                log_prob=log_prob,
                state=state,
                reward=reward,
                next_state=next_state,
                done=current_done,
                gamma=gamma
            )

            actor_losses.append(actor_loss_value)
            critic_losses.append(critic_loss_value)

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state
        done = current_done

    if algo == "reinforce":
        returns = get_discounted_rewards(
            rewards,
            gamma
        )

        loss_value = train_reinforce(
            policy_optimizer,
            log_probs,
            returns
        )

        actor_losses.append(loss_value)

    elif algo == "mc_ac":
        returns = get_discounted_rewards(
            rewards,
            gamma
        )

        actor_loss_value, critic_loss_value = train_mc_actor_critic(
            baseline,
            baseline_optimizer,
            baseline_criterion,
            policy_optimizer,
            states,
            returns,
            log_probs
        )

        actor_losses.append(actor_loss_value)
        critic_losses.append(critic_loss_value)

    elif algo == "td_ac":
        pass

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

        if avg_eval_return > best_eval_return:
            best_eval_return = avg_eval_return

            torch.save(
                policy.state_dict(),
                f"{args.model_name}_best_policy.pth"
            )

            if baseline is not None:
                torch.save(
                    baseline.state_dict(),
                    f"{args.model_name}_best_critic.pth"
                )

            print(f"New best eval return: {best_eval_return:.2f}. Model saved.")

    avg_actor_loss = np.mean(actor_losses[-100:]) if len(actor_losses) > 0 else None
    avg_critic_loss = np.mean(critic_losses[-100:]) if len(critic_losses) > 0 else None

    log_training(
        log_file=log_file,
        episode=episode + 1,
        episode_return=episode_return,
        train_avg=train_avg,
        avg_eval_return=avg_eval_return,
        actor_loss=avg_actor_loss,
        critic_loss=avg_critic_loss
    )


env.close()
log_file.close()


torch.save(
    policy.state_dict(),
    f"{args.model_name}_final_policy.pth"
)

if baseline is not None:
    torch.save(
        baseline.state_dict(),
        f"{args.model_name}_final_critic.pth"
    )


plot_returns(
    train_avg_returns=train_avg_returns,
    eval_episodes=eval_episodes,
    eval_rewards=eval_rewards,
    n_episode=n_episode,
    save_path=args.plot_file
)