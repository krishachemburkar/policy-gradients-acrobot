import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
from network import PolicyNetwork
from torch.distributions import Categorical

def get_discounted_rewards(rewards, discount_factor):
    returns = [0 for i in range(len(rewards))]
    returns[-1] = rewards[-1]

    for i in range(len(rewards) - 2, -1, -1):
        returns[i] = rewards[i] + discount_factor * returns[i + 1]

    return returns

def normalize_returns(returns):

    returns = torch.tensor(returns, dtype=torch.float32)

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns

visual_env = gym.make(
    "CartPole-v1",
    render_mode="rgb_array"
)

visual_env = RecordVideo(
    visual_env,
    video_folder="videos",
    episode_trigger=lambda episode_id: episode_id % 100 == 0
)

n_episode = 800

states_all = []
actions_all = []
rewards_all = []
returns_all = []
log_probs_all = []

discounting_factor = 0.9

policy = PolicyNetwork(
    state_dim=visual_env.observation_space.shape[0],
    action_dim=visual_env.action_space.n
)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

for n in range(n_episode):

    states = []
    actions = []
    rewards = []
    log_probs = []

    state, info = visual_env.reset()
    done = False

    while not done:

        state_tensor = torch.tensor(state, dtype=torch.float32)

        probs = policy(state_tensor)

        distribution = Categorical(probs)

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        next_state, reward, terminated, truncated, info = visual_env.step(action.item())

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state

        done = terminated or truncated

    returns = get_discounted_rewards(rewards, discounting_factor)

    returns_norm = normalize_returns(returns)
    loss = 0

    for log_prob, disc_return in zip(log_probs, returns_norm):
        loss += -log_prob * disc_return

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    states_all.append(states)
    actions_all.append(actions)
    rewards_all.append(rewards)
    returns_all.append(returns)
    log_probs_all.append(log_probs)

    print("Episode:", n + 1)
    print("Episode length:", len(rewards))
    print("Total reward:", sum(rewards))
    print("------------------")

visual_env.close()