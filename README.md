# Policy Gradient using REINFORCE on CartPole

## Overview

This project implements a basic **Policy Gradient Reinforcement Learning algorithm (REINFORCE)** using PyTorch and Gymnasium on the CartPole-v1 environment.

The agent learns to balance a pole on a moving cart by directly learning a policy that maps states to action probabilities.

---

# Environment

## CartPole-v1

The environment consists of:
- A cart moving left/right
- A pole attached to the cart

Goal:
- Keep the pole balanced upright for as long as possible

---

# State Space

The environment provides a 4-dimensional state:

$$
[x,\dot{x},\theta,\dot{\theta}]
$$

Where:

| Variable | Meaning |
|---|---|
| $x$ | Cart Position |
| $\dot{x}$ | Cart Velocity |
| $\theta$ | Pole Angle |
| $\dot{\theta}$ | Pole Angular Velocity |

---

# Action Space

The agent can take 2 discrete actions:

| Action | Meaning |
|---|---|
| 0 | Push Cart Left |
| 1 | Push Cart Right |

---

# Reward

The environment provides:

$$
+1
$$

reward for every timestep the pole remains balanced.

Thus:

$$
\text{Total Reward} = \text{Episode Length}
$$

---

# Results of Trajectory

# Episode 0

![Episode 0](videos/rl-video-episode-0.gif)

# Episode 400

![Episode 400](videos/rl-video-episode-400.gif)

# Episode 800

![Episode 800](videos/rl-video-episode-800.gif)

---

# Policy Gradient Idea

Instead of learning Q-values, the policy network directly learns:

$$
\pi_\theta(a|s)
$$

which represents:
- probability of taking action $a$
- given state $s$

---

# Policy Network

The neural network:
- takes state as input
- outputs action probabilities

Architecture:

```python
Input: 4
Hidden Layer: 128
Output: 2
Activation: ReLU + Softmax
```

Example output:

```python
[0.3, 0.7]
```

Meaning:
- 30% probability of moving left
- 70% probability of moving right

---

# Action Sampling

Actions are sampled using a categorical probability distribution.

```python
distribution = Categorical(probs)
action = distribution.sample()
```

This allows:
- stochastic exploration
- probabilistic behavior

---

# Discounted Returns

For each timestep:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...
$$

Where:
- $G_t$ = discounted future reward
- $\gamma$ = discount factor

This gives more importance to:
- actions that helped long-term survival

---

# Return Computation

Returns are computed backward using:

$$
G_t = r_t + \gamma G_{t+1}
$$

---

# REINFORCE Loss

The loss function used is:

$$
L = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

Where:
- $\log \pi_\theta(a_t|s_t)$ = log probability of chosen action
- $G_t$ = discounted return

Intuition:
- good actions become more probable
- bad actions become less reinforced

---

# Training Process

For every episode:

1. Reset environment
2. Run episode using current policy
3. Store:
   - states
   - actions
   - rewards
   - log probabilities
4. Compute discounted returns
5. Normalize returns
6. Compute policy gradient loss
7. Backpropagate loss
8. Update policy network

---

# Optimization

Optimizer used:

```python
Adam
```

Learning Rate:

```python
0.001
```

---

# Video Recording

Training videos are recorded every 100 episodes using:

```python
RecordVideo
```

Videos are stored in:

```text
videos/
```

---

# Libraries Used

- PyTorch
- Gymnasium
- NumPy

---

# Conclusion

This project implements a complete vanilla Policy Gradient pipeline using REINFORCE on CartPole.

The agent learns entirely through:
- interaction with the environment
- reward signals
- policy optimization

without any supervised labels or expert trajectories.