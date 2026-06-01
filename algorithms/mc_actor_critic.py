import torch
from utils import normalize


def train_mc_actor_critic(
    baseline,
    baseline_optimizer,
    baseline_criterion,
    optimizer,
    states,
    returns,
    log_probs
):
    states_tensor = torch.tensor(
        states,
        dtype=torch.float32
    )

    returns_tensor = torch.tensor(
        returns,
        dtype=torch.float32
    ).unsqueeze(1)

    predicted_returns = baseline(
        states_tensor
    )

    critic_loss = baseline_criterion(
        predicted_returns,
        returns_tensor
    )

    baseline_optimizer.zero_grad()

    critic_loss.backward()

    baseline_optimizer.step()

    advantages = (
        returns_tensor
        - predicted_returns.detach()
    )

    advantages = normalize(
        advantages
    ).squeeze()

    actor_loss = 0

    for log_prob, advantage in zip(
        log_probs,
        advantages
    ):
        actor_loss += (
            -log_prob * advantage
        )

    optimizer.zero_grad()

    actor_loss.backward()

    optimizer.step()

    return (
        actor_loss.item(),
        critic_loss.item()
    )