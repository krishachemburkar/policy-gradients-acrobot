import torch


def train_td_actor_critic(
    baseline,
    baseline_optimizer,
    baseline_criterion,
    optimizer,
    log_prob,
    state,
    reward,
    next_state,
    done,
    gamma
):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

    value = baseline(state_tensor)

    with torch.no_grad():
        next_value = baseline(next_state_tensor)
        td_target = reward + gamma * next_value * (1 - int(done))

    td_error = td_target - value

    critic_loss = baseline_criterion(value, td_target)

    baseline_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
    baseline_optimizer.step()

    actor_loss = -log_prob * td_error.detach()

    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item()