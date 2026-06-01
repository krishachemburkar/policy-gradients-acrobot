import torch
from utils import normalize

def train_reinforce(
    optimizer,
    log_probs,
    returns
):
    returns = torch.tensor(
        returns,
        dtype=torch.float32
    )

    returns = normalize(returns)

    policy_loss = 0

    for log_prob, ret in zip(
        log_probs,
        returns
    ):
        policy_loss += -log_prob * ret

    optimizer.zero_grad()

    policy_loss.backward()

    optimizer.step()

    return policy_loss.item()