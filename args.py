import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--algo",
    type=str,
    default="reinforce",
    choices=[
        "reinforce",
        "mc_ac",
        "td_ac",
        "ppo"
    ]
)

parser.add_argument(
    "--episodes",
    type=int,
    default=800
)

parser.add_argument(
    "--env",
    type=str,
    default="LunarLander-v3"
)

parser.add_argument(
    "--nn_baseline_hidden",
    type=int,
    default=128
)

parser.add_argument(
    "--policy_hidden",
    type=int,
    default=128
)

parser.add_argument(
    "--critic_lr",
    type=float,
    default=1e-3
)

parser.add_argument(
    "--actor_lr",
    type=float,
    default=3e-4
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.99
)

parser.add_argument(
    "--eval_freq",
    type=int,
    default=50
)

args = parser.parse_args()