import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--algo",
    type=str,
    default="reinforce",
    choices=["reinforce", "mc_ac", "td_ac", "ppo"]
)

parser.add_argument("--episodes", type=int, default=800)
parser.add_argument("--env", type=str, default="LunarLander-v3")
parser.add_argument("--nn_baseline_hidden", type=int, default=128)
parser.add_argument("--policy_hidden", type=int, default=128)
parser.add_argument("--critic_lr", type=float, default=5e-4)
parser.add_argument("--actor_lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--eval_freq", type=int, default=50)

parser.add_argument(
    "--log_file",
    type=str,
    default="training_log.txt"
)

parser.add_argument(
    "--model_name",
    type=str,
    default="model"
)

parser.add_argument(
    "--plot_file",
    type=str,
    default="plots/training_curve.png"
)

parser.add_argument(
    "--video_folder",
    type=str,
    default="videos"
)


parser.add_argument(
    "--model_path",
    type=str,
    default="models/td_ac_lander_best_policy.pth"
)

parser.add_argument(
    "--inference_episodes",
    type=int,
    default=5
)

parser.add_argument(
    "--sample_action",
    action="store_true"
)

parser.add_argument(
    "--save_video",
    action="store_true",
    help="Record videos during training/inference"
)

args = parser.parse_args()