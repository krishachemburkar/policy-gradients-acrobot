import torch
import gymnasium as gym

from torch.distributions import Categorical

from args import args
from network import PolicyNetwork
from gymnasium.wrappers import RecordVideo 


def run_inference():

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
        hidden_size=args.policy_hidden
    )

    checkpoint = torch.load(
        args.model_path,
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:
        policy.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()

    for episode in range(args.inference_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            with torch.no_grad():
                probs = policy(state_tensor)

                if args.sample_action:
                    dist = Categorical(probs)
                    action = dist.sample().item()
                else:
                    action = torch.argmax(probs).item()

            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state
            done = terminated or truncated

        print(
            f"Inference Episode {episode + 1}, "
            f"Total Reward: {total_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    run_inference()