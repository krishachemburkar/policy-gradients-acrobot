import gymnasium as gym
import torch
import matplotlib.pyplot as plt


def get_discounted_rewards(rewards, gamma):
    returns = [0 for _ in range(len(rewards))]
    returns[-1] = rewards[-1]

    for i in range(len(rewards) - 2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i + 1]

    return returns


def normalize(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    return (x - x.mean()) / (x.std() + 1e-8)


def evaluate_policy(policy, env_name, n_eval_episodes=5):
    eval_env = gym.make(env_name)

    policy.eval()
    episode_returns = []

    with torch.no_grad():
        for _ in range(n_eval_episodes):
            state, info = eval_env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()

                next_state, reward, terminated, truncated, info = eval_env.step(action)

                total_reward += reward
                state = next_state
                done = terminated or truncated

            episode_returns.append(total_reward)

    eval_env.close()
    policy.train()

    return sum(episode_returns) / len(episode_returns)


def log_training(
    log_file,
    episode,
    episode_return,
    train_avg,
    avg_eval_return,
    actor_loss=None,
    critic_loss=None
):
    log_line = (
        f"Episode: {episode}, "
        f"Train Return: {episode_return:.2f}, "
        f"Train Avg Return: {train_avg:.2f}, "
        f"Eval Avg Return: {avg_eval_return}, "
        f"Actor Loss: {actor_loss}, "
        f"Critic Loss: {critic_loss}\n"
    )

    print(log_line.strip())
    log_file.write(log_line)
    log_file.flush()


def plot_returns(
    train_avg_returns,
    eval_episodes,
    eval_rewards,
    n_episode,
    save_path
):
    plt.figure()

    plt.plot(
        range(1, n_episode + 1),
        train_avg_returns,
        label="Train Avg Return"
    )

    plt.plot(
        eval_episodes,
        eval_rewards,
        label="Eval Avg Return"
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("Train vs Eval Return")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.show()