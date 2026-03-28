import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, A2C, PPO
from environment.lifeloop_env import WildlifeLoopEnv


MODELS = {
    "DQN": ("models/dqn/wildlife_dqn_final.zip", DQN),
    "A2C": ("models/a2c/wildlife_a2c_final.zip", A2C),
    "PPO": ("models/ppo/wildlife_ppo_final.zip", PPO),
}

EPISODES = 10


def evaluate_model(model, env, name):
    rewards = []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"{name} Episode {ep+1}: Reward = {total_reward}")

    return rewards


def main():
    results = {}

    for name, (path, algo) in MODELS.items():
        print(f"\n=== Loading {name} ===")

        if not os.path.exists(path):
            print(f"❌ Model not found: {path}")
            continue

        env = WildlifeLoopEnv()
        model = algo.load(path)

        rewards = evaluate_model(model, env, name)
        results[name] = rewards

        env.close()

    plt.figure()

    for name, rewards in results.items():
        plt.plot(rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Model Comparison")
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison.png")
    plt.show()

    print("\n Plot saved to results/comparison.png")


if __name__ == "__main__":
    main()