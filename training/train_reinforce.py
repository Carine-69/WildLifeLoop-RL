"""
training/train_reinforce.py
============================
REINFORCE (Monte-Carlo Policy Gradient) for WildlifeLoopEnv.

SB3 does not ship REINFORCE, so this is a clean from-scratch
PyTorch implementation that saves checkpoints in the same format
as the other algorithms (models/reinforce/) so all evaluation
and plotting scripts work unchanged.

Usage:
    python training/train_reinforce.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.lifeloop_env import WildlifeLoopEnv

# ─── HYPERPARAMETERS (default / best run) ─────────────────────────────────────
CONFIG = {
    "total_episodes":    5_000,      # total training episodes
    "gamma":             0.99,       # discount factor
    "learning_rate":     3e-4,       # Adam LR
    "hidden_size":       128,        # neurons per hidden layer
    "entropy_coef":      0.01,       # entropy bonus (exploration)
    "max_grad_norm":     0.5,        # gradient clipping
    "save_every":        500,        # save checkpoint every N episodes
    "model_dir":         "models/reinforce",
    "log_every":         100,        # print progress every N episodes
    "seed":              42,
}
# ──────────────────────────────────────────────────────────────────────────────


class PolicyNetwork(nn.Module):
    """Simple 2-layer MLP policy for discrete action space."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def act(self, obs: np.ndarray):
        """Sample action and return (action, log_prob)."""
        x    = torch.FloatTensor(obs).unsqueeze(0)
        probs = self(x)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    """Discounted Monte-Carlo returns, normalised for stability."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    # normalise
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train(cfg: dict = CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    os.makedirs(cfg["model_dir"], exist_ok=True)

    env     = WildlifeLoopEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]   # 22 (updated env)
    act_dim = env.action_space.n               # 7

    policy    = PolicyNetwork(obs_dim, act_dim, cfg["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    episode_rewards = []
    best_mean       = -np.inf
    best_ep         = 0

    print("=" * 60)
    print("REINFORCE Training — WildlifeLoopEnv")
    print("=" * 60)
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"hidden={cfg['hidden_size']}  lr={cfg['learning_rate']}")
    print(f"  gamma={cfg['gamma']}  entropy_coef={cfg['entropy_coef']}")
    print(f"  total_episodes={cfg['total_episodes']:,}")
    print("=" * 60)

    for ep in range(1, cfg["total_episodes"] + 1):
        obs, _       = env.reset()
        done         = False
        truncated    = False
        log_probs    = []
        entropies    = []
        rewards      = []
        total_reward = 0.0

        # ── collect one full episode ──────────────────────────────────────────
        while not done and not truncated:
            action, log_prob, entropy = policy.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            total_reward += reward

        episode_rewards.append(total_reward)

        # ── compute loss and update ───────────────────────────────────────────
        returns   = compute_returns(rewards, cfg["gamma"])
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        policy_loss  = -(log_probs * returns).mean()
        entropy_loss = -entropies.mean()
        loss         = policy_loss + cfg["entropy_coef"] * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
        optimizer.step()

        # ── logging ───────────────────────────────────────────────────────────
        if ep % cfg["log_every"] == 0:
            recent = episode_rewards[-cfg["log_every"]:]
            mean_r = np.mean(recent)
            print(f"  ep={ep:>5,}  mean_reward={mean_r:>8.2f}  "
                  f"loss={loss.item():>7.4f}  "
                  f"ep_reward={total_reward:>8.2f}")

        # ── checkpointing ─────────────────────────────────────────────────────
        if ep % cfg["save_every"] == 0:
            ckpt = {
                "episode":          ep,
                "policy_state":     policy.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "config":           cfg,
                "episode_rewards":  episode_rewards,
            }
            path = os.path.join(cfg["model_dir"],
                                f"wildlife_reinforce_{ep}_episodes.pt")
            torch.save(ckpt, path)
            print(f"  ✔ Saved checkpoint → {path}")

        # ── track best ────────────────────────────────────────────────────────
        if len(episode_rewards) >= 100:
            m = np.mean(episode_rewards[-100:])
            if m > best_mean:
                best_mean = m
                best_ep   = ep

    # ── save final ────────────────────────────────────────────────────────────
    final_ckpt = {
        "episode":         cfg["total_episodes"],
        "policy_state":    policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":          cfg,
        "episode_rewards": episode_rewards,
    }
    final_path = os.path.join(cfg["model_dir"], "wildlife_reinforce_final.pt")
    torch.save(final_ckpt, final_path)

    env.close()

    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  Best 100-ep mean : {best_mean:.2f}  (around episode {best_ep:,})")
    print(f"  Final checkpoint : {final_path}")
    print("=" * 60)

    return policy, episode_rewards


if __name__ == "__main__":
    train()