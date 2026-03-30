"""
training/hyperparameter_tuning.py
===================================
Runs 10 hyperparameter configurations for each algorithm
(DQN, REINFORCE, PPO, A2C) and saves a results CSV + PNG table
per algorithm into results/hyperparameter_tuning/.

Each run trains for a fixed budget and evaluates the final policy
over EVAL_EPISODES episodes to measure mean reward.

Usage:
    python training/hyperparameter_tuning.py
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.lifeloop_env import WildlifeLoopEnv

# ─── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR        = "results/hyperparameter_tuning"
EVAL_EPISODES  = 5
TRAIN_STEPS    = 100_000    # per SB3 run  (lower = faster tuning)
REINFORCE_EPS  = 1_000      # episodes per REINFORCE run

COLORS = {"DQN": "#E94560", "PPO": "#0F9B8E", "A2C": "#F5A623",
          "REINFORCE": "#8B5CF6"}
# ──────────────────────────────────────────────────────────────────────────────

# ─── 10 HYPERPARAMETER GRIDS ──────────────────────────────────────────────────

DQN_CONFIGS = [
    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64,  "buffer_size": 50_000,  "exploration_fraction": 0.3,  "tau": 1.0,  "train_freq": 4},
    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 50_000,  "exploration_fraction": 0.2,  "tau": 1.0,  "train_freq": 4},
    {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 64,  "buffer_size": 100_000, "exploration_fraction": 0.2,  "tau": 1.0,  "train_freq": 4},
    {"learning_rate": 1e-3, "gamma": 0.95, "batch_size": 32,  "buffer_size": 50_000,  "exploration_fraction": 0.4,  "tau": 0.5,  "train_freq": 4},
    {"learning_rate": 5e-4, "gamma": 0.95, "batch_size": 128, "buffer_size": 100_000, "exploration_fraction": 0.1,  "tau": 1.0,  "train_freq": 8},
    {"learning_rate": 2e-4, "gamma": 0.99, "batch_size": 32,  "buffer_size": 10_000,  "exploration_fraction": 0.5,  "tau": 0.1,  "train_freq": 1},
    {"learning_rate": 1e-3, "gamma": 0.90, "batch_size": 64,  "buffer_size": 50_000,  "exploration_fraction": 0.3,  "tau": 0.5,  "train_freq": 4},
    {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 256, "buffer_size": 200_000, "exploration_fraction": 0.1,  "tau": 1.0,  "train_freq": 16},
    {"learning_rate": 1e-4, "gamma": 0.98, "batch_size": 64,  "buffer_size": 100_000, "exploration_fraction": 0.25, "tau": 0.8,  "train_freq": 4},
    {"learning_rate": 5e-3, "gamma": 0.99, "batch_size": 32,  "buffer_size": 50_000,  "exploration_fraction": 0.2,  "tau": 1.0,  "train_freq": 4},
]

PPO_CONFIGS = [
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.0,   "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.01,  "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64,  "n_epochs": 5,  "ent_coef": 0.01,  "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "gamma": 0.95, "n_steps": 512,  "batch_size": 32,  "n_epochs": 10, "ent_coef": 0.0,   "clip_range": 0.1, "gae_lambda": 0.90},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 2048, "batch_size": 128, "n_epochs": 20, "ent_coef": 0.0,   "clip_range": 0.3, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 4096, "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.05,  "clip_range": 0.2, "gae_lambda": 0.98},
    {"learning_rate": 2e-4, "gamma": 0.98, "n_steps": 1024, "batch_size": 32,  "n_epochs": 15, "ent_coef": 0.001, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 256, "n_epochs": 5,  "ent_coef": 0.0,   "clip_range": 0.25,"gae_lambda": 0.95},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.1,   "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 1e-4, "gamma": 0.90, "n_steps": 512,  "batch_size": 64,  "n_epochs": 10, "ent_coef": 0.01,  "clip_range": 0.15,"gae_lambda": 0.80},
]

A2C_CONFIGS = [
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,   "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 5,   "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 10,  "ent_coef": 0.0,   "vf_coef": 0.25, "max_grad_norm": 0.5, "gae_lambda": 0.9},
    {"learning_rate": 7e-4, "gamma": 0.95, "n_steps": 20,  "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 5,   "ent_coef": 0.05,  "vf_coef": 0.5,  "max_grad_norm": 1.0, "gae_lambda": 1.0},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 32,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.95},
    {"learning_rate": 7e-4, "gamma": 0.98, "n_steps": 5,   "ent_coef": 0.001, "vf_coef": 1.0,  "max_grad_norm": 0.5, "gae_lambda": 1.0},
    {"learning_rate": 2e-4, "gamma": 0.99, "n_steps": 16,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.9},
    {"learning_rate": 1e-3, "gamma": 0.90, "n_steps": 5,   "ent_coef": 0.05,  "vf_coef": 0.25, "max_grad_norm": 2.0, "gae_lambda": 1.0},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 64,  "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "gae_lambda": 0.95},
]

REINFORCE_CONFIGS = [
    {"learning_rate": 3e-4, "gamma": 0.99, "hidden_size": 128, "entropy_coef": 0.01,  "max_grad_norm": 0.5},
    {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 128, "entropy_coef": 0.0,   "max_grad_norm": 0.5},
    {"learning_rate": 1e-4, "gamma": 0.99, "hidden_size": 128, "entropy_coef": 0.01,  "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "gamma": 0.95, "hidden_size": 128, "entropy_coef": 0.05,  "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "gamma": 0.99, "hidden_size": 256, "entropy_coef": 0.01,  "max_grad_norm": 0.5},
    {"learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 64,  "entropy_coef": 0.0,   "max_grad_norm": 1.0},
    {"learning_rate": 3e-4, "gamma": 0.90, "hidden_size": 128, "entropy_coef": 0.1,   "max_grad_norm": 0.5},
    {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 256, "entropy_coef": 0.001, "max_grad_norm": 0.5},
    {"learning_rate": 2e-4, "gamma": 0.98, "hidden_size": 128, "entropy_coef": 0.01,  "max_grad_norm": 2.0},
    {"learning_rate": 3e-4, "gamma": 0.99, "hidden_size": 128, "entropy_coef": 0.01,  "max_grad_norm": 0.5},
]


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def make_env():
    return WildlifeLoopEnv(render_mode=None)


def evaluate_sb3(model, n_episodes=EVAL_EPISODES):
    env = make_env()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total = 0.0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            total += r
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ─── REINFORCE helpers ─────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    def act(self, obs):
        x = torch.FloatTensor(obs).unsqueeze(0)
        probs = self(x)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()


def compute_returns(rewards, gamma):
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)
    t = torch.FloatTensor(ret)
    if t.std() > 1e-8:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


def train_reinforce(cfg, n_episodes=REINFORCE_EPS):
    env  = make_env()
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
    policy    = PolicyNet(obs_dim, act_dim, cfg["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
    ep_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        log_probs, entropies, rewards = [], [], []
        total = 0.0

        while not done and not truncated:
            a, lp, ent = policy.act(obs)
            obs, r, done, truncated, _ = env.step(a)
            log_probs.append(lp); entropies.append(ent); rewards.append(r)
            total += r

        ep_rewards.append(total)
        returns  = compute_returns(rewards, cfg["gamma"])
        lp_t     = torch.stack(log_probs)
        ent_t    = torch.stack(entropies)
        loss     = -(lp_t * returns).mean() - cfg["entropy_coef"] * ent_t.mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
        optimizer.step()

    env.close()

    # evaluate
    eval_env = make_env()
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = truncated = False
        total = 0.0
        while not done and not truncated:
            a, _, _ = policy.act(obs)
            obs, r, done, truncated, _ = eval_env.step(a)
            total += r
        eval_rewards.append(total)
    eval_env.close()

    return float(np.mean(eval_rewards)), float(np.std(eval_rewards))


# ─── RUN TUNING ────────────────────────────────────────────────────────────────

def run_sb3_tuning(algo_name, model_class, configs):
    rows = []
    print(f"\n{'='*60}")
    print(f"Tuning {algo_name} — {len(configs)} runs × {TRAIN_STEPS:,} steps")
    print(f"{'='*60}")

    for i, cfg in enumerate(configs, 1):
        print(f"  Run {i:>2}/10  params={cfg}")
        try:
            env   = make_env()
            model = model_class("MlpPolicy", env, verbose=0, **cfg)
            model.learn(total_timesteps=TRAIN_STEPS)
            mean_r, std_r = evaluate_sb3(model)
            env.close()
            status = "ok"
        except Exception as e:
            mean_r, std_r = np.nan, np.nan
            status = str(e)[:60]

        row = {"run": i, "mean_reward": round(mean_r, 2),
               "std_reward": round(std_r, 2), "status": status, **cfg}
        rows.append(row)
        print(f"         → mean_reward={mean_r:.2f}  std={std_r:.2f}")

    return pd.DataFrame(rows)


def run_reinforce_tuning():
    rows = []
    print(f"\n{'='*60}")
    print(f"Tuning REINFORCE — 10 runs × {REINFORCE_EPS} episodes")
    print(f"{'='*60}")

    for i, cfg in enumerate(REINFORCE_CONFIGS, 1):
        print(f"  Run {i:>2}/10  params={cfg}")
        try:
            mean_r, std_r = train_reinforce(cfg)
            status = "ok"
        except Exception as e:
            mean_r, std_r = np.nan, np.nan
            status = str(e)[:60]

        row = {"run": i, "mean_reward": round(mean_r, 2),
               "std_reward": round(std_r, 2), "status": status, **cfg}
        rows.append(row)
        print(f"         → mean_reward={mean_r:.2f}  std={std_r:.2f}")

    return pd.DataFrame(rows)


# ─── SAVE RESULTS ──────────────────────────────────────────────────────────────

def save_table_png(df: pd.DataFrame, algo: str, out_dir: str):
    """Render DataFrame as a clean PNG table."""
    color = COLORS.get(algo, "#888888")

    # columns to show (drop verbose ones)
    skip = {"status", "run"}
    cols = ["run"] + [c for c in df.columns if c not in skip and c != "run"]

    fig, ax = plt.subplots(figsize=(max(14, len(cols) * 1.4), len(df) * 0.55 + 1.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # round floats for display
    display_df = df[cols].copy()
    for c in display_df.columns:
        if display_df[c].dtype == float:
            display_df[c] = display_df[c].round(5)

    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # style header
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor(color)
        cell.set_text_props(color="white", fontweight="bold")

    # style rows — highlight best
    best_idx = df["mean_reward"].idxmax()
    for i in range(1, len(df) + 1):
        for j in range(len(cols)):
            cell = tbl[i, j]
            if i - 1 == best_idx:
                cell.set_facecolor("#1f3d2f")
                cell.set_text_props(color="#44ff88", fontweight="bold")
            else:
                cell.set_facecolor("#161b22" if i % 2 == 0 else "#1c2128")
                cell.set_text_props(color="#c9d1d9")
            cell.set_edgecolor("#30363d")

    ax.set_title(
        f"{algo} — Hyperparameter Tuning Results  "
        f"(★ best: run {best_idx+1}, mean={df.loc[best_idx,'mean_reward']:.1f})",
        fontsize=11, color="#e6edf3", pad=12,
    )

    path = os.path.join(out_dir, f"{algo.lower()}_tuning_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved table PNG → {path}")


def save_reward_bar(results: dict, out_dir: str):
    """Bar chart — best reward per algorithm across all runs."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Hyperparameter Tuning — Best Mean Reward per Algorithm",
                 fontsize=13, color="#e6edf3", fontweight="bold")

    algos  = list(results.keys())
    bests  = [results[a]["mean_reward"].max() for a in algos]
    colors = [COLORS[a] for a in algos]
    bars   = ax.bar(algos, bests, color=colors, width=0.5)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, fontsize=12)

    for bar, val in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bests) * 0.01,
                f"{val:.0f}", ha="center", color="#e6edf3",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Best Mean Reward"); ax.grid(True, axis="y")
    path = os.path.join(out_dir, "tuning_best_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
        "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor":  "#e6edf3", "xtick.color":     "#8b949e",
        "ytick.color":      "#8b949e", "grid.color":      "#21262d",
        "grid.linestyle":   "--",      "grid.linewidth":  0.6,
        "legend.facecolor": "#161b22", "legend.edgecolor":"#30363d",
        "legend.labelcolor":"#c9d1d9",
    })


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    # DQN
    df_dqn = run_sb3_tuning("DQN", DQN, DQN_CONFIGS)
    df_dqn.to_csv(os.path.join(OUT_DIR, "dqn_tuning.csv"), index=False)
    save_table_png(df_dqn, "DQN", OUT_DIR)
    all_results["DQN"] = df_dqn

    # PPO
    df_ppo = run_sb3_tuning("PPO", PPO, PPO_CONFIGS)
    df_ppo.to_csv(os.path.join(OUT_DIR, "ppo_tuning.csv"), index=False)
    save_table_png(df_ppo, "PPO", OUT_DIR)
    all_results["PPO"] = df_ppo

    # A2C
    df_a2c = run_sb3_tuning("A2C", A2C, A2C_CONFIGS)
    df_a2c.to_csv(os.path.join(OUT_DIR, "a2c_tuning.csv"), index=False)
    save_table_png(df_a2c, "A2C", OUT_DIR)
    all_results["A2C"] = df_a2c

    # REINFORCE
    df_rf = run_reinforce_tuning()
    df_rf.to_csv(os.path.join(OUT_DIR, "reinforce_tuning.csv"), index=False)
    save_table_png(df_rf, "REINFORCE", OUT_DIR)
    all_results["REINFORCE"] = df_rf

    # combined bar chart
    save_reward_bar(all_results, OUT_DIR)

    # print summary
    print(f"\n{'='*60}")
    print("TUNING SUMMARY — Best run per algorithm")
    print(f"{'='*60}")
    for algo, df in all_results.items():
        best = df.loc[df["mean_reward"].idxmax()]
        print(f"\n  {algo}")
        print(f"    mean_reward : {best['mean_reward']:.2f}")
        for col in df.columns:
            if col not in {"run", "mean_reward", "std_reward", "status"}:
                print(f"    {col:<25} {best[col]}")

    print(f"\nAll results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()