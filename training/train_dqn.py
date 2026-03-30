"""
training/train_dqn.py
======================
DQN training for WildlifeLoopEnv (updated for 22-dim obs space).
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.lifeloop_env import WildlifeLoopEnv

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 1_000_000
SAVE_FREQ     = 5_000
MODEL_DIR     = "./models/dqn"
LOG_DIR       = "./logs/dqn"
RESULTS_DIR   = "./results"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

DQN_PARAMS = {
    "learning_rate":          1e-4,
    "buffer_size":            100_000,  # larger buffer for 22-dim obs
    "learning_starts":        2_000,    # more warmup before training
    "batch_size":             128,
    "tau":                    1.0,
    "gamma":                  0.99,
    "train_freq":             4,
    "target_update_interval": 1_000,
    "exploration_fraction":   0.2,      # more exploration given harder env
    "exploration_final_eps":  0.05,
    "verbose":                1,
    "policy_kwargs":          dict(net_arch=[256, 256, 128]),  # deeper for 22-dim
}
# ──────────────────────────────────────────────────────────────────────────────


def make_env(seed=0):
    env = WildlifeLoopEnv(render_mode=None)
    env = Monitor(env, LOG_DIR)
    return env


def save_training_plot(log_dir: str, out_dir: str):
    """Parse Monitor CSV and save reward curve."""
    import glob, pandas as pd
    files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
    if not files:
        return
    dfs = []
    for f in files:
        df = pd.read_csv(f, skiprows=1)
        dfs.append(df)
    df = pd.concat(dfs).sort_values("t").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="#0d1117")
    fig.suptitle("DQN Training — WildlifeLoopEnv", fontsize=13,
                 color="#e6edf3", fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    # episode reward
    ax1 = axes[0]
    ax1.plot(df["t"], df["r"], color="#E94560", linewidth=0.6, alpha=0.4)
    if len(df) >= 20:
        roll = df["r"].rolling(20).mean()
        ax1.plot(df["t"], roll, color="#E94560", linewidth=2, label="Rolling mean (20)")
    ax1.set_ylabel("Episode Reward", color="#c9d1d9")
    ax1.set_xlabel("Timesteps", color="#c9d1d9")
    ax1.legend(fontsize=9)
    ax1.grid(True, color="#21262d", linestyle="--", linewidth=0.5)

    # episode length
    ax2 = axes[1]
    ax2.plot(df["t"], df["l"], color="#8B5CF6", linewidth=0.6, alpha=0.4)
    if len(df) >= 20:
        roll_l = df["l"].rolling(20).mean()
        ax2.plot(df["t"], roll_l, color="#8B5CF6", linewidth=2)
    ax2.set_ylabel("Episode Length", color="#c9d1d9")
    ax2.set_xlabel("Timesteps", color="#c9d1d9")
    ax2.grid(True, color="#21262d", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dqn_training_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Training curve saved → {path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("DQN Training — WildlifeLoopEnv")
    print("=" * 60)
    print(f"  Device        : {DEVICE}")
    print(f"  Total steps   : {TOTAL_STEPS:,}")
    print(f"  Obs dim       : 22")
    print(f"  Net arch      : {DQN_PARAMS['policy_kwargs']['net_arch']}")
    print(f"  Learning rate : {DQN_PARAMS['learning_rate']}")
    print(f"  Buffer size   : {DQN_PARAMS['buffer_size']:,}")
    print("=" * 60)

    train_env = make_env()
    eval_env  = Monitor(WildlifeLoopEnv(render_mode=None))

    model = DQN(
        "MlpPolicy",
        train_env,
        **DQN_PARAMS,
        device=DEVICE,
        tensorboard_log=LOG_DIR,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=MODEL_DIR,
            name_prefix="wildlife_dqn",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, "best"),
            log_path=LOG_DIR,
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "wildlife_dqn_final"))
    print(f"\n  Final model saved → {MODEL_DIR}/wildlife_dqn_final.zip")

    train_env.close()
    eval_env.close()

    save_training_plot(LOG_DIR, RESULTS_DIR)
    print("\nDQN training complete.")


if __name__ == "__main__":
    main()