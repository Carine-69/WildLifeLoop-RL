"""
training/train_ppo.py
======================
PPO training for WildlifeLoopEnv (updated for 22-dim obs space).
"""

import os
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.lifeloop_env import WildlifeLoopEnv

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TOTAL_STEPS = 1_000_000
SAVE_FREQ   = 10_000
N_ENVS      = 4          # parallel envs — speeds up PPO significantly
MODEL_DIR   = "./models/ppo"
LOG_DIR     = "./logs/ppo"
RESULTS_DIR = "./results"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

PPO_PARAMS = {
    "learning_rate": 2.5e-4,
    "n_steps":       2048,
    "batch_size":    256,     # larger batch for stability with 22-dim obs
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "ent_coef":      0.01,    # entropy encourages exploration
    "vf_coef":       0.5,
    "clip_range":    0.2,
    "max_grad_norm": 0.5,
    "verbose":       1,
    "policy_kwargs": dict(net_arch=[dict(pi=[256, 256, 128],
                                        vf=[256, 256, 128])]),
}
# ──────────────────────────────────────────────────────────────────────────────


def make_env_fn(rank: int):
    def _init():
        env = WildlifeLoopEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init


def save_training_plot(log_dir: str, out_dir: str):
    import glob, pandas as pd
    files = glob.glob(os.path.join(log_dir, "**/*.monitor.csv"), recursive=True)
    if not files:
        return
    dfs = [pd.read_csv(f, skiprows=1) for f in files]
    df  = pd.concat(dfs).sort_values("t").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="#0d1117")
    fig.suptitle("PPO Training — WildlifeLoopEnv", fontsize=13,
                 color="#e6edf3", fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    ax1 = axes[0]
    ax1.plot(df["t"], df["r"], color="#0F9B8E", linewidth=0.6, alpha=0.4)
    if len(df) >= 20:
        ax1.plot(df["t"], df["r"].rolling(20).mean(),
                 color="#0F9B8E", linewidth=2, label="Rolling mean (20)")
    ax1.set_ylabel("Episode Reward", color="#c9d1d9")
    ax1.set_xlabel("Timesteps",      color="#c9d1d9")
    ax1.legend(fontsize=9)
    ax1.grid(True, color="#21262d", linestyle="--", linewidth=0.5)

    ax2 = axes[1]
    ax2.plot(df["t"], df["l"], color="#8B5CF6", linewidth=0.6, alpha=0.4)
    if len(df) >= 20:
        ax2.plot(df["t"], df["l"].rolling(20).mean(),
                 color="#8B5CF6", linewidth=2)
    ax2.set_ylabel("Episode Length", color="#c9d1d9")
    ax2.set_xlabel("Timesteps",      color="#c9d1d9")
    ax2.grid(True, color="#21262d", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ppo_training_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Training curve saved → {path}")


def main():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(LOG_DIR,     exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("PPO Training — WildlifeLoopEnv")
    print("=" * 60)
    print(f"  Device        : {DEVICE}")
    print(f"  Total steps   : {TOTAL_STEPS:,}")
    print(f"  Parallel envs : {N_ENVS}")
    print(f"  Obs dim       : 22")
    print(f"  Net arch      : pi/vf [256, 256, 128]")
    print(f"  Learning rate : {PPO_PARAMS['learning_rate']}")
    print(f"  n_steps       : {PPO_PARAMS['n_steps']}")
    print(f"  ent_coef      : {PPO_PARAMS['ent_coef']}")
    print("=" * 60)

    train_env = SubprocVecEnv([make_env_fn(i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env, LOG_DIR)
    eval_env  = Monitor(WildlifeLoopEnv(render_mode=None))

    model = PPO(
        "MlpPolicy",
        train_env,
        **PPO_PARAMS,
        device=DEVICE,
        tensorboard_log=LOG_DIR,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=SAVE_FREQ // N_ENVS,   # adjust for parallel envs
            save_path=MODEL_DIR,
            name_prefix="wildlife_ppo",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, "best"),
            log_path=LOG_DIR,
            eval_freq=10_000 // N_ENVS,
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

    model.save(os.path.join(MODEL_DIR, "wildlife_ppo_final"))
    print(f"\n  Final model saved → {MODEL_DIR}/wildlife_ppo_final.zip")

    train_env.close()
    eval_env.close()

    save_training_plot(LOG_DIR, RESULTS_DIR)
    print("\nPPO training complete.")


if __name__ == "__main__":
    main()