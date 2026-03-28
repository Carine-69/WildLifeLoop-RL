"""
evaluate_and_record.py
======================
1. Evaluates every checkpoint for DQN, PPO, A2C
2. Finds the best checkpoint per algorithm
3. Saves all plots (per-algo curves + combined comparison)
4. Records 3 MP4 videos for each best model
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import DQN, PPO, A2C
from environment.lifeloop_env import WildlifeLoopEnv

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_FOLDER   = "models"
VIDEO_FOLDER   = "videos"
RESULTS_FOLDER = "results"
EVAL_EPISODES  = 5          
N_VIDEO_EPS    = 3         

MODEL_MAP = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

# colour palette
COLORS = {"dqn": "#E94560", "ppo": "#0F9B8E", "a2c": "#F5A623"}
# ───────────────────────────────────────────────────────────────────────────────


def extract_step(filename: str) -> int:
    """Pull the numeric step count from a checkpoint filename."""
    match = re.search(r"_(\d+)_steps", filename)
    if match:
        return int(match.group(1))
    if "final" in filename:
        return int(1e9)        
    return -1


def evaluate_checkpoint(model_path: str, model_class, n_episodes: int = EVAL_EPISODES):
    """Return (mean_reward, std_reward) over n_episodes."""
    env = WildlifeLoopEnv(render_mode="rgb_array")
    model = model_class.load(model_path, device="cpu")
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total = 0.0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward
        rewards.append(total)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))



def evaluate_all():
    """
    Returns
    -------
    results : dict  {algo: {"steps": [...], "means": [...], "stds": [...],
                             "best_path": str, "best_step": int, "best_mean": float}}
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    results = {}

    for algo, model_class in MODEL_MAP.items():
        algo_dir = os.path.join(MODEL_FOLDER, algo)
        if not os.path.isdir(algo_dir):
            print(f"[SKIP] {algo_dir} not found")
            continue

        checkpoints = [
            f for f in os.listdir(algo_dir) if f.endswith(".zip")
        ]
        # sort by step count
        checkpoints.sort(key=extract_step)

        steps, means, stds = [], [], []
        best_mean, best_path, best_step = -np.inf, None, None

        total = len(checkpoints)
        print(f"\n{'='*60}")
        print(f"Evaluating {algo.upper()} — {total} checkpoints × {EVAL_EPISODES} episodes")
        print(f"{'='*60}")

        for i, fname in enumerate(checkpoints, 1):
            step = extract_step(fname)
            path = os.path.join(algo_dir, fname)

            try:
                mean, std = evaluate_checkpoint(path, model_class)
            except Exception as e:
                print(f"  [{i}/{total}] SKIP {fname}: {e}")
                continue

            steps.append(step)
            means.append(mean)
            stds.append(std)

            marker = " ◀ BEST" if mean > best_mean else ""
            if mean > best_mean:
                best_mean  = mean
                best_path  = path
                best_step  = step

            print(f"  [{i:>3}/{total}] step={step:>9,}  "
                  f"mean={mean:>8.2f}  std={std:>6.2f}{marker}")

        results[algo] = {
            "steps":     steps,
            "means":     means,
            "stds":      stds,
            "best_path": best_path,
            "best_step": best_step,
            "best_mean": best_mean,
        }
        print(f"\n  ✔ Best {algo.upper()}: step={best_step:,}  mean={best_mean:.2f}  → {best_path}")

    return results

# PLOTTING 

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#c9d1d9",
        "axes.titlecolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.labelcolor":"#c9d1d9",
        "font.family":      "DejaVu Sans",
    })


def plot_individual(algo: str, data: dict):
    """Reward curve + std band for a single algorithm."""
    _style()
    steps = np.array(data["steps"])
    means = np.array(data["means"])
    stds  = np.array(data["stds"])
    color = COLORS[algo]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{algo.upper()} — Training Evaluation", fontsize=15,
                 color="#e6edf3", fontweight="bold", y=0.98)

   
    ax = axes[0]
    ax.fill_between(steps, means - stds, means + stds,
                    color=color, alpha=0.18, label="±1 std")
    ax.plot(steps, means, color=color, linewidth=2, label="Mean reward")

 
    if len(means) >= 10:
        window = max(5, len(means) // 20)
        roll = np.convolve(means, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], roll, color="white",
                linewidth=1.2, linestyle="--", alpha=0.6, label=f"Rolling avg ({window})")

  
    bx, by = data["best_step"], data["best_mean"]
    ax.scatter([bx], [by], color="gold", s=120, zorder=5,
               marker="*", label=f"Best  ({by:.1f})")
    ax.axvline(bx, color="gold", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True)
    ax.set_xlim(steps[0], steps[-1])


    ax2 = axes[1]
    ax2.bar(steps, stds, color=color, alpha=0.5, width=(steps[-1] - steps[0]) / len(steps) * 0.8)
    ax2.set_ylabel("Std Dev", fontsize=9)
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.grid(True)
    ax2.set_xlim(steps[0], steps[-1])

    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, f"{algo}_reward_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined(results: dict):
    """All algorithms on one axes."""
    _style()
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Algorithm Comparison — Mean Episode Reward",
                 fontsize=14, color="#e6edf3", fontweight="bold")

    for algo, data in results.items():
        steps = np.array(data["steps"])
        means = np.array(data["means"])
        stds  = np.array(data["stds"])
        color = COLORS[algo]

        ax.fill_between(steps, means - stds, means + stds,
                        color=color, alpha=0.12)
        ax.plot(steps, means, color=color, linewidth=2, label=algo.upper())

        bx, by = data["best_step"], data["best_mean"]
        ax.scatter([bx], [by], color=color, s=160, zorder=5,
                   marker="*", edgecolors="white", linewidths=0.6)

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, "comparison_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_bar(results: dict):
    """Bar chart comparing best rewards across algorithms."""
    _style()
    algos  = list(results.keys())
    bests  = [results[a]["best_mean"] for a in algos]
    colors = [COLORS[a] for a in algos]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Best Mean Reward per Algorithm",
                 fontsize=13, color="#e6edf3", fontweight="bold")

    bars = ax.bar(algos, bests, color=colors, width=0.5, edgecolor="#30363d")
    for bar, val in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(max(bests)) * 0.01,
                f"{val:.1f}", ha="center", va="bottom",
                color="#e6edf3", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos], fontsize=12)
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, "best_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reward_distribution(results: dict):
    """Violin / box plot of all evaluated rewards per algorithm."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Reward Distribution Across All Checkpoints",
                 fontsize=13, color="#e6edf3", fontweight="bold")

    data_list  = [results[a]["means"] for a in results]
    algo_names = [a.upper() for a in results]
    colors     = [COLORS[a] for a in results]

    parts = ax.violinplot(data_list, showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color("#8b949e")
    parts["cmins"].set_color("#8b949e")
    parts["cbars"].set_color("#8b949e")

    ax.set_xticks(range(1, len(algo_names) + 1))
    ax.set_xticklabels(algo_names, fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, "reward_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_convergence_speed(results: dict):
    """Steps-to-reach-X%-of-best-reward for each algorithm."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Convergence — Reward Over Time (Normalised)",
                 fontsize=13, color="#e6edf3", fontweight="bold")

    for algo, data in results.items():
        steps = np.array(data["steps"])
        means = np.array(data["means"])
        norm  = (means - means.min()) / (means.max() - means.min() + 1e-8)
        ax.plot(steps, norm, color=COLORS[algo], linewidth=2, label=algo.upper())

    ax.axhline(0.9, color="gold", linewidth=0.8, linestyle="--", alpha=0.7,
               label="90 % of best")
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Normalised Reward", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, "convergence_speed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_grid(results: dict):
    """2×2 dashboard combining the key plots."""
    _style()
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("LifeLoop RL — Full Evaluation Dashboard",
                 fontsize=16, color="#e6edf3", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)


    ax1 = fig.add_subplot(gs[0, 0])
    for algo, data in results.items():
        steps = np.array(data["steps"])
        means = np.array(data["means"])
        ax1.plot(steps, means, color=COLORS[algo], linewidth=1.8, label=algo.upper())
    ax1.set_title("Mean Reward per Checkpoint", fontsize=11)
    ax1.set_xlabel("Steps"); ax1.set_ylabel("Reward")
    ax1.legend(fontsize=9); ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    algos  = list(results.keys())
    bests  = [results[a]["best_mean"] for a in algos]
    bars   = ax2.bar(algos, bests, color=[COLORS[a] for a in algos], width=0.5)
    for bar, val in zip(bars, bests):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + abs(max(bests)) * 0.01,
                 f"{val:.1f}", ha="center", color="#e6edf3", fontsize=10)
    ax2.set_title("Best Mean Reward", fontsize=11)
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels([a.upper() for a in algos])
    ax2.grid(True, axis="y")

    ax3 = fig.add_subplot(gs[1, 0])
    parts = ax3.violinplot([results[a]["means"] for a in algos],
                           showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [COLORS[a] for a in algos]):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color("#8b949e")
    parts["cmins"].set_color("#8b949e")
    parts["cbars"].set_color("#8b949e")
    ax3.set_xticks(range(1, len(algos) + 1))
    ax3.set_xticklabels([a.upper() for a in algos])
    ax3.set_title("Reward Distribution", fontsize=11)
    ax3.set_ylabel("Reward"); ax3.grid(True, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    for algo, data in results.items():
        steps = np.array(data["steps"])
        means = np.array(data["means"])
        norm  = (means - means.min()) / (means.max() - means.min() + 1e-8)
        ax4.plot(steps, norm, color=COLORS[algo], linewidth=1.8, label=algo.upper())
    ax4.axhline(0.9, color="gold", linewidth=0.8, linestyle="--", alpha=0.7)
    ax4.set_title("Normalised Convergence", fontsize=11)
    ax4.set_xlabel("Steps"); ax4.set_ylabel("Norm. Reward")
    ax4.legend(fontsize=9); ax4.grid(True)

    path = os.path.join(RESULTS_FOLDER, "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def save_all_plots(results: dict):
    print(f"\n{'='*60}")
    print("Saving plots …")
    print(f"{'='*60}")
    for algo, data in results.items():
        plot_individual(algo, data)
    plot_combined(results)
    plot_best_bar(results)
    plot_reward_distribution(results)
    plot_convergence_speed(results)
    plot_summary_grid(results)


#  FRAME RENDERER

def _env_to_frame(env, algo: str, ep: int, step: int,
                  total_reward: float, fig, ax) -> np.ndarray:
    """
    Draw the current env state onto a matplotlib figure and return an RGB
    numpy array. No window is ever shown (Agg backend).
    Reads directly from env internals so no render_mode dependency.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    ax.clear()
    ax.set_facecolor("#0d1f0d")
    ax.set_xlim(-0.5, env.GRID - 0.5)
    ax.set_ylim(-0.5, env.GRID - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    GRID = env.GRID

    for zid in env._visited:
        gx = zid % GRID
        gy = zid // GRID
        ax.add_patch(mpatches.Rectangle(
            (gx - 0.5, gy - 0.5), 1, 1,
            linewidth=0, facecolor="#1a3d1a", alpha=0.6, zorder=1
        ))

    for i in range(GRID + 1):
        ax.axhline(i - 0.5, color="#1f3d1f", linewidth=0.4, zorder=2)
        ax.axvline(i - 0.5, color="#1f3d1f", linewidth=0.4, zorder=2)

    for i, (px, py) in enumerate(env.POACHER_SPOTS):
        active = env._poacher_on[i]
        color  = "#ff2222" if active else "#555555"
        ax.add_patch(mpatches.Circle(
            (px, py), 0.35, color=color, alpha=0.5, zorder=3
        ))
        ax.text(px, py, "P", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)

    for i, (ax_, ay) in enumerate(env._animals):
        anom = env._anomaly[i]
        c = plt.cm.RdYlGn(1.0 - anom)
        ax.scatter(ax_, ay, s=60, color=c, zorder=5,
                   edgecolors="white", linewidths=0.4)

    rx, ry = env._ranger
    ax.scatter(rx, ry, s=180, color="#00cfff", zorder=7,
               marker="^", edgecolors="white", linewidths=0.8)

 
    from environment.lifeloop_env import DETECT_R
    ax.add_patch(mpatches.Circle(
        (rx, ry), DETECT_R, fill=False,
        edgecolor="#00cfff", linewidth=0.8, linestyle="--", alpha=0.5, zorder=6
    ))


    batt = np.clip(env._battery, 0.0, 1.0)
    batt_col = "#44ff44" if batt > 0.4 else ("#ffaa00" if batt > 0.2 else "#ff2222")
    ax.add_patch(mpatches.Rectangle(
        (-0.4, -0.45), batt * (GRID - 0.2), 0.18,
        color=batt_col, alpha=0.85, zorder=8
    ))
    ax.text(-0.4, -0.55, f"Battery {batt*100:.0f}%",
            fontsize=7, color=batt_col, zorder=9)

    coverage = len(env._visited) / (GRID * GRID) * 100
    title = (f"{algo.upper()}  |  Episode {ep}  |  Step {step}"
             f"  |  Reward {total_reward:+.1f}"
             f"  |  Coverage {coverage:.0f}%"
             f"  |  Caught {env._caught}  Missed {env._missed}")
    ax.set_title(title, fontsize=8, color="#c9d1d9",
                 pad=4, fontfamily="monospace")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return frame[:, :, :3].copy()  # drop alpha -> RGB



def record_videos(results: dict, n_episodes: int = N_VIDEO_EPS, fps: int = 15):
    """
    Records one MP4 per episode for each best model.
    Uses matplotlib (Agg) to render frames — no window, no moviepy, no RecordVideo.
    pip install imageio imageio-ffmpeg
    """
    try:
        import imageio
    except ImportError:
        print("[ERROR] Run: pip install imageio imageio-ffmpeg")
        return

    plt.switch_backend("Agg")

    print(f"\n{'='*60}")
    print("Recording videos (silent, no display) …")
    print(f"{'='*60}")
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)

    for algo, data in results.items():
        model_class = MODEL_MAP[algo]
        best_path   = data["best_path"]
        best_step   = data["best_step"]

        if best_path is None:
            print(f"[SKIP] No valid model found for {algo.upper()}")
            continue

        out_dir = os.path.join(VIDEO_FOLDER, f"{algo}_best_step{best_step}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  {algo.upper()} — best step={best_step:,}  → {out_dir}")
        model = model_class.load(best_path, device="cpu")

        for ep in range(1, n_episodes + 1):
            env = WildlifeLoopEnv(render_mode=None)
            obs, _ = env.reset()
            done = truncated = False
            total = 0.0
            frames = []

            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total += reward
                frame = _env_to_frame(env, algo, ep, info["step"], total, fig, ax)
                frames.append(frame)

            env.close()

            out_path = os.path.join(out_dir, f"{algo}_ep{ep}.mp4")
            writer = imageio.get_writer(
                out_path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
            )
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"    ep{ep}  steps={info['step']}  reward={total:.1f}"
                  f"  caught={env._caught}  frames={len(frames)}  → {out_path}")

    plt.close(fig)
    print(f"\nAll videos saved to {VIDEO_FOLDER}/")


def main():
    results = evaluate_all()

    save_all_plots(results)

    record_videos(results)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    for algo, data in results.items():
        print(f"  {algo.upper():>4}  best_step={data['best_step']:>10,}  "
              f"mean_reward={data['best_mean']:>8.2f}  -> {data['best_path']}")
    print(f"\nPlots  -> {RESULTS_FOLDER}/")
    print(f"Videos -> {VIDEO_FOLDER}/")


if __name__ == "__main__":
    main()