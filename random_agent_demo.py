"""
random_agent_demo.py
=====================
Shows the agent taking RANDOM actions (no model) in the environment.
Instead of a live simulation (which freezes the computer), this script:
  1. Runs 1 episode of random actions
  2. Captures a frame every N steps using matplotlib Agg (no window)
  3. Saves a visual grid of frames → results/random_agent_demo.png
  4. Saves a summary CSV  → results/random_agent_steps.csv

This satisfies the rubric requirement:
  "Create a static file that shows the agent taking random actions
   (not using a model) in the custom environment."

Usage:
    python random_agent_demo.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from environment.lifeloop_env import WildlifeLoopEnv, ACTION_NAMES, DETECT_R, POACHER_SPOTS, GRID

RESULTS_DIR   = "results"
CAPTURE_STEPS = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499]
SEED          = 42


def render_frame(env, step, action, reward, total_reward, ax):
    """Draw env state onto matplotlib axes — no window."""
    ax.clear()
    ax.set_facecolor("#0d1f0d")
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-0.5, GRID - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # coverage
    for zid in env._visited:
        gx, gy = zid % GRID, zid // GRID
        ax.add_patch(mpatches.Rectangle(
            (gx - 0.5, gy - 0.5), 1, 1,
            linewidth=0, facecolor="#1a3d1a", alpha=0.55, zorder=1
        ))
    # grid lines
    for i in range(GRID + 1):
        ax.axhline(i - 0.5, color="#1f3d1f", lw=0.3, zorder=2)
        ax.axvline(i - 0.5, color="#1f3d1f", lw=0.3, zorder=2)

    # poacher spots
    for i, (px, py) in enumerate(POACHER_SPOTS):
        col = "#ff3333" if env._poacher_on[i] else "#444444"
        ax.add_patch(mpatches.Circle((px, py), 0.32, color=col, alpha=0.7, zorder=3))
        ax.text(px, py, "P", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=4)

    # animals
    for i, (ax_, ay) in enumerate(env._animals):
        c = plt.cm.RdYlGn(1.0 - env._anomaly[i])
        ax.scatter(ax_, ay, s=40, color=c, zorder=5,
                   edgecolors="white", linewidths=0.3)

    # ranger
    rx, ry = env._ranger
    ax.scatter(rx, ry, s=160, color="#00cfff", zorder=7,
               marker="^", edgecolors="white", linewidths=0.7)
    ax.add_patch(mpatches.Circle(
        (rx, ry), DETECT_R, fill=False,
        edgecolor="#00cfff", lw=0.7, linestyle="--", alpha=0.4, zorder=6
    ))

    # battery bar
    batt = float(np.clip(env._battery, 0, 1))
    col  = "#44ff44" if batt > 0.5 else "#ffaa00" if batt > 0.25 else "#ff2222"
    ax.add_patch(mpatches.Rectangle(
        (-0.4, -0.44), batt * (GRID - 0.2), 0.16,
        color=col, alpha=0.85, zorder=8
    ))

    cov = len(env._visited) / (GRID * GRID) * 100
    ax.set_title(
        f"Step {step}  |  {ACTION_NAMES[action]}\n"
        f"r={reward:+.1f}  total={total_reward:+.0f}  "
        f"bat={batt*100:.0f}%  cov={cov:.0f}%",
        fontsize=6.5, color="#c9d1d9", pad=2, fontfamily="monospace"
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env  = WildlifeLoopEnv(render_mode=None)
    obs, _ = env.reset(seed=SEED)
    rng  = np.random.default_rng(SEED)

    rows         = []
    total_reward = 0.0
    frames       = {}   # step → (action, reward, env snapshot copy)

    print("Running random agent episode …")
    done = truncated = False
    step = 0

    # --- run episode, capture snapshots at key steps ---
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        rows.append({
            "step":           step,
            "action":         ACTION_NAMES[action],
            "reward":         round(reward, 3),
            "total_reward":   round(total_reward, 3),
            "battery":        round(info["battery"], 3),
            "coverage_pct":   round(info["coverage_pct"] * 100, 1),
            "threats_caught": info["threats_caught"],
            "threats_missed": info["threats_missed"],
            "false_alerts":   info["false_alerts"],
        })

        if step in CAPTURE_STEPS or done or truncated:
            frames[step] = {
                "action":       action,
                "reward":       reward,
                "total_reward": total_reward,
                # deep-copy env state for rendering
                "ranger":    env._ranger.copy(),
                "animals":   env._animals.copy(),
                "anomaly":   env._anomaly.copy(),
                "poacher_on":env._poacher_on.copy(),
                "visited":   set(env._visited),
                "battery":   env._battery,
            }

    env.close()
    print(f"  Episode ended at step {step}  total_reward={total_reward:.2f}")

    # --- save CSV ---
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "random_agent_steps.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Step log saved → {csv_path}")

    # --- build frame grid ---
    captured = sorted(frames.keys())
    n_frames = len(captured)
    ncols    = 4
    nrows    = int(np.ceil(n_frames / ncols))

    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.8), facecolor="#0d1117")
    fig.suptitle(
        "WildlifeLoopEnv — Random Agent Demo  (no model, random actions)",
        fontsize=12, color="#e6edf3", fontweight="bold", y=1.01
    )

    for idx, step_num in enumerate(captured):
        ax  = fig.add_subplot(nrows, ncols, idx + 1)
        snap = frames[step_num]

        # temporarily patch env state for rendering helper
        env2 = WildlifeLoopEnv(render_mode=None)
        env2.reset(seed=0)
        env2._ranger     = snap["ranger"]
        env2._animals    = snap["animals"]
        env2._anomaly    = snap["anomaly"]
        env2._poacher_on = snap["poacher_on"]
        env2._visited    = snap["visited"]
        env2._battery    = snap["battery"]

        render_frame(env2, step_num, snap["action"],
                     snap["reward"], snap["total_reward"], ax)
        env2.close()

    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, "random_agent_demo.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Frame grid saved → {img_path}")

    # --- print summary ---
    print("\n" + "=" * 55)
    print("RANDOM AGENT SUMMARY")
    print("=" * 55)
    print(f"  Total steps     : {step}")
    print(f"  Total reward    : {total_reward:.2f}")
    print(f"  Threats caught  : {rows[-1]['threats_caught']}")
    print(f"  Threats missed  : {rows[-1]['threats_missed']}")
    print(f"  False alerts    : {rows[-1]['false_alerts']}")
    print(f"  Grid coverage   : {rows[-1]['coverage_pct']:.1f}%")
    print(f"  Battery left    : {rows[-1]['battery']*100:.1f}%")
    print(f"\n  Action distribution:")
    act_counts = df["action"].value_counts()
    for act, cnt in act_counts.items():
        print(f"    {act:<18} {cnt:>4}x  ({cnt/step*100:.1f}%)")


if __name__ == "__main__":
    main()