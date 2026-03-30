import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.lifeloop_env import WildlifeLoopEnv

# CONFIG
OUT_DIR      = "results/evaluation"
MODEL_DIR    = "models"
N_EPISODES   = 10       
DEVICE       = "cpu"

ALGO_CONFIG = {
    "Random":    {"path": None,  "cls": None,  "color": "#888780"},
    "DQN":       {"path": None,  "cls": None,  "color": "#E94560"},
    "REINFORCE": {"path": None,  "cls": None,  "color": "#8B5CF6"},
    "PPO":       {"path": None,  "cls": None,  "color": "#0F9B8E"},
    "A2C":       {"path": None,  "cls": None,  "color": "#F5A623"},
}

DARK_BG   = "#0d1117"
AXES_BG   = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#c9d1d9"
TITLE_COL = "#e6edf3"
TICK_COL  = "#8b949e"


def find_best_model(algo: str) -> str | None:
    """Find best checkpoint by scanning model folder."""
    folder = os.path.join(MODEL_DIR, algo.lower())
    if not os.path.isdir(folder):
        return None

    # prefer best/ subfolder (saved by EvalCallback)
    best_sub = os.path.join(folder, "best", "best_model.zip")
    if os.path.exists(best_sub):
        return best_sub

    # fall back to final
    final = os.path.join(folder, f"wildlife_{algo.lower()}_final.zip")
    if os.path.exists(final):
        return final

    # fall back to REINFORCE .pt
    final_pt = os.path.join(folder, f"wildlife_{algo.lower()}_final.pt")
    if os.path.exists(final_pt):
        return final_pt

    # scan for highest-step checkpoint
    files = [f for f in os.listdir(folder)
             if f.endswith(".zip") or f.endswith(".pt")]
    if not files:
        return None

    def step_of(fn):
        m = re.search(r"_(\d+)_", fn)
        return int(m.group(1)) if m else 0

    files.sort(key=step_of, reverse=True)
    return os.path.join(folder, files[0])


def load_sb3_model(algo: str, path: str):
    from stable_baselines3 import DQN, PPO, A2C
    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo.lower()]
    return cls.load(path, device=DEVICE)


def load_reinforce(path: str):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(22, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, 7),
            )
        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    ckpt = torch.load(path, map_location="cpu")
    net  = Net()
    net.load_state_dict(ckpt["policy_state"])
    net.eval()

    class Wrapper:
        def predict(self, obs, deterministic=True):
            import torch
            x = torch.FloatTensor(obs).unsqueeze(0)
            p = net(x)
            a = p.argmax(-1) if deterministic else \
                torch.distributions.Categorical(p).sample()
            return a.item(), None

    return Wrapper()


#SINGLE EPISODE RUNNER

def run_episode(model, seed: int):
    """
    Run one episode and return detailed step-level log + summary.
    model=None → random agent.
    """
    env = WildlifeLoopEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    total = 0.0

    step_rewards   = []
    cum_rewards    = []
    battery_log    = []
    coverage_log   = []
    action_counts  = np.zeros(7, dtype=int)

    while not done and not truncated:
        if model is None:
            action = env.action_space.sample()
            _      = None
        else:
            action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, truncated, info = env.step(action)
        total += reward

        step_rewards.append(reward)
        cum_rewards.append(total)
        battery_log.append(info["battery"])
        coverage_log.append(info["coverage_pct"] * 100)
        action_counts[action] += 1

    env.close()

    summary = {
        "total_reward":   total,
        "steps":          info["step"],
        "caught":         info["threats_caught"],
        "missed":         info["threats_missed"],
        "false_alerts":   info["false_alerts"],
        "coverage_pct":   info["coverage_pct"] * 100,
        "battery_left":   info["battery"] * 100,
        "recharges":      info["recharge_count"],
        "end_reason":     info["terminated_reason"],
        "action_counts":  action_counts,
        "step_rewards":   step_rewards,
        "cum_rewards":    cum_rewards,
        "battery_log":    battery_log,
        "coverage_log":   coverage_log,
    }
    return summary


#  MULTI-EPISODE EVALUATION 

def evaluate_agent(name: str, model, n_episodes: int):
    print(f"  Evaluating {name:<12} × {n_episodes} episodes …", end="", flush=True)
    episodes = []
    for i in range(n_episodes):
        ep = run_episode(model, seed=i * 7 + 13)
        episodes.append(ep)

    rewards  = [e["total_reward"]  for e in episodes]
    steps    = [e["steps"]         for e in episodes]
    caught   = [e["caught"]        for e in episodes]
    missed   = [e["missed"]        for e in episodes]
    false_al = [e["false_alerts"]  for e in episodes]
    coverage = [e["coverage_pct"]  for e in episodes]
    battery  = [e["battery_left"]  for e in episodes]
    recharge = [e["recharges"]     for e in episodes]

    # aggregate action counts
    agg_actions = np.sum([e["action_counts"] for e in episodes], axis=0)
    total_acts  = agg_actions.sum()
    action_pct  = (agg_actions / total_acts * 100).tolist() if total_acts > 0 else [0]*7

    # best episode curves (for time-series plots)
    best_idx  = int(np.argmax(rewards))
    best_ep   = episodes[best_idx]

    result = {
        "name":           name,
        "mean_reward":    np.mean(rewards),
        "std_reward":     np.std(rewards),
        "min_reward":     np.min(rewards),
        "max_reward":     np.max(rewards),
        "mean_steps":     np.mean(steps),
        "mean_caught":    np.mean(caught),
        "mean_missed":    np.mean(missed),
        "mean_false":     np.mean(false_al),
        "mean_coverage":  np.mean(coverage),
        "mean_battery":   np.mean(battery),
        "mean_recharges": np.mean(recharge),
        "action_pct":     action_pct,
        "best_cum_rewards":  best_ep["cum_rewards"],
        "best_battery_log":  best_ep["battery_log"],
        "best_coverage_log": best_ep["coverage_log"],
        "all_rewards":    rewards,
    }

    print(f"  mean={np.mean(rewards):>8.1f}  std={np.std(rewards):>6.1f}  "
          f"caught={np.mean(caught):.1f}  cov={np.mean(coverage):.0f}%")
    return result


# PLOT STYLE

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TICK_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, fontsize=9, color=TITLE_COL, pad=6, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=TEXT_COL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=TEXT_COL)


def _legend(ax, results):
    handles = [
        plt.Line2D([0], [0], color=ALGO_CONFIG.get(r["name"], {}).get("color", "#888"),
                   linewidth=2, label=r["name"])
        for r in results
    ]
    ax.legend(handles=handles, fontsize=7, facecolor=AXES_BG,
              edgecolor=GRID_COL, labelcolor=TEXT_COL,
              loc="best", framealpha=0.8)


# INDIVIDUAL PLOTS

def plot_reward_bar(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Mean episode reward per agent (±1 std)", "Agent", "Mean Reward")

    names  = [r["name"]       for r in results]
    means  = [r["mean_reward"]for r in results]
    stds   = [r["std_reward"] for r in results]
    colors = [ALGO_CONFIG.get(n, {}).get("color", "#888") for n in names]

    bars = ax.bar(names, means, color=colors, width=0.55,
                  edgecolor=DARK_BG, linewidth=0.5, zorder=3)
    ax.errorbar(names, means, yerr=stds, fmt="none",
                color=TITLE_COL, capsize=4, linewidth=1.2, zorder=4)

    for bar, val in zip(bars, means):
        ypos = bar.get_height() + (max(means) * 0.02 if val >= 0 else -max(abs(m) for m in means) * 0.06)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:,.0f}", ha="center", va="bottom",
                fontsize=8, color=TITLE_COL, fontweight="bold")

    ax.axhline(0, color=TICK_COL, linewidth=0.6, linestyle="--")
    plt.tight_layout()
    path = os.path.join(out_dir, "random_vs_trained.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cumulative_reward(results, out_dir):
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Cumulative reward — best episode per agent", "Step", "Cumulative Reward")

    for r in results:
        col   = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        curve = r["best_cum_rewards"]
        steps = list(range(1, len(curve) + 1))
        ax.plot(steps, curve, color=col, linewidth=1.8,
                label=r["name"], alpha=0.9)

    _legend(ax, results)
    ax.axhline(0, color=TICK_COL, linewidth=0.5, linestyle=":")
    plt.tight_layout()
    path = os.path.join(out_dir, "cumulative_reward.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_action_distribution(results, out_dir):
    from environment.lifeloop_env import ACTION_NAMES
    short = ["N", "S", "E", "W", "Investigate", "Dispatch", "Recharge"]

    n     = len(results)
    x     = np.arange(len(short))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Action distribution (% of total actions)", "Action", "Usage %")

    for i, r in enumerate(results):
        col    = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, r["action_pct"], width,
               color=col, label=r["name"],
               edgecolor=DARK_BG, linewidth=0.4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8, color=TEXT_COL)
    _legend(ax, results)
    plt.tight_layout()
    path = os.path.join(out_dir, "action_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_battery(results, out_dir):
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Battery level over time — best episode per agent", "Step", "Battery %")

    for r in results:
        col  = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        bat  = [b * 100 for b in r["best_battery_log"]]
        steps = list(range(1, len(bat) + 1))
        ax.plot(steps, bat, color=col, linewidth=1.8, label=r["name"], alpha=0.9)

    ax.axhline(25, color="#E24B4A", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(5, 27, "Critical (25%)", fontsize=7, color="#E24B4A")
    ax.set_ylim(-5, 110)
    _legend(ax, results)
    plt.tight_layout()
    path = os.path.join(out_dir, "battery_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_caught_missed(results, out_dir):
    names  = [r["name"]       for r in results]
    caught = [r["mean_caught"]for r in results]
    missed = [r["mean_missed"]for r in results]
    colors = [ALGO_CONFIG.get(n, {}).get("color", "#888") for n in names]
    x      = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Threats caught vs missed per episode (mean)", "Agent", "Count")

    bars1 = ax.bar(x - 0.2, caught, 0.38, color=colors,
                   label="Caught", edgecolor=DARK_BG, linewidth=0.4)
    bars2 = ax.bar(x + 0.2, missed, 0.38, color=colors, alpha=0.35,
                   label="Missed", edgecolor=DARK_BG, linewidth=0.4, hatch="//")

    for bar, val in zip(list(bars1) + list(bars2), caught + missed):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7.5, color=TITLE_COL)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, color=TEXT_COL)
    ax.legend(fontsize=8, facecolor=AXES_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    plt.tight_layout()
    path = os.path.join(out_dir, "caught_vs_missed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_coverage(results, out_dir):
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Grid coverage over time — best episode", "Step", "Coverage %")

    for r in results:
        col  = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        cov  = r["best_coverage_log"]
        steps = list(range(1, len(cov) + 1))
        ax.plot(steps, cov, color=col, linewidth=1.8, label=r["name"], alpha=0.9)

    ax.set_ylim(0, 105)
    _legend(ax, results)
    plt.tight_layout()
    path = os.path.join(out_dir, "coverage_over_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_false_alarms(results, out_dir):
    names  = [r["name"]      for r in results]
    falses = [r["mean_false"]for r in results]
    colors = [ALGO_CONFIG.get(n, {}).get("color", "#888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=DARK_BG)
    _style_ax(ax, "Mean false alerts per episode", "Agent", "False Alerts")

    bars = ax.bar(names, falses, color=colors, width=0.5,
                  edgecolor=DARK_BG, linewidth=0.4)
    for bar, val in zip(bars, falses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=8, color=TITLE_COL, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "false_alarms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_episode_length(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=DARK_BG)

    # left: mean steps bar
    ax1 = axes[0]
    _style_ax(ax1, "Mean episode length", "Agent", "Steps")
    names  = [r["name"]      for r in results]
    steps  = [r["mean_steps"]for r in results]
    colors = [ALGO_CONFIG.get(n, {}).get("color", "#888") for n in names]
    bars   = ax1.bar(names, steps, color=colors, width=0.5,
                     edgecolor=DARK_BG, linewidth=0.4)
    ax1.axhline(500, color=TICK_COL, linewidth=0.8,
                linestyle="--", alpha=0.5, label="Max (500)")
    for bar, val in zip(bars, steps):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 3,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=8, color=TITLE_COL)
    ax1.set_ylim(0, 560)
    ax1.legend(fontsize=7, facecolor=AXES_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # right: reward distribution violin
    ax2 = axes[1]
    _style_ax(ax2, "Reward distribution across episodes", "Agent", "Total Reward")
    data_list = [r["all_rewards"] for r in results]
    parts     = ax2.violinplot(data_list, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"],
                       [ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
                        for r in results]):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color(TICK_COL)
    parts["cmins"].set_color(TICK_COL)
    parts["cbars"].set_color(TICK_COL)
    ax2.set_xticks(range(1, len(names) + 1))
    ax2.set_xticklabels(names, fontsize=8, color=TEXT_COL)
    ax2.axhline(0, color=TICK_COL, linewidth=0.5, linestyle=":")

    plt.tight_layout()
    path = os.path.join(out_dir, "episode_length.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dashboard(results, out_dir):
    """Full 3×3 dashboard combining all key plots."""
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    fig.suptitle("WildlifeLoopEnv — Full Agent Evaluation Dashboard",
                 fontsize=15, color=TITLE_COL, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.32)

    names  = [r["name"]       for r in results]
    colors_list = [ALGO_CONFIG.get(n, {}).get("color", "#888") for n in names]

    #  reward bar
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "Mean reward (±std)", "", "Reward")
    means = [r["mean_reward"] for r in results]
    stds  = [r["std_reward"]  for r in results]
    bars  = ax1.bar(names, means, color=colors_list, width=0.55,
                    edgecolor=DARK_BG, linewidth=0.4)
    ax1.errorbar(names, means, yerr=stds, fmt="none",
                 color=TITLE_COL, capsize=3, linewidth=1)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + abs(max(means))*0.02,
                 f"{val:,.0f}", ha="center", fontsize=7,
                 color=TITLE_COL, fontweight="bold")
    ax1.axhline(0, color=TICK_COL, linewidth=0.5, linestyle="--")
    ax1.tick_params(axis="x", labelsize=7)

    #  cumulative reward curves \
    ax2 = fig.add_subplot(gs[0, 1:])
    _style_ax(ax2, "Cumulative reward — best episode", "Step", "Reward")
    for r in results:
        col   = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        curve = r["best_cum_rewards"]
        ax2.plot(range(1, len(curve)+1), curve,
                 color=col, linewidth=1.6, label=r["name"], alpha=0.9)
    ax2.axhline(0, color=TICK_COL, linewidth=0.5, linestyle=":")
    _legend(ax2, results)

    #  caught vs missed
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Caught vs missed", "", "Count / episode")
    caught = [r["mean_caught"]for r in results]
    missed = [r["mean_missed"]for r in results]
    x = np.arange(len(names))
    ax3.bar(x-0.2, caught, 0.36, color=colors_list, edgecolor=DARK_BG, linewidth=0.3)
    ax3.bar(x+0.2, missed, 0.36, color=colors_list, alpha=0.3,
            edgecolor=DARK_BG, linewidth=0.3, hatch="//")
    ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=7)
    from matplotlib.lines import Line2D
    ax3.legend(handles=[Line2D([0],[0],color="white",lw=2,label="Caught"),
                         Line2D([0],[0],color="white",lw=2,alpha=0.3,label="Missed")],
               fontsize=7, facecolor=AXES_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    #  action distribution 
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Action distribution", "Action", "Usage %")
    short = ["N","S","E","W","Inv","Dis","Rec"]
    xx    = np.arange(7)
    n     = len(results)
    w     = 0.8 / n
    for i, r in enumerate(results):
        col = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        ax4.bar(xx + (i - n/2 + 0.5)*w, r["action_pct"], w,
                color=col, alpha=0.85, edgecolor=DARK_BG, linewidth=0.3)
    ax4.set_xticks(xx); ax4.set_xticklabels(short, fontsize=7)
    _legend(ax4, results)

    # coverage over time
    ax5 = fig.add_subplot(gs[1, 2])
    _style_ax(ax5, "Coverage over time", "Step", "Coverage %")
    for r in results:
        col = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        cov = r["best_coverage_log"]
        ax5.plot(range(1, len(cov)+1), cov,
                 color=col, linewidth=1.4, label=r["name"])
    ax5.set_ylim(0, 105)
    _legend(ax5, results)

    #  battery 
    ax6 = fig.add_subplot(gs[2, 0])
    _style_ax(ax6, "Battery over time", "Step", "Battery %")
    for r in results:
        col = ALGO_CONFIG.get(r["name"], {}).get("color", "#888")
        bat = [b*100 for b in r["best_battery_log"]]
        ax6.plot(range(1, len(bat)+1), bat,
                 color=col, linewidth=1.4, label=r["name"])
    ax6.axhline(25, color="#E24B4A", linewidth=0.7, linestyle="--", alpha=0.6)
    ax6.set_ylim(-5, 110)
    _legend(ax6, results)

    #  false alarms
    ax7 = fig.add_subplot(gs[2, 1])
    _style_ax(ax7, "False alerts per episode", "", "Mean false alerts")
    falses = [r["mean_false"]for r in results]
    bars   = ax7.bar(names, falses, color=colors_list, width=0.5,
                     edgecolor=DARK_BG, linewidth=0.4)
    for bar, val in zip(bars, falses):
        ax7.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.1,
                 f"{val:.1f}", ha="center", fontsize=7,
                 color=TITLE_COL, fontweight="bold")
    ax7.tick_params(axis="x", labelsize=7)

    #violin 
    ax8 = fig.add_subplot(gs[2, 2])
    _style_ax(ax8, "Reward distribution", "", "Total reward")
    parts = ax8.violinplot([r["all_rewards"] for r in results],
                           showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors_list):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color(TICK_COL)
    parts["cmins"].set_color(TICK_COL)
    parts["cbars"].set_color(TICK_COL)
    ax8.set_xticks(range(1, len(names)+1))
    ax8.set_xticklabels(names, fontsize=7)
    ax8.axhline(0, color=TICK_COL, linewidth=0.5, linestyle=":")

    path = os.path.join(out_dir, "full_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


#  SAVE CSV

def save_csv(results, out_dir):
    from environment.lifeloop_env import ACTION_NAMES
    rows = []
    for r in results:
        row = {
            "algorithm":      r["name"],
            "mean_reward":    round(r["mean_reward"],  2),
            "std_reward":     round(r["std_reward"],   2),
            "min_reward":     round(r["min_reward"],   2),
            "max_reward":     round(r["max_reward"],   2),
            "mean_steps":     round(r["mean_steps"],   1),
            "mean_caught":    round(r["mean_caught"],  2),
            "mean_missed":    round(r["mean_missed"],  2),
            "mean_false_alerts": round(r["mean_false"], 2),
            "mean_coverage_pct": round(r["mean_coverage"], 1),
            "mean_battery_left": round(r["mean_battery"], 1),
            "mean_recharges": round(r["mean_recharges"], 2),
        }
        for i, aname in enumerate(ACTION_NAMES):
            row[f"action_pct_{aname.replace(' ','_')}"] = round(r["action_pct"][i], 1)
        rows.append(row)

    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, "summary_table.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # pretty print
    print(f"\n{'='*75}")
    print("EVALUATION SUMMARY")
    print(f"{'='*75}")
    print(df[["algorithm","mean_reward","std_reward","mean_caught",
              "mean_missed","mean_false_alerts","mean_coverage_pct",
              "mean_battery_left"]].to_string(index=False))


# MAIN 

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("WildlifeLoopEnv — Agent Evaluation")
    print(f"Episodes per agent : {N_EPISODES}")
    print("=" * 60)

    results = []

    # random agent
    results.append(evaluate_agent("Random", None, N_EPISODES))

    # trained agents
    for algo in ["DQN", "REINFORCE", "PPO", "A2C"]:
        path = find_best_model(algo)
        if path is None:
            print(f"  [SKIP] {algo} — no model found in {MODEL_DIR}/{algo.lower()}/")
            continue
        print(f"  Loading {algo} from {path}")
        try:
            if algo == "REINFORCE":
                model = load_reinforce(path)
            else:
                model = load_sb3_model(algo, path)
            results.append(evaluate_agent(algo, model, N_EPISODES))
        except Exception as e:
            print(f"  [ERROR] {algo}: {e}")

    if len(results) < 2:
        print("\n[ERROR] Need at least 2 agents to compare. Train models first.")
        return

    print(f"\n{'='*60}")
    print("Generating plots …")
    print(f"{'='*60}")

    save_csv(results, OUT_DIR)
    plot_reward_bar(results, OUT_DIR)
    plot_cumulative_reward(results, OUT_DIR)
    plot_action_distribution(results, OUT_DIR)
    plot_battery(results, OUT_DIR)
    plot_caught_missed(results, OUT_DIR)
    plot_coverage(results, OUT_DIR)
    plot_false_alarms(results, OUT_DIR)
    plot_episode_length(results, OUT_DIR)
    plot_dashboard(results, OUT_DIR)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()