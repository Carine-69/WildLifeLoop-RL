"""
analyse_results.py

1. Evaluates every checkpoint for DQN, PPO, A2C
2. Extracts hyperparameters from each best model's zip file
3. Produces:
   - results/all_checkpoints.csv      — full table of every checkpoint
   - results/best_models.csv          — one row per algorithm (best only)
   - results/hyperparameters.csv      — hyperparameters of each best model
   - results/full_report.txt          — human-readable summary report
   - results/all_checkpoints.png      — reward curve per algo
   - results/best_comparison.png      — bar chart of best rewards
"""

import os
import re
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import DQN, PPO, A2C
from environment.lifeloop_env import WildlifeLoopEnv

MODEL_FOLDER   = "models"
RESULTS_FOLDER = "results"
EVAL_EPISODES  = 10   # episodes per checkpoint

MODEL_MAP = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

COLORS = {"dqn": "#E94560", "ppo": "#0F9B8E", "a2c": "#F5A623"}

COMMON_KEYS = [
    "learning_rate", "gamma", "batch_size",
    "buffer_size", "learning_starts",          # DQN
    "n_steps", "ent_coef", "vf_coef",          # PPO/A2C
    "clip_range",                               # PPO
    "n_epochs",                                 # PPO
    "gae_lambda",                               # PPO/A2C
    "max_grad_norm",
    "tau",                                      # DQN
    "train_freq", "target_update_interval",     # DQN
    "exploration_fraction", "exploration_final_eps",  # DQN
    "policy",
    "n_envs",
]


def extract_step(filename: str) -> int:
    match = re.search(r"_(\d+)_steps", filename)
    if match:
        return int(match.group(1))
    if "final" in filename:
        return int(1e9)
    return -1


def evaluate_checkpoint(model_path: str, model_class, n_episodes: int = EVAL_EPISODES):
    env = WildlifeLoopEnv(render_mode=None)
    model = model_class.load(model_path, device="cpu")
    rewards, caught_list, missed_list, coverage_list, false_list = [], [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total = 0.0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total += reward
        rewards.append(total)
        caught_list.append(info["threats_caught"])
        missed_list.append(info["threats_missed"])
        coverage_list.append(info["coverage_pct"] * 100)
        false_list.append(info["false_alerts"])

    env.close()
    return {
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "min_reward":    float(np.min(rewards)),
        "max_reward":    float(np.max(rewards)),
        "mean_caught":   float(np.mean(caught_list)),
        "mean_missed":   float(np.mean(missed_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "mean_false":    float(np.mean(false_list)),
    }


def extract_hyperparams(model_path: str) -> dict:
    """Read hyperparameters directly from the model's zip file."""
    params = {}
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            # SB3 stores model data in 'data' (JSON)
            if "data" in zf.namelist():
                raw = json.loads(zf.read("data").decode("utf-8"))

                # flatten nested policy_kwargs if present
                policy_kwargs = raw.get("policy_kwargs", {})
                if isinstance(policy_kwargs, dict):
                    for k, v in policy_kwargs.items():
                        params[f"policy_kwargs.{k}"] = v

                for key in COMMON_KEYS:
                    val = raw.get(key, None)
                    if val is not None:
                        # unwrap SB3's Schedule objects {"__type__":"function",...}
                        if isinstance(val, dict) and "__type__" in val:
                            val = val.get("initial_value", str(val))
                        params[key] = val

                # also grab n_envs from _n_envs
                if "_n_envs" in raw:
                    params["n_envs"] = raw["_n_envs"]

    except Exception as e:
        params["_error"] = str(e)
    return params



def evaluate_all():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    all_rows   = []   # every checkpoint → CSV
    best_data  = {}   # best per algo → plots + best CSV

    for algo, model_class in MODEL_MAP.items():
        algo_dir = os.path.join(MODEL_FOLDER, algo)
        if not os.path.isdir(algo_dir):
            print(f"[SKIP] {algo_dir} not found")
            continue

        checkpoints = sorted(
            [f for f in os.listdir(algo_dir) if f.endswith(".zip")],
            key=extract_step
        )

        best_mean, best_path, best_step = -np.inf, None, None
        steps, means, stds = [], [], []

        total = len(checkpoints)
        print(f"\n{'='*60}")
        print(f"Evaluating {algo.upper()} — {total} checkpoints × {EVAL_EPISODES} episodes")
        print(f"{'='*60}")

        for i, fname in enumerate(checkpoints, 1):
            step = extract_step(fname)
            path = os.path.join(algo_dir, fname)

            try:
                metrics = evaluate_checkpoint(path, model_class)
            except Exception as e:
                print(f"  [{i:>3}/{total}] SKIP {fname}: {e}")
                continue

            row = {
                "algorithm":   algo.upper(),
                "checkpoint":  fname,
                "step":        step,
                "model_path":  path,
                **metrics,
            }
            all_rows.append(row)

            # skip final (step=1e9) from plot x-axis
            if step < 1_100_000:
                steps.append(step)
                means.append(metrics["mean_reward"])
                stds.append(metrics["std_reward"])

            marker = " ◀ BEST" if metrics["mean_reward"] > best_mean else ""
            if metrics["mean_reward"] > best_mean:
                best_mean = metrics["mean_reward"]
                best_path = path
                best_step = step

            print(f"  [{i:>3}/{total}] step={step:>10,}  "
                  f"mean={metrics['mean_reward']:>8.2f}  "
                  f"caught={metrics['mean_caught']:.1f}  "
                  f"cov={metrics['mean_coverage']:.0f}%{marker}")

        best_data[algo] = {
            "steps": steps, "means": means, "stds": stds,
            "best_path": best_path, "best_step": best_step, "best_mean": best_mean,
        }
        print(f"\n  ✔ Best {algo.upper()}: step={best_step:,}  "
              f"mean={best_mean:.2f}  → {best_path}")

    return all_rows, best_data


def save_csvs(all_rows: list, best_data: dict):

    df_all = pd.DataFrame(all_rows)
    df_all = df_all.sort_values(["algorithm", "step"]).reset_index(drop=True)
    path_all = os.path.join(RESULTS_FOLDER, "all_checkpoints.csv")
    df_all.to_csv(path_all, index=False, float_format="%.4f")
    print(f"  Saved: {path_all}  ({len(df_all)} rows)")


    best_rows = []
    for algo, data in best_data.items():
  
        mask = (df_all["algorithm"] == algo.upper()) & (df_all["step"] == data["best_step"])
        match = df_all[mask]
        if not match.empty:
            best_rows.append(match.iloc[0].to_dict())
    df_best = pd.DataFrame(best_rows)
    path_best = os.path.join(RESULTS_FOLDER, "best_models.csv")
    df_best.to_csv(path_best, index=False, float_format="%.4f")
    print(f"  Saved: {path_best}")

  
    hp_rows = []
    for algo, data in best_data.items():
        if data["best_path"] is None:
            continue
        hp = extract_hyperparams(data["best_path"])
        hp_rows.append({
            "algorithm":  algo.upper(),
            "best_step":  data["best_step"],
            "best_mean":  round(data["best_mean"], 2),
            "model_path": data["best_path"],
            **hp,
        })
    df_hp = pd.DataFrame(hp_rows)
    path_hp = os.path.join(RESULTS_FOLDER, "hyperparameters.csv")
    df_hp.to_csv(path_hp, index=False)
    print(f"  Saved: {path_hp}")

    return df_all, df_best, df_hp



def save_report(df_all: pd.DataFrame, df_best: pd.DataFrame,
                df_hp: pd.DataFrame, best_data: dict):
    path = os.path.join(RESULTS_FOLDER, "full_report.txt")
    sep  = "=" * 70

    with open(path, "w") as f:
        f.write(f"LIFELOOP RL — FULL EVALUATION REPORT\n{sep}\n\n")

    
        f.write("BEST MODEL PER ALGORITHM\n" + "-" * 40 + "\n")
        for _, row in df_best.iterrows():
            f.write(
                f"  {row['algorithm']:<5}  "
                f"step={int(row['step']):>10,}  "
                f"mean_reward={row['mean_reward']:>8.2f}  "
                f"std={row['std_reward']:>6.2f}  "
                f"caught={row['mean_caught']:.1f}  "
                f"missed={row['mean_missed']:.1f}  "
                f"coverage={row['mean_coverage']:.0f}%  "
                f"false_alerts={row['mean_false']:.1f}\n"
            )
        f.write("\n")

        f.write("HYPERPARAMETERS OF BEST MODELS\n" + "-" * 40 + "\n")
        for _, row in df_hp.iterrows():
            f.write(f"\n  [{row['algorithm']}]  step={int(row['best_step']):,}"
                    f"  mean_reward={row['best_mean']}\n")
            skip = {"algorithm", "best_step", "best_mean", "model_path"}
            for col in df_hp.columns:
                if col in skip:
                    continue
                val = row[col]
                if pd.notna(val):
                    f.write(f"    {col:<35} {val}\n")
        f.write("\n")


        f.write("PER-ALGORITHM SUMMARY ACROSS ALL CHECKPOINTS\n" + "-" * 40 + "\n")
        for algo in df_all["algorithm"].unique():
            sub = df_all[df_all["algorithm"] == algo]["mean_reward"]
            f.write(
                f"  {algo:<5}  checkpoints={len(sub)}  "
                f"min={sub.min():.1f}  max={sub.max():.1f}  "
                f"mean={sub.mean():.1f}  median={sub.median():.1f}  "
                f"std={sub.std():.1f}\n"
            )
        f.write("\n")

   
        f.write("FULL CHECKPOINT TABLE\n" + "-" * 40 + "\n")
        col_order = ["algorithm", "step", "mean_reward", "std_reward",
                     "mean_caught", "mean_missed", "mean_coverage",
                     "mean_false", "checkpoint"]
        cols = [c for c in col_order if c in df_all.columns]
        f.write(df_all[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        f.write("\n")

    print(f"  Saved: {path}")


# PLOTS

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


def save_plots(best_data: dict, df_all: pd.DataFrame):
    _style()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("LifeLoop RL — Model Evaluation Results",
                 fontsize=16, color="#e6edf3", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    
    ax1 = fig.add_subplot(gs[0, :2])
    for algo, data in best_data.items():
        if not data["steps"]:
            continue
        steps = np.array(data["steps"])
        means = np.array(data["means"])
        stds  = np.array(data["stds"])
        col   = COLORS[algo]
        ax1.fill_between(steps, means - stds, means + stds, color=col, alpha=0.15)
        ax1.plot(steps, means, color=col, linewidth=2, label=algo.upper())
      
        if len(means) >= 10:
            w    = max(5, len(means) // 20)
            roll = np.convolve(means, np.ones(w) / w, mode="valid")
            ax1.plot(steps[w - 1:], roll, color=col, linewidth=1,
                     linestyle="--", alpha=0.7)
   
        bx, by = data["best_step"], data["best_mean"]
        if bx < 1_100_000:
            ax1.scatter([bx], [by], color=col, s=150, zorder=5,
                        marker="*", edgecolors="white", linewidths=0.5)
    ax1.set_title("Mean Reward per Checkpoint (dashed = rolling avg, ★ = best)",
                  fontsize=11)
    ax1.set_xlabel("Training Steps"); ax1.set_ylabel("Mean Episode Reward")
    ax1.set_xlim(0, 1_000_000)
    ax1.legend(fontsize=10); ax1.grid(True)

   
    ax2 = fig.add_subplot(gs[0, 2])
    algos = list(best_data.keys())
    bests = [best_data[a]["best_mean"] for a in algos]
    bars  = ax2.bar(algos, bests, color=[COLORS[a] for a in algos], width=0.5)
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels([a.upper() for a in algos], fontsize=11)
    for bar, val in zip(bars, bests):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(bests) * 0.01,
                 f"{val:.0f}", ha="center", color="#e6edf3",
                 fontsize=11, fontweight="bold")
    ax2.set_title("Best Mean Reward", fontsize=11)
    ax2.set_ylabel("Mean Episode Reward"); ax2.grid(True, axis="y")

    ax3 = fig.add_subplot(gs[1, 0])
    df_b = df_all.copy()
    best_steps = {a.upper(): d["best_step"] for a, d in best_data.items()}
    df_b = df_b[df_b.apply(
        lambda r: r["step"] == best_steps.get(r["algorithm"]), axis=1
    )]
    x     = np.arange(len(df_b))
    width = 0.35
    ax3.bar(x - width/2, df_b["mean_caught"], width,
            color=[COLORS[a.lower()] for a in df_b["algorithm"]], alpha=0.9,
            label="Caught")
    ax3.bar(x + width/2, df_b["mean_missed"], width,
            color=[COLORS[a.lower()] for a in df_b["algorithm"]], alpha=0.4,
            label="Missed", hatch="//")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_b["algorithm"].tolist(), fontsize=11)
    ax3.set_title("Best Model: Threats Caught vs Missed", fontsize=11)
    ax3.set_ylabel("Mean per Episode"); ax3.grid(True, axis="y")
    ax3.legend(fontsize=9)


    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(df_b["algorithm"].tolist(), df_b["mean_coverage"],
            color=[COLORS[a.lower()] for a in df_b["algorithm"]], width=0.5)
    ax4.set_xticks(range(len(df_b)))
    ax4.set_xticklabels(df_b["algorithm"].tolist(), fontsize=11)
    ax4.set_title("Best Model: Grid Coverage %", fontsize=11)
    ax4.set_ylabel("Coverage (%)"); ax4.grid(True, axis="y")
    ax4.set_ylim(0, 100)
    for i, (_, row) in enumerate(df_b.iterrows()):
        ax4.text(i, row["mean_coverage"] + 1,
                 f"{row['mean_coverage']:.1f}%",
                 ha="center", color="#e6edf3", fontsize=10)


    ax5 = fig.add_subplot(gs[1, 2])
    data_list  = [df_all[df_all["algorithm"] == a.upper()]["mean_reward"].tolist()
                  for a in best_data]
    algo_names = [a.upper() for a in best_data]
    parts = ax5.violinplot(data_list, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [COLORS[a] for a in best_data]):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color("#8b949e")
    parts["cmins"].set_color("#8b949e")
    parts["cbars"].set_color("#8b949e")
    ax5.set_xticks(range(1, len(algo_names) + 1))
    ax5.set_xticklabels(algo_names, fontsize=11)
    ax5.set_title("Reward Distribution (all checkpoints)", fontsize=11)
    ax5.set_ylabel("Mean Episode Reward"); ax5.grid(True, axis="y")

    out = os.path.join(RESULTS_FOLDER, "evaluation_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")




def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    
    all_rows, best_data = evaluate_all()

    print(f"\n{'='*60}")
    print("Saving result files …")
    print(f"{'='*60}")

   
    df_all, df_best, df_hp = save_csvs(all_rows, best_data)

 
    save_report(df_all, df_best, df_hp, best_data)

 
    save_plots(best_data, df_all)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for algo, data in best_data.items():
        print(f"  {algo.upper():<5}  best_step={data['best_step']:>10,}  "
              f"mean_reward={data['best_mean']:>8.2f}  → {data['best_path']}")

    print(f"\nOutput files in: {RESULTS_FOLDER}/")
    print(f"  • all_checkpoints.csv    — every checkpoint evaluated")
    print(f"  • best_models.csv        — best checkpoint per algorithm")
    print(f"  • hyperparameters.csv    — hyperparams of each best model")
    print(f"  • full_report.txt        — human-readable full report")
    print(f"  • evaluation_dashboard.png — all plots in one figure")


if __name__ == "__main__":
    main()