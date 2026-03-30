"""
main.py
========
Entry point — runs the best performing model and produces:
  1. Terminal verbose output (step-by-step + episode summary)
  2. An animated GIF saved to results/best_agent_demo.gif
  3. A static PNG contact sheet → results/best_agent_frames.png

No live window — no freezing. Demo plays in any browser/image viewer.

Usage:
    python main.py                   # PPO best model (overall best)
    python main.py --algo dqn
    python main.py --algo reinforce
    python main.py --episodes 3
    python main.py --no-gif          # terminal only
"""

import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from environment.lifeloop_env import (
    WildlifeLoopEnv, ACTION_NAMES, DETECT_R, POACHER_SPOTS, GRID
)

BEST_MODELS = {
    "ppo":       "models/ppo/wildlife_ppo_final.zip",
    "dqn":       "models/dqn/wildlife_dqn_final.zip",
    "a2c":       "models/a2c/wildlife_a2c_final.zip",
    "reinforce": "models/reinforce/wildlife_reinforce_final.pt",
}
OVERALL_BEST = "ppo"
RESULTS_DIR  = "results"


def load_model(algo, path):
    if algo == "reinforce":
        return _load_reinforce(path)
    from stable_baselines3 import DQN, PPO, A2C
    return {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo].load(path, device="cpu")


def _load_reinforce(path):
    import torch, torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(22,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,7))
        def forward(self,x): return torch.softmax(self.net(x),dim=-1)
    ckpt = torch.load(path, map_location="cpu")
    net  = Net(); net.load_state_dict(ckpt["policy_state"]); net.eval()
    class W:
        def predict(self,obs,deterministic=True):
            import torch
            x=torch.FloatTensor(obs).unsqueeze(0); p=net(x)
            return (p.argmax(-1) if deterministic else torch.distributions.Categorical(p).sample()).item(), None
    return W()


def draw_frame(env, step, action, reward, total, fig, ax, algo, ep):
    ax.clear(); ax.set_facecolor("#0d1f0d")
    ax.set_xlim(-0.5,GRID-0.5); ax.set_ylim(-0.5,GRID-0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for zid in env._visited:
        ax.add_patch(mpatches.Rectangle((zid%GRID-0.5,zid//GRID-0.5),1,1,
            linewidth=0,facecolor="#1a3d1a",alpha=0.55,zorder=1))
    for i in range(GRID+1):
        ax.axhline(i-0.5,color="#1f3d1f",lw=0.3,zorder=2)
        ax.axvline(i-0.5,color="#1f3d1f",lw=0.3,zorder=2)
    for i,(px,py) in enumerate(POACHER_SPOTS):
        col="#ff2222" if env._poacher_on[i] else "#444"
        ax.add_patch(mpatches.Circle((px,py),0.32,color=col,alpha=0.65,zorder=3))
        ax.text(px,py,"P",ha="center",va="center",fontsize=6,color="white",fontweight="bold",zorder=4)
    for i,(ax_,ay) in enumerate(env._animals):
        ax.scatter(ax_,ay,s=45,color=plt.cm.RdYlGn(1-env._anomaly[i]),zorder=5,edgecolors="white",linewidths=0.3)
    rx,ry=env._ranger
    ax.scatter(rx,ry,s=170,color="#00cfff",zorder=7,marker="^",edgecolors="white",linewidths=0.7)
    ax.add_patch(mpatches.Circle((rx,ry),DETECT_R,fill=False,edgecolor="#00cfff",lw=0.7,linestyle="--",alpha=0.4,zorder=6))
    batt=float(np.clip(env._battery,0,1))
    col="#44ff44" if batt>0.5 else "#ffaa00" if batt>0.25 else "#ff3333"
    ax.add_patch(mpatches.Rectangle((-0.4,-0.44),batt*(GRID-0.2),0.16,color=col,alpha=0.85,zorder=8))
    cov=len(env._visited)/(GRID*GRID)*100
    ax.set_title(
        f"{algo.upper()} ep{ep} step={step}  {ACTION_NAMES[action]}\n"
        f"r={reward:+.1f} total={total:+.0f} bat={batt*100:.0f}% caught={env._caught} cov={cov:.0f}%",
        fontsize=6.5,color="#c9d1d9",pad=3,fontfamily="monospace")
    fig.canvas.draw()
    w,h=fig.canvas.get_width_height()
    frame=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h,w,4)
    return frame[:,:,:3].copy()


def print_header(algo, path):
    print("="*62)
    print("  LIFELOOP — WILDLIFE PROTECTION RL AGENT")
    print("="*62)
    print(f"  Algorithm  : {algo.upper()}")
    print(f"  Model      : {path}")
    print(f"  Mission    : Patrol 10x10 reserve, catch poachers,")
    print(f"               protect animals, manage battery")
    print(f"  Obs space  : 22-dim  |  Actions: 7  |  Max steps: 500")
    print("="*62+"\n")


def run(model, algo, n_episodes, save_gif):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,6), facecolor="#0d1117")
    fig.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.04)
    gif_frames  = []
    key_frames  = []
    KEY_STEPS   = {1,50,100,200,300,400,499}

    for ep in range(1, n_episodes+1):
        env = WildlifeLoopEnv(render_mode=None)
        obs,_ = env.reset()
        done=truncated=False; total=0.0; t0=time.time()
        print(f"\n  ── Episode {ep}/{n_episodes} {'─'*40}")

        while not done and not truncated:
            action,_ = model.predict(obs, deterministic=True)
            obs,reward,done,truncated,info = env.step(action)
            total += reward
            print(f"  step={info['step']:>4}  {ACTION_NAMES[action]:<16}  "
                  f"r={reward:>+7.2f}  bat={info['battery']*100:>5.1f}%  "
                  f"caught={info['threats_caught']}  cov={info['coverage_pct']*100:>4.0f}%")
            frame = draw_frame(env,info["step"],action,reward,total,fig,ax,algo,ep)
            if save_gif: gif_frames.append(frame)
            if info["step"] in KEY_STEPS: key_frames.append((ep,info["step"],frame.copy()))

        env.close()
        elapsed = time.time()-t0
        print(f"\n  ── Episode {ep} Summary ──")
        print(f"  Total Reward   : {total:.2f}")
        print(f"  Steps          : {info['step']}")
        print(f"  Threats Caught : {info['threats_caught']}")
        print(f"  Threats Missed : {info['threats_missed']}")
        print(f"  False Alerts   : {info['false_alerts']}")
        print(f"  Recharges      : {info['recharge_count']}/3")
        print(f"  Coverage       : {info['coverage_pct']*100:.1f}%")
        print(f"  Battery Left   : {info['battery']*100:.1f}%")
        print(f"  End reason     : {info['terminated_reason']}")
        print(f"  Wall time      : {elapsed:.1f}s")

    plt.close(fig)

    if save_gif and gif_frames:
        try:
            import imageio
            path = os.path.join(RESULTS_DIR, f"{algo}_best_agent_demo.gif")
            imageio.mimwrite(path, gif_frames[::3], fps=10, loop=0)
            print(f"\n  Animated GIF → {path}")
        except ImportError:
            print("\n  [INFO] pip install imageio  to save GIF")

    if key_frames:
        ncols=7; nrows=int(np.ceil(len(key_frames)/ncols))
        fig2,axes=plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*3.2),facecolor="#0d1117")
        fig2.suptitle(f"{algo.upper()} Best Agent — Key Steps",fontsize=11,color="#e6edf3",fontweight="bold")
        flat=np.array(axes).flatten() if hasattr(axes,"flatten") else [axes]
        for i,(ep,s,fr) in enumerate(key_frames):
            flat[i].imshow(fr); flat[i].set_title(f"ep{ep} s{s}",fontsize=7,color="#8b949e"); flat[i].axis("off")
        for j in range(len(key_frames),len(flat)): flat[j].axis("off")
        plt.tight_layout()
        path2=os.path.join(RESULTS_DIR,f"{algo}_best_agent_frames.png")
        fig2.savefig(path2,dpi=130,bbox_inches="tight",facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"  Contact sheet → {path2}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--algo",       default=OVERALL_BEST, choices=list(BEST_MODELS.keys()))
    p.add_argument("--episodes",   type=int, default=3)
    p.add_argument("--no-gif",     action="store_true")
    p.add_argument("--model-path", default=None)
    args=p.parse_args()

    algo=args.algo.lower()
    path=args.model_path or BEST_MODELS[algo]
    if not os.path.exists(path):
        print(f"[ERROR] Model not found: {path}")
        sys.exit(1)

    print_header(algo, path)
    print(f"  Loading {algo.upper()} … ", end="", flush=True)
    model=load_model(algo,path)
    print("done.\n")

    run(model, algo, args.episodes, save_gif=not args.no_gif)
    print(f"\n  All outputs → {RESULTS_DIR}/")

if __name__=="__main__":
    main()