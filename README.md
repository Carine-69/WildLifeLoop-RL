# WildlifeLoop RL — Anti-Poaching Ranger Agent

> **Mission-Based Reinforcement Learning | ALU Machine Learning Techniques II**
> Autonomous drone ranger trained to patrol a wildlife reserve, intercept poachers, and monitor animal welfare using Deep Q-Networks, PPO, A2C, and REINFORCE.

##  Project Overview

WildlifeLoop RL simulates an **autonomous drone ranger** patrolling a 10×10 wildlife reserve grid. The agent must:

- **Patrol** the reserve to maximize zone coverage
- **Detect and intercept** active poachers at 4 fixed hotspots
- **Monitor animal welfare** and respond to distress signals
- **Manage battery** to avoid premature episode termination
- **Dispatch alerts** to human rangers with minimal false alarms

Three RL algorithms are compared: **DQN**, **PPO**and **REINFORCE** — each tuned across 10 hyperparameter configurations (40 models total).

**Best model: DQN

##  Environment

| Property | Value |
|---|---|
| Grid size | 10 × 10 |
| Max steps per episode | 500 |
| Action space | Discrete(7) |
| Observation space | Box(18,) float32 — normalised to [0, 1] |
| Animals | 8 (random positions, stochastic anomaly scores) |
| Poacher hotspots | 4 (fixed locations, stochastic activation) |
| Detection radius | 2.0 grid units |

### Actions

| ID | Action | Battery Cost |
|---|---|---|
| 0 | Move North | −0.003 |
| 1 | Move South | −0.003 |
| 2 | Move East | −0.003 |
| 3 | Move West | −0.003 |
| 4 | Investigate | −0.005 |
| 5 | Dispatch Alert | −0.004 |
| 6 | Recharge | +0.020 |

### Reward Structure

```
R_step         = −0.10   (per-step efficiency penalty)
R_poacher      = +33.0   per poacher caught (capped at 40.0 per investigate)
R_welfare      = +5.0    per distressed animal treated (anomaly > 0.7)
R_patrol       = +1.0    per new zone entered (max 25 zones)
R_alert_ok     = +15.0   successful dispatch near active poacher
R_false_alarm  = −6.0    dispatch with no nearby threat
R_missed       = −10.0   poacher exits undetected
R_battery_term = −20.0   terminal: battery depleted
R_battery_cont = −0.20 × (1 − battery)   smooth low-charge penalty
```

### Observation Space (18 dimensions)

| Index | Feature | Description |
|---|---|---|
| 0–1 | Ranger position | Normalised (x, y) coordinates |
| 2–3 | Nearest animal | Position of closest animal |
| 4 | Anomaly score | Welfare distress of nearest animal |
| 5 | Acoustic sensor | Binary: poacher nearby or random noise |
| 6 | Vibration sensor | Continuous footfall reading |
| 7 | Pressure sensor | Continuous ground pressure |
| 8–11 | Poacher distances | Normalised distance to each hotspot |
| 12 | Battery | Current charge as fraction of max |
| 13 | Active threat ratio | Fraction of poachers currently active |
| 14 | Coverage | Fraction of grid zones visited |
| 15 | Time remaining | Episode progress signal |
| 16 | Nearest active poacher | Distance to closest active threat |
| 17 | False alert rate | Running false-alarm ratio |

---

## Project Structure

```
carine_umugabekazi_rl_summative/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   └── rendering.py         # Pygame 2D visualisation GUI
├── training/
│   └── train_all.py         # All 4 algorithms × 10 runs + plots
├── models/
│   ├── dqn/
│   │   └── best_dqn.zip     # Best DQN model (SB3)
│   └── pg/
│       ├── best_ppo.zip     # Best PPO model (SB3)
│       ├── best_a2c.zip     # Best A2C model (SB3)
│       └── best_reinforce.zip
├── plots/                   # All training plots (PNG)
├── results/                 # Hyperparameter CSVs per algorithm
├── main.py                  # Entry point: eval / GUI / API
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/carine-umugabekazi/carine_umugabekazi_rl_summative.git
cd carine_umugabekazi_rl_summative

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, pip

---

## 🚀 Quick Start

### Run the best agent (headless evaluation)

```bash
python main.py
```

### Run with pygame visualisation

```bash
python main.py --gui
```

### Evaluate a specific model

```bash
python main.py --model best_dqn --episodes 20
python main.py --model best_ppo --episodes 10
python main.py --model best_a2c --episodes 10
```

### Random agent demo (no model — visualization only)

```bash
python environment/rendering.py --random
```

---

## Training

Run all 30 training experiments (10 per algorithm):

```bash
python training/
```

This will:
- Train DQN, REINFORCE and  PPO each with 10 hyperparameter configurations
- Save the best model per algorithm to `models/`
- Export per-run metrics to `results/*.csv`
- Generate all plots to `plots/`

**Estimated training time:** ~30 minutes on a modern CPU (5 000,000 steps per run)

To run a faster demo with fewer steps, edit `TOTAL_STEPS` in `training/train_all.py`:

```python
TOTAL_STEPS = 20_000   # fast demo
TOTAL_STEPS = 80_000   # full training (default)
```

---

## Evaluation & Visualization

### Pygame GUI Controls

| Key | Action |
|---|---|
| `R` | Reset episode |
| `Q` | Quit |

The GUI shows:
- 🟢 Ranger position with detection ring
- 🟡 Animals (colour indicates anomaly level — orange = distressed)
- 🔴 Active poacher hotspots (triangles)
- ⬛ Visited zones (shaded)
- 🟢 Movement trail
- Side panel: battery, coverage, reward, action taken, legend

### Save a static screenshot

```bash
python environment/rendering.py --screenshot
# Saves to plots/env_screenshot.png
```

---

## 📊 Results Summary

| Algorithm | Best Mean Reward | Convergence Episode | Poachers Caught | False Alerts |
|---|---|---|---|---|
| **PPO** | **1,237.8** | ~42 | **3.4** | 0.2 |
| DQN | 856.9 | ~51 | 2.8 | 0.4 |
| A2C | 534.7 | ~61 | 1.9 | 0.8 |
| REINFORCE | 423.6 | ~68 | 1.5 | 1.1 |

### Generalisation (20 unseen seeds)

| Algorithm | Mean Reward | Std |
|---|---|---|
| PPO | 1,198.3 | ±187.4 |
| DQN | 812.4 | ±198.3 |
| REINFORCE | 334.6 | ±312.7 |

> PPO's clipped surrogate objective makes it the most stable and generalisable algorithm for this stochastic, sparse-reward environment.





## Algorithms

### DQN (Deep Q-Network)
Off-policy, value-based. Learns Q(s, a) via Bellman updates on a replay buffer. Fast early learning due to experience reuse. Best config: `lr=1e-3, γ=0.99, buffer=50k, batch=64`.

### PPO (Proximal Policy Optimisation)
On-policy, policy gradient. Clipped surrogate objective prevents catastrophic policy updates. Most stable and highest-performing algorithm. Best config: `lr=3e-4, γ=0.99, clip=0.2, epochs=10`.

### REINFORCE
Monte Carlo policy gradient. High variance, no bootstrapping. Implemented via PPO with `clip_range=1.0, n_epochs=1, gae_lambda=1.0`. Simple baseline for policy gradient comparison.

### A2C (Advantage Actor-Critic)
Synchronous actor-critic. Reduces gradient variance vs REINFORCE via value baseline. Faster per-step updates (n_steps=5) but benefits from larger rollouts in sparse-reward settings. Best config: `lr=7e-4, n_steps=20`.

---

## Author

**Carine Umugabekazi**
African Leadership University — Machine Learning Techniques II, 2026 January Term


*Built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [Gymnasium](https://gymnasium.farama.org/), [Pygame](https://www.pygame.org/), and [FastAPI](https://fastapi.tiangolo.com/).*
