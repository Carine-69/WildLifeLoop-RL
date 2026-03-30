"""
WildlifeLoopEnv — Wildlife Reserve Protection
RL environment where a ranger drone must:
  - Patrol a 10×10 reserve grid
  - Detect and intercept poachers at known hot-spots
  - Monitor animal welfare (anomaly scores)
  - Dispatch verified alerts to ground teams
  - Manage drone battery responsibly

Fixes over original:
  - Recharge is capped (3× per episode) and penalised beyond cap
  - False alarm penalty escalates with repeated false dispatches
  - Patrol reward increased + stepping penalty stronger
  - Catching poachers gives scaled proximity bonus
  - Episode ends immediately if poacher escapes undetected 3× (missed)
  - Observation space extended to 22 dims (adds recharge count,
    false-alarm rate, active threat count, steps remaining)
  - render_mode="rgb_array" now works — returns numpy frame
    via matplotlib Agg (no window opened)
  - Full info dict for analysis scripts
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

# CONSTANTS 
GRID       = 10
MAX_STEPS  = 500
N_ANIMALS  = 8
N_POACHERS = 4
DETECT_R   = 2.0          # detection radius
PATROL_CAP = 40           # max unique zones rewarded (was 25)
MAX_RECHARGES   = 3       # cap per episode
MAX_MISSES      = 3       # episode ends early if 3 poachers escape

POACHER_SPOTS = np.array(
    [[1., 1.], [8., 2.], [2., 8.], [8., 8.]], dtype=np.float32
)

#  REWARDS 
R_POACHER       =  50.0   # catching a poacher (was 30)
R_ZONE_CLEAR    =   5.0   # zone secured bonus  (was 3)
R_WELFARE       =   8.0   # animal welfare intervention (was 5)
R_PATROL        =   3.0   # new zone visited (was 1)
R_ALERT_OK      =  20.0   # correct dispatch (was 15)
R_FALSE_ALARM   = -15.0   # false dispatch    (was -6 — now costly)
R_FALSE_ESCALATE = -5.0   # extra penalty per false alarm beyond 3
R_MISSED        = -20.0   # poacher escaped   (was -10)
R_BATTERY_DIE   = -30.0   # battery hits 0    (was -20)
R_RECHARGE_OVER = -5.0    # recharging beyond cap
R_STEP          = -0.3    # per step          (was -0.1)
R_COVERAGE_BONUS =  0.5   # bonus every 10% new coverage milestone
MAX_INVESTIGATE_REWARD = 60.0

ACTION_NAMES = [
    "Move North",     # 0
    "Move South",     # 1
    "Move East",      # 2
    "Move West",      # 3
    "Investigate",    # 4 — checks nearby poachers + animals
    "Dispatch Alert", # 5 — calls ground team
    "Recharge",       # 6 — drone battery top-up
]


class WildlifeLoopEnv(gym.Env):
    """
    Wildlife reserve patrol environment.

    Observation space (22 dims):
        [0-1]   ranger x, y  (normalised 0-1)
        [2-3]   nearest animal x, y (normalised)
        [4]     nearest animal anomaly score
        [5]     acoustic sensor (binary-ish)
        [6]     vibration sensor
        [7]     pressure sensor
        [8-11]  distance to each of 4 poacher hot-spots (normalised)
        [12]    battery level
        [13]    active threat ratio (active poachers / total)
        [14]    grid coverage fraction
        [15]    time remaining fraction
        [16]    distance to nearest active poacher (normalised)
        [17]    false alarm rate so far
        [18]    recharge count fraction (recharges used / cap)
        [19]    threats caught fraction
        [20]    threats missed fraction
        [21]    current step fraction (how far into episode)

    Action space: Discrete(7)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # expose constants for rendering / analysis scripts
        self.POACHER_SPOTS = POACHER_SPOTS
        self.MAX_STEPS     = MAX_STEPS
        self.GRID          = GRID
        self.ACTION_NAMES  = ACTION_NAMES

        # 22-dimensional observation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(22,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        self._rng      = np.random.default_rng()
        self._renderer = None          # pygame renderer (human mode)
        self._fig      = None          # matplotlib fig  (rgb_array mode)



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed) if seed is not None \
                    else np.random.default_rng()

        # ranger starts at a random interior position
        self._ranger   = self._rng.uniform(1, GRID - 1, 2).astype(np.float32)
        self._animals  = self._rng.uniform(0, GRID, (N_ANIMALS, 2)).astype(np.float32)
        self._anomaly  = self._rng.uniform(0, 0.3, N_ANIMALS).astype(np.float32)

        # start with 1-3 poachers active for immediate threat presence
        n_start = self._rng.integers(1, 3, endpoint=True)
        self._poacher_on = np.zeros(N_POACHERS, dtype=bool)
        self._poacher_on[self._rng.choice(N_POACHERS, n_start, replace=False)] = True

        self._step           = 0
        self._battery        = 1.0
        self._recharge_count = 0
        self._visited        = {self._zone(self._ranger)}
        self._patrol_cnt     = 0
        self._n_alerts       = 0
        self._n_false        = 0
        self._caught         = 0
        self._missed         = 0
        self._cum_reward     = 0.0
        self._coverage_milestone = 0   # tracks 10% coverage milestones
        self._total_spawned  = n_start

        return self._get_obs(), {}

    # STEP 

    def step(self, action):
        action = int(action)
        assert 0 <= action <= 6

        self._step += 1
        reward = R_STEP  # base step penalty — forces purposeful action

        #  movement
        if action == 0:
            self._ranger[1] = min(self._ranger[1] + 1.0, GRID - 1)
            self._battery  -= 0.004
        elif action == 1:
            self._ranger[1] = max(self._ranger[1] - 1.0, 0.0)
            self._battery  -= 0.004
        elif action == 2:
            self._ranger[0] = min(self._ranger[0] + 1.0, GRID - 1)
            self._battery  -= 0.004
        elif action == 3:
            self._ranger[0] = max(self._ranger[0] - 1.0, 0.0)
            self._battery  -= 0.004

        # investigate
        elif action == 4:
            reward     += self._investigate()
            self._battery -= 0.006

        # dispatch alert
        elif action == 5:
            reward     += self._dispatch()
            self._battery -= 0.005

        # recharge
        elif action == 6:
            if self._recharge_count < MAX_RECHARGES:
                self._battery         = min(self._battery + 0.25, 1.0)
                self._recharge_count += 1
                # no reward for recharging — it's a necessary action, not a goal
            else:
                # penalise recharge spam beyond cap
                reward += R_RECHARGE_OVER

        # patrol reward (new zone)
        zid = self._zone(self._ranger)
        if zid not in self._visited and self._patrol_cnt < PATROL_CAP:
            self._visited.add(zid)
            self._patrol_cnt += 1
            reward += R_PATROL
        else:
            self._visited.add(zid)

        # coverage milestone bonus (every 10% of grid covered)
        coverage_pct = len(self._visited) / (GRID * GRID)
        milestone    = int(coverage_pct * 10)   # 0-10
        if milestone > self._coverage_milestone:
            reward += R_COVERAGE_BONUS * (milestone - self._coverage_milestone)
            self._coverage_milestone = milestone

        #world dynamics
        self._world_step()

        # battery drain penalty 
        reward -= (1.0 - self._battery) * 0.1  # was 0.2 — gentler background

        # termination 
        battery_dead = bool(self._battery <= 0.0)
        if battery_dead:
            reward += R_BATTERY_DIE

        too_many_missed = bool(self._missed >= MAX_MISSES)
        if too_many_missed:
            reward -= 15.0   # episode-end penalty for letting too many escape

        time_up = bool(self._step >= MAX_STEPS)

        terminated = battery_dead or too_many_missed
        truncated  = time_up and not terminated

        self._cum_reward += reward

        info = {
            "step":              int(self._step),
            "battery":           float(np.clip(self._battery, 0.0, 1.0)),
            "coverage_pct":      float(coverage_pct),
            "threats_caught":    int(self._caught),
            "threats_missed":    int(self._missed),
            "false_alerts":      int(self._n_false),
            "total_alerts":      int(self._n_alerts),
            "recharge_count":    int(self._recharge_count),
            "patrol_zones":      int(self._patrol_cnt),
            "cumulative_reward": float(self._cum_reward),
            "terminated_reason": (
                "battery_dead"    if battery_dead    else
                "too_many_missed" if too_many_missed else
                "time_up"         if truncated       else
                "running"
            ),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    #  OBSERVATION

    def _get_obs(self):
        rx, ry = self._ranger / GRID

        # nearest animal
        dists = np.linalg.norm(self._animals - self._ranger, axis=1)
        ni    = int(np.argmin(dists))
        nax, nay = self._animals[ni] / GRID
        anom = float(np.clip(self._anomaly[ni], 0.0, 1.0))

        # sensor simulation — noisy but correlated with true threat
        near_poacher = any(
            self._poacher_on[i] and
            np.linalg.norm(self.POACHER_SPOTS[i] - self._ranger) < DETECT_R * 2
            for i in range(N_POACHERS)
        )
        acoustic  = float(near_poacher or (self._rng.random() < 0.02))
        vibration = float(np.clip(
            0.7 if near_poacher else self._rng.normal(0.1, 0.05), 0.0, 1.0
        ))
        pressure  = float(np.clip(
            0.75 if near_poacher else (self._rng.normal(0, 0.1) + 1) / 2, 0.0, 1.0
        ))

        diag = GRID * np.sqrt(2)
        pd   = np.clip(
            np.linalg.norm(self.POACHER_SPOTS - self._ranger, axis=1) / diag,
            0.0, 1.0
        )

        battery     = float(np.clip(self._battery, 0.0, 1.0))
        active_th   = float(np.sum(self._poacher_on)) / N_POACHERS
        coverage    = float(len(self._visited)) / (GRID * GRID)
        time_rem    = 1.0 - self._step / MAX_STEPS

        active_idx  = np.where(self._poacher_on)[0]
        near_p      = (
            float(np.min(
                np.linalg.norm(self.POACHER_SPOTS[active_idx] - self._ranger, axis=1)
            ) / diag)
            if len(active_idx) else 1.0
        )

        false_rate    = self._n_false / max(self._n_alerts, 1)
        recharge_frac = self._recharge_count / MAX_RECHARGES
        caught_frac   = self._caught / max(self._total_spawned, 1)
        missed_frac   = self._missed / MAX_MISSES
        step_frac     = self._step / MAX_STEPS

        return np.array([
            rx, ry,                          # 0-1  ranger position
            nax, nay, anom,                  # 2-4  nearest animal
            acoustic, vibration, pressure,   # 5-7  sensors
            pd[0], pd[1], pd[2], pd[3],      # 8-11 poacher spot distances
            battery,                         # 12   battery
            active_th,                       # 13   active threat ratio
            coverage,                        # 14   coverage
            time_rem,                        # 15   time remaining
            np.clip(near_p,      0.0, 1.0),  # 16   nearest active poacher dist
            np.clip(false_rate,  0.0, 1.0),  # 17   false alarm rate
            np.clip(recharge_frac,0.0,1.0),  # 18   recharges used
            np.clip(caught_frac, 0.0, 1.0),  # 19   catch rate
            np.clip(missed_frac, 0.0, 1.0),  # 20   miss rate
            step_frac,                       # 21   episode progress
        ], dtype=np.float32)

    #  INVESTIGATE 

    def _investigate(self):
        reward = 0.0

        for i in range(N_POACHERS):
            if self._poacher_on[i]:
                dist = np.linalg.norm(self.POACHER_SPOTS[i] - self._ranger)
                if dist <= DETECT_R:
                    self._poacher_on[i] = False
                    self._caught       += 1
                    # proximity bonus — closer catch = better
                    proximity_bonus = (DETECT_R - dist) / DETECT_R * 15.0
                    reward += R_POACHER + R_ZONE_CLEAR + proximity_bonus

        for i in range(N_ANIMALS):
            dist = np.linalg.norm(self._animals[i] - self._ranger)
            if dist <= DETECT_R and self._anomaly[i] > 0.7:
                reward         += R_WELFARE
                self._anomaly[i] = max(self._anomaly[i] - 0.4, 0.1)

        return float(min(reward, MAX_INVESTIGATE_REWARD))

    #  DISPATCH ALERT

    def _dispatch(self):
        self._n_alerts += 1

        # check if there's a genuine nearby threat
        for i in range(N_POACHERS):
            if self._poacher_on[i]:
                dist = np.linalg.norm(self.POACHER_SPOTS[i] - self._ranger)
                if dist <= DETECT_R * 1.5:
                    return R_ALERT_OK

        # false alarm — penalty escalates with repeat offences
        self._n_false += 1
        escalation = R_FALSE_ESCALATE * max(0, self._n_false - 3)
        return R_FALSE_ALARM + escalation   # gets worse each time

    #  WORLD DYNAMICS 

    def _world_step(self):
        # animals drift randomly
        drift          = self._rng.normal(0, 0.15, (N_ANIMALS, 2))
        self._animals  = np.clip(self._animals + drift, 0.0, GRID)

        # anomaly scores evolve
        self._anomaly  = np.clip(
            self._anomaly + self._rng.normal(0, 0.02, N_ANIMALS),
            0.0, 1.0
        )

        # new poachers spawn randomly (~2% chance per step)
        if self._rng.random() < 0.02:
            off = np.where(~self._poacher_on)[0]
            if len(off):
                idx = self._rng.choice(off)
                self._poacher_on[idx] = True
                self._total_spawned  += 1

        # poachers escape (~1% chance per step per active poacher)
        for i in np.where(self._poacher_on)[0]:
            if self._rng.random() < 0.01:
                self._poacher_on[i] = False
                self._missed       += 1
                # also penalise the step directly
                # (reward already has R_STEP; missed counter drives termination)

    #  ZONE ID

    def _zone(self, pos):
        x = int(np.clip(pos[0], 0, GRID - 1))
        y = int(np.clip(pos[1], 0, GRID - 1))
        return y * GRID + x

    #  RENDER

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        """Pygame window render."""
        from environment.rendering import Renderer
        if self._renderer is None:
            self._renderer = Renderer(self)
        self._renderer.update_coverage(self._visited)
        self._renderer.render()

    def _render_rgb_array(self):
        """
        Returns an (H, W, 3) uint8 numpy array.
        Uses matplotlib Agg — no window opened.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self._fig is None:
            self._fig, self._ax = plt.subplots(
                figsize=(6, 6), facecolor="#0d1f0d"
            )
            self._fig.subplots_adjust(
                left=0.02, right=0.98, top=0.94, bottom=0.04
            )

        ax = self._ax
        ax.clear()
        ax.set_facecolor("#0d1f0d")
        ax.set_xlim(-0.5, GRID - 0.5)
        ax.set_ylim(-0.5, GRID - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # coverage shading
        for zid in self._visited:
            gx = zid % GRID
            gy = zid // GRID
            ax.add_patch(mpatches.Rectangle(
                (gx - 0.5, gy - 0.5), 1, 1,
                linewidth=0, facecolor="#1a3d1a", alpha=0.6, zorder=1
            ))

        # grid lines
        for i in range(GRID + 1):
            ax.axhline(i - 0.5, color="#1f3d1f", linewidth=0.4, zorder=2)
            ax.axvline(i - 0.5, color="#1f3d1f", linewidth=0.4, zorder=2)

        # poacher hot-spots
        for i, (px, py) in enumerate(POACHER_SPOTS):
            color = "#ff2222" if self._poacher_on[i] else "#555555"
            ax.add_patch(mpatches.Circle(
                (px, py), 0.35, color=color, alpha=0.6, zorder=3
            ))
            ax.text(px, py, "P", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=4)

        # animals (colour = anomaly score)
        for i, (ax_, ay) in enumerate(self._animals):
            c = plt.cm.RdYlGn(1.0 - self._anomaly[i])
            self._ax.scatter(ax_, ay, s=55, color=c, zorder=5,
                             edgecolors="white", linewidths=0.4)

        # ranger
        rx, ry = self._ranger
        ax.scatter(rx, ry, s=180, color="#00cfff", zorder=7,
                   marker="^", edgecolors="white", linewidths=0.8)

        # detection radius
        ax.add_patch(mpatches.Circle(
            (rx, ry), DETECT_R, fill=False,
            edgecolor="#00cfff", linewidth=0.8,
            linestyle="--", alpha=0.5, zorder=6
        ))

        # battery bar
        batt     = float(np.clip(self._battery, 0.0, 1.0))
        batt_col = (
            "#44ff44" if batt > 0.5 else
            "#ffaa00" if batt > 0.25 else
            "#ff2222"
        )
        ax.add_patch(mpatches.Rectangle(
            (-0.4, -0.45), batt * (GRID - 0.2), 0.18,
            color=batt_col, alpha=0.85, zorder=8
        ))

        coverage = len(self._visited) / (GRID * GRID) * 100
        title = (
            f"Step {self._step}/{MAX_STEPS}  |  "
            f"Battery {batt*100:.0f}%  |  "
            f"Caught {self._caught}  Missed {self._missed}  |  "
            f"Coverage {coverage:.0f}%  |  "
            f"Reward {self._cum_reward:+.0f}"
        )
        ax.set_title(title, fontsize=7, color="#c9d1d9",
                     pad=3, fontfamily="monospace")

        self._fig.canvas.draw()
        w, h = self._fig.canvas.get_width_height()
        frame = np.frombuffer(
            self._fig.canvas.buffer_rgba(), dtype=np.uint8
        ).reshape(h, w, 4)
        return frame[:, :, :3].copy()

   

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None