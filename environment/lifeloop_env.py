import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

GRID = 10
MAX_STEPS = 500
N_ANIMALS = 8
N_POACHERS = 4
DETECT_R = 2.0
PATROL_CAP = 25

POACHER_SPOTS = np.array(
    [[1., 1.], [8., 2.], [2., 8.], [8., 8.]], dtype=np.float32
)

# Rewards
R_POACHER = 30.0
R_ZONE_CLEAR = 3.0
R_WELFARE = 5.0
R_PATROL = 1.0
R_ALERT_OK = 15.0
R_FALSE_ALARM = -6.0
R_MISSED = -10.0
R_BATTERY = -20.0
R_STEP = -0.1
MAX_INVESTIGATE_REWARD = 40.0

ACTION_NAMES = [
    "Move North", "Move South", "Move East", "Move West",
    "Investigate", "Dispatch Alert", "Recharge",
]

class WildlifeLoopEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Attach constants so renderer can access
        self.POACHER_SPOTS = POACHER_SPOTS
        self.MAX_STEPS = MAX_STEPS
        self.GRID = GRID

        # ✅ REQUIRED SPACES
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        self._rng = np.random.default_rng()
        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed) if seed else np.random.default_rng()

        self._ranger = self._rng.uniform(1, self.GRID - 1, 2).astype(np.float32)
        self._animals = self._rng.uniform(0, self.GRID, (N_ANIMALS, 2)).astype(np.float32)
        self._anomaly = self._rng.uniform(0, 0.3, N_ANIMALS).astype(np.float32)
        self._poacher_on = (self._rng.random(N_POACHERS) < 0.5)

        self._step = 0
        self._battery = 1.0
        self._visited = {self._zone(self._ranger)}
        self._patrol_cnt = 0
        self._n_alerts = 0
        self._n_false = 0
        self._caught = 0
        self._missed = 0
        self._cum_reward = 0.0

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        assert 0 <= action <= 6

        self._step += 1
        reward = R_STEP

        # Movement
        if action == 0:
            self._ranger[1] = min(self._ranger[1] + 1.0, self.GRID - 1)
            self._battery -= 0.003
        elif action == 1:
            self._ranger[1] = max(self._ranger[1] - 1.0, 0.0)
            self._battery -= 0.003
        elif action == 2:
            self._ranger[0] = min(self._ranger[0] + 1.0, self.GRID - 1)
            self._battery -= 0.003
        elif action == 3:
            self._ranger[0] = max(self._ranger[0] - 1.0, 0.0)
            self._battery -= 0.003
        elif action == 4:
            reward += self._investigate()
            self._battery -= 0.005
        elif action == 5:
            reward += self._dispatch()
            self._battery -= 0.004
        elif action == 6:
            self._battery = min(self._battery + 0.02, 1.0)

        # Patrol reward
        zid = self._zone(self._ranger)
        if zid not in self._visited and self._patrol_cnt < PATROL_CAP:
            self._visited.add(zid)
            self._patrol_cnt += 1
            reward += R_PATROL
        else:
            self._visited.add(zid)

        self._world_step()
        reward -= (1.0 - self._battery) * 0.2

        terminated = bool(self._battery <= 0.0 or self._step >= self.MAX_STEPS)
        truncated = False

        self._cum_reward += reward

        info = {
            "step": int(self._step),
            "battery": float(self._battery),
            "coverage_pct": float(len(self._visited) / (self.GRID * self.GRID)),
            "threats_caught": int(self._caught),
            "threats_missed": int(self._missed),
            "false_alerts": int(self._n_false),
            "cumulative_reward": float(self._cum_reward),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        rx, ry = self._ranger / self.GRID

        dists = np.linalg.norm(self._animals - self._ranger, axis=1)
        ni = int(np.argmin(dists))
        nax, nay = self._animals[ni] / self.GRID
        anom = float(np.clip(self._anomaly[ni], 0.0, 1.0))

        near_poacher = any(
            self._poacher_on[i] and np.linalg.norm(self.POACHER_SPOTS[i] - self._ranger) < DETECT_R * 2
            for i in range(N_POACHERS)
        )

        acoustic = float(near_poacher or (self._rng.random() < 0.02))
        vibration = float(np.clip(0.6 if near_poacher else self._rng.normal(0.1, 0.05), 0.0, 1.0))
        pressure = float(np.clip(0.7 if near_poacher else (self._rng.normal(0, 0.1) + 1)/2, 0.0, 1.0))

        diag = self.GRID * np.sqrt(2)
        pd = np.clip(np.linalg.norm(self.POACHER_SPOTS - self._ranger, axis=1)/diag, 0.0, 1.0)

        battery = np.clip(self._battery, 0.0, 1.0)
        active_th = np.sum(self._poacher_on) / N_POACHERS
        coverage = len(self._visited) / (self.GRID * self.GRID)
        time_rem = 1.0 - self._step / self.MAX_STEPS

        active_idx = np.where(self._poacher_on)[0]
        near_p = (np.min(np.linalg.norm(self.POACHER_SPOTS[active_idx]-self._ranger, axis=1))/diag
                  if len(active_idx) else 1.0)

        far = self._n_false / max(self._n_alerts, 1)

        return np.array([
            rx, ry, nax, nay, anom,
            acoustic, vibration, pressure,
            pd[0], pd[1], pd[2], pd[3],
            battery, active_th, coverage, time_rem,
            np.clip(near_p, 0.0, 1.0),
            np.clip(far, 0.0, 1.0)
        ], dtype=np.float32)

    def _investigate(self):
        reward = 0.0

        for i in range(N_POACHERS):
            if self._poacher_on[i]:
                dist = np.linalg.norm(self.POACHER_SPOTS[i] - self._ranger)
                if dist <= DETECT_R:
                    self._poacher_on[i] = False
                    self._caught += 1
                    reward += (R_POACHER + R_ZONE_CLEAR)

        for i in range(N_ANIMALS):
            dist = np.linalg.norm(self._animals[i] - self._ranger)
            if dist <= DETECT_R and self._anomaly[i] > 0.7:
                reward += R_WELFARE
                self._anomaly[i] = 0.1

        return float(min(reward, MAX_INVESTIGATE_REWARD))

    def _dispatch(self):
        self._n_alerts += 1

        signal_strength = sum(self._poacher_on)

        for i in range(N_POACHERS):
            if self._poacher_on[i]:
                dist = np.linalg.norm(self.POACHER_SPOTS[i] - self._ranger)
                if dist <= DETECT_R * 1.5 and signal_strength > 0:
                    return R_ALERT_OK

        self._n_false += 1
        return R_FALSE_ALARM

    def _world_step(self):
        drift = self._rng.normal(0, 0.15, (N_ANIMALS, 2))
        self._animals = np.clip(self._animals + drift, 0.0, self.GRID)

        self._anomaly = np.clip(
            self._anomaly + self._rng.normal(0, 0.02, N_ANIMALS),
            0.0, 1.0
        )

        if self._rng.random() < 0.02:
            off = np.where(~self._poacher_on)[0]
            if len(off):
                self._poacher_on[self._rng.choice(off)] = True

        for i in np.where(self._poacher_on)[0]:
            if self._rng.random() < 0.01:
                self._poacher_on[i] = False
                self._missed += 1

    def _zone(self, pos):
        x = int(np.clip(pos[0], 0, self.GRID - 1))
        y = int(np.clip(pos[1], 0, self.GRID - 1))
        return y * self.GRID + x

    def render(self):
        from .rendering import Renderer

        if self._renderer is None:
            self._renderer = Renderer(self)

        self._renderer.update_coverage(self._visited)
        self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None