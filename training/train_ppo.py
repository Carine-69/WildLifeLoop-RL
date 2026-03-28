import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.lifeloop_env import WildlifeLoopEnv

# Environment
env = WildlifeLoopEnv()
env = Monitor(env)

ppo_params = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "verbose": 1,
    "policy_kwargs": dict(net_arch=[256, 256])
}

model = PPO("MlpPolicy", env, **ppo_params, device="cuda" if torch.cuda.is_available() else "cpu")

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/ppo", name_prefix="wildlife_ppo")

model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

model.save("./models/ppo/wildlife_ppo_final")

env.close()