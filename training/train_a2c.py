import gym
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.lifeloop_env import WildlifeLoopEnv

# Environment
env = WildlifeLoopEnv()
env = Monitor(env)

a2c_params = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "verbose": 1,
    "policy_kwargs": dict(net_arch=[256, 256])
}

model = A2C("MlpPolicy", env, **a2c_params, device="cuda" if torch.cuda.is_available() else "cpu")

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./models/a2c", name_prefix="wildlife_a2c")

model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

model.save("./models/a2c/wildlife_a2c_final")

env.close()