import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.lifeloop_env import WildlifeLoopEnv

# Create environment
env = WildlifeLoopEnv()
env = Monitor(env)

# Hyperparameters
dqn_params = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01,
    "verbose": 1,
    "policy_kwargs": dict(net_arch=[256, 256])
}

# Model
model = DQN("MlpPolicy", env, **dqn_params, device="cuda" if torch.cuda.is_available() else "cpu")

# Checkpointing
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./models/dqn", name_prefix="wildlife_dqn")

# Train
model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

# Save final model
model.save("./models/dqn/wildlife_dqn_final")

env.close()