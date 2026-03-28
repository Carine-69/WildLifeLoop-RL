import gym
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.lifeloop_env import WildlifeLoopEnv
from stable_baselines3 import DQN, PPO, A2C

models = {
    "dqn": "./models/dqn/wildlife_dqn_final.zip",
    "ppo": "./models/ppo/wildlife_ppo_final.zip",
    "a2c": "./models/a2c/wildlife_a2c_final.zip"
}

for name, path in models.items():
    print(f"Evaluating {name.upper()}...")
    env = WildlifeLoopEnv()
    model = None
    if name == "dqn":
        model = DQN.load(path, env=env)
    elif name == "ppo":
        model = PPO.load(path, env=env)
    elif name == "a2c":
        model = A2C.load(path, env=env)

    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            obs = env.reset()
    print(f"{name.upper()} total reward: {total_reward}")
    env.close()