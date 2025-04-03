
from sys import argv
import gymnasium as gym
from stable_baselines3 import PPO, DQN, DDPG, SAC

model_name = argv[1]
timesteps = 20_000

def train():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    match model_name:
        case "PPO":
            model = PPO("MlpPolicy", env, verbose=1)
        case "DQN":
            model = DQN("MlpPolicy", env, verbose=1)
        case "DDPG":
            model = DDPG("MlpPolicy", env, verbose=1)
        case "SAC":
            model = SAC("MlpPolicy", env, verbose=1)
        case _:
            print("Usage: train.py [PPO | DQN | DDPG | SAC]")
            return
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(f"models/{model_name}")

if __name__ == '__main__':
    train()