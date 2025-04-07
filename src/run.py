
from sys import argv
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG, SAC

model_name = argv[1]

def run():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    match model_name:
        case "PPO":
            model = PPO.load("models/PPO", env=env)
        case "A2C":
            model = A2C.load("models/A2C", env=env)
        case "DDPG":
            model = DDPG.load("models/DDPG", env=env)
        case "SAC":
            model = SAC.load("models/SAC", env=env)
        case _:
            print("Usage: run.py [PPO | A2C | DDPG | SAC]")
            return
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        
if __name__ == '__main__':
    run()