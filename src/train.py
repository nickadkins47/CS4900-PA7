from sys import argv
import gymnasium as gym
from stable_baselines3 import PPO, DQN, DDPG, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "env_name": "LunarLander-v3",
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
}

run = wandb.init(
    project="CS4900-PA7",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)

def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"models/{run.id}/videos",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)

model_name = argv[1]
timesteps = config["total_timesteps"]
policy = config["policy_type"]
log_dir = f"models/{run.id}/log"
callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

match model_name:
    case "PPO":
        model =  PPO(policy, env, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    case "DQN":
        model =  DQN(policy, env, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    case "DDPG":
        model = DDPG(policy, env, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    case "SAC":
        model =  SAC(policy, env, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    case _:
        print("Usage: train.py [PPO | DQN | DDPG | SAC]")

run.finish()