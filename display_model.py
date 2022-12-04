import os
import cv2
import time
from WordleEnv import WordleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


model_path = f"models/1670110794/6000000.zip"
logdir = f"logs/temp/"

env = WordleEnv(logdir, harsh=True)
obs = env.reset()

model = PPO.load(model_path, env, device='cuda')

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render()
    key = cv2.waitKey(1)
    if dones:
        while key != 13:
            key = cv2.waitKey(1)
    if key == 13:
        env.reset()

    

    