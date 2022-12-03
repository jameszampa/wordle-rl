import os
import cv2
import time
from WordleEnv import WordleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


model_path = f"models/1670083179/12000000.zip"
logdir = f"logs/temp/"

env = WordleEnv(logdir, harsh=False)
obs = env.reset()

model = PPO.load(model_path, env)

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()
    key = cv2.waitKey(1)

    while key != 13:
        key = cv2.waitKey(1)
    

    