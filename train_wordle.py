import os
import time
from WordleEnv import WordleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


ts = time.time()
models_dir = f"models/{int(ts)}/"
logdir = f"logs/{int(ts)}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env(WordleEnv, n_envs=16, env_kwargs={ 'logdir' : logdir })
obs = env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, n_steps=6 * 1000)

TIMESTEPS = 1000000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", log_interval=1)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")