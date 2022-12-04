import os
import time
from WordleEnv import WordleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.total_games = None
        self.total_games_won = None

    def _on_rollout_end(self):
        rollout_games = 0
        rollout_games_won = 0
        for env in self.training_env.envs:
            rollout_games += env.games
            rollout_games_won += env.games_won
        if not self.total_games is None:
            prev_total = self.total_games
            prev_total_won = self.total_games_won
            self.total_games = rollout_games
            self.total_games_won = rollout_games_won
            rollout_games -= prev_total
            rollout_games_won -= prev_total_won
        else:
            self.total_games = rollout_games
            self.total_games_won = rollout_games_won
        if rollout_games > 0:
            self.logger.record("games/win_rate", rollout_games_won / rollout_games)
        else:
            self.logger.record("games/win_rate", 0)
        self.logger.record("games/played", rollout_games)
        self.logger.record("games/won", rollout_games_won)

    def _on_step(self):
        return True


ts = time.time()
models_dir = f"models/{int(ts)}/"
logdir = f"logs/{int(ts)}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env(WordleEnv, n_envs=64, env_kwargs={ 'logdir' : logdir, 'harsh' : True })
#env = WordleEnv(logdir, harsh=True)
obs = env.reset()

model_path = f"models/1670110794/6000000.zip"
model = PPO.load(model_path, env, verbose=1, tensorboard_log=logdir, n_steps=1024)

TIMESTEPS = 1000000
iters = 6000000 / TIMESTEPS
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", log_interval=1, callback=TensorboardCallback())
    model.save(f"{models_dir}/{TIMESTEPS*iters}")