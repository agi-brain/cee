import os
import time

all_envs = ['ALE/BeamRider-v5']
index= 0

envs = [all_envs[index]]
config_name = 'atari'
methods = ['Nill']
alg = 'PPO_Atari'
n_repeats = 1
total_timesteps = 1000000
log_interval = 1

for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            cmd_line = f"python -m min_red.train " \
                       f" --f min_red/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --env_id {env}" \
                       f" --method {method} " \
                       f" --algorithm.learn.log_interval {log_interval} " \
                       f" --total_timesteps {total_timesteps}" \
                       f" --wandb False & "
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)

