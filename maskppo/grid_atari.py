import os
import time

all_envs = ['ALE/MsPacman-v5','ALE/Qbert-v5','ALE/DemonAttack-v5','ALE/BeamRider-v5','ALE/KungFuMaster-v5','ALE/TimePilot-v5','ALE/Kangaroo-v5','ALE/WizardOfWor-v5','ALE/Phoenix-v5','ALE/Carnival-v5','ALE/AirRaid-v5']
index= 8

#envs = [all_envs[index]]

envs = ['ALE/NameThisGame-v5']


config_name = 'atari'

alg = 'PPO_Atari'
n_repeats = 1
total_timesteps = 10000000
log_interval = 1
mask = ["True"]
mask_threshold = 0.5
# n_steps = 512

for trials in range(n_repeats):
    for mask_flag in mask:
        for env in envs:
            cmd_line = f"python -m maskppo.train_atari" \
                       f" --f maskppo/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --env_id {env}" \
                       f" --mask {mask_flag} " \
                       f" --mask_threshold {mask_threshold} " \
                       f" --log_interval {log_interval} " \
                       f" --total_timesteps {total_timesteps} &"
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
