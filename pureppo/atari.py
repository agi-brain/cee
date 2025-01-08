import os
import time

# all_envs = ['ALE/MsPacman-v5','ALE/Qbert-v5','DemonAttack-v5','BeamRider-v5']
# index= 0

# envs = [all_envs[index]]
# envs = ['ALE/MsPacman-v5','ALE/Qbert-v5','DemonAttack-v5','BeamRider-v5']
envs = ['ALE/MsPacman-v5']
config_name = 'atari'

alg = 'PPO_Atari'
n_repeats = 1
total_timesteps = 10000000
log_interval = 1
# n_steps = 512

for trials in range(n_repeats):
        for env in envs:
            cmd_line = f"python -m pureppo.train" \
                       f" --f pureppo/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --env_id {env}" \
                       f" --log_interval {log_interval} " \
                       f" --total_timesteps {total_timesteps} &"
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
