import gym
import envs
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from min_red.action_model_trainer import ActionModelTrainer
from min_red.mf_model_trainer import MfModelTrainer
from min_red.config.parser_args import get_config
from min_red.config.config import Config
import argparse
from stable_baselines3.common.env_util import make_vec_env,make_atari_env

from stable_baselines3 import PPO
# from mixture.make_atari_stack_env import make_atari_stack_env
from common.format_string import pretty
import wandb
import os
from stable_baselines3.common.logger import configure
# tmp_path = "tmp/test_log/"
# new_logger = configure(tmp_path,["stdout","csv","tensorboard"])
# import min_red.dqn.policies
import torch

from pureppo.rnd import RNDCustomCallback, initialize_rnd

torch.set_num_threads(8)

curiosity_policy_path_dict = {"unlockpickupar-v0": "log/unlockpickupactionbonus-v0_PPO_n-1_2023-04-13-10-14-21/model614400",
                         "unlockpickupactionbonus-v0": "log/unlockpickupactionbonus-v0_PPO_n-1_2023-04-13-10-14-21/model614400",
                         "unlockpickupuncertaingoals-v0":"log/unlockpickupuncertaingoalsactionbonus-v0_PPO_n-1_2023-03-19-19-49-37/model409600"}
curiosity_policy_path_dict = {}
def eval_policy(env, model):
    obs = env.reset()
    traj_rewards = [0]
    while True:
        action, _state = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        if done:
            obs = env.reset()
            m = input('Enter member idx: ')
            env.member = int(m)
            print(f"env member: {env.member}, R: {np.mean(traj_rewards)}")
            traj_rewards.append(0)


def train(config, log_path,logger):
    if config.is_atari:
        make_env = make_atari_env  # make_atari_stack_env, # tecaher make_vec_env
        env = make_env(config.env_id, n_envs=8, vec_env_cls=DummyVecEnv,
                       vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)
        env = VecFrameStack(env, n_stack=4)

        # origin n_envs=1 jin change 8

    else:
        make_env = make_vec_env
        env = make_env(config.env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                       vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)
    # env = gym.make(config.env_id)
    obs_shape = list(env.observation_space.shape)

    # initialize the RND settings
    use_rnd_curiosity = config.algorithm.rnd_curiosity if hasattr(config.algorithm, "rnd_curiosity") else False
    if use_rnd_curiosity:
        input_channels = 4  # Grayscale image, should be the same as n_stack.
        output_dim = 512  # Example output dimension
        target_network, predictor_network, optimizer = initialize_rnd(input_channels, output_dim)
        rnd_callback = RNDCustomCallback(target_network, predictor_network, optimizer)
    else:
        rnd_callback = None

    if config.algorithm.discrete:
        from min_red.min_red_ppo import MinRedPPO as Algorithm
        from stable_baselines3.dqn.policies import MultiInputPolicy as ActionModel
        # from min_red.dqn.policies import CnnPolicy as MfModel
        if len(obs_shape) > 2:
            s_space = (obs_shape[2], *obs_shape[:2])
            ssprime_shape = (2 * obs_shape[2], *obs_shape[:2])
            policy = 'CnnPolicy'
        else:
            s_space = (obs_shape[0], )
            ssprime_shape = (2 * obs_shape[0],)
            policy = 'MlpPolicy'
    else:
        pass
        # from min_red.min_red_sac import MinRedSAC as Algorithm
        # # from stable_baselines3.sac import MlpPolicy as Model
        # from continuous_action_model import DiagGaussianPolicy as ActionModel
        # ssprime_shape = (2*obs_shape[0],)
        # policy = 'MlpPolicy'

    # create action model obs space by extending env's obs space
    ssprime_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                       high=env.observation_space.high.max(),
                                       shape=ssprime_shape,
                                       dtype=env.observation_space.dtype)

    pi_obs_spce = gym.spaces.Box(low=0.,
                                 high=1.,
                                 shape=(env.action_space.n,),
                                 dtype=env.observation_space.dtype)

    new_ssprime_obs_space = gym.spaces.Dict({
        "image": ssprime_obs_space,
        "vector": pi_obs_spce,
    })

    action_model = ActionModel(observation_space=new_ssprime_obs_space,
                               action_space=env.action_space,
                               lr_schedule= lambda x: 3e-4).to(config.device)

    action_trainer = ActionModelTrainer(action_model=action_model,
                                        discrete=config.algorithm.discrete,
                                        new_logger=logger)

    #mask_factor_model: input: s + s' + action[one-hot vector]  output: Mask(s,a,b) = KL(P_a|P_b)

    s_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                       high=env.observation_space.high.max(),
                                       shape=s_space,
                                       dtype=env.observation_space.dtype)

    sa_obs_space = gym.spaces.Dict({
        "image": s_obs_space,
        "vector": pi_obs_spce,
    })
    mf_model = ActionModel(observation_space=sa_obs_space,
                               action_space=env.action_space,
                               lr_schedule=lambda x: 3e-4).to(config.device)
    # mf_model = MfModel(observation_space=s_obs_space,
    #                        action_space=env.action_space,
    #                        net_arch=[128, dict(vf=[256], pi=[16])],
    #                        lr_schedule=lambda x: config.algorithm.policy.learning_rate).to(config.device)

    mf_trainer = MfModelTrainer(mf_model=mf_model,
                                discrete=config.algorithm.discrete,
                                new_logger=logger)
    # if config.env_id == "unlockpickupar-v0" or config.env_id == "unlockpickupactionbonus-v0":
    #     # curiosity_ppo_path = "log/unlockpickupactionbonus-v0_PPO_n-1_2023-03-15-15-55-10/model2000.zip"
    #     curiosity_ppo_path = "log/unlockpickupactionbonus-v0_PPO_n-1_2023-04-13-10-14-21/model614400"
    # elif config.env_id == "unlockpickupuncertaingoals-v0":
    #     curiosity_ppo_path = "log/unlockpickupuncertaingoalsactionbonus-v0_PPO_n-1_2023-03-19-19-49-37/model409600"
    # else:
    #     curiosity_ppo_path = None
    if config.env_id in curiosity_policy_path_dict:
        curiosity_ppo_path = curiosity_policy_path_dict[config.env_id]
    else:
        curiosity_ppo_path = None

    model = Algorithm(policy, env, action_trainer=action_trainer, mf_trainer=mf_trainer, log_path=log_path, **config.algorithm.policy)
    if curiosity_ppo_path:
        curiosity_ppo = PPO.load(curiosity_ppo_path,device=config.device)
        model.policy.load_state_dict(curiosity_ppo.policy.state_dict())

    model.set_logger(logger)
    # model.set_logger(new_logger)

    model.learn(**config.algorithm.learn,callback=rnd_callback)
    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model_path = os.path.join(log_path,"model")
        model.save(model_path,exclude=["action_trainer","mf_trainer"])

        mf_model_path = os.path.join(log_path, "mfmodel")
        mf_model.save(mf_model_path)
        # test_mf_model = ActionModel.load(mf_model_path)

        action_model_path = os.path.join(log_path, "actionmodel")
        action_model.save(action_model_path)
        # test_action_model = ActionModel.load(action_model_path)
    if config.play_model:
        eval_policy(env, model)


def bcast_config_vals(config):
    algorithm_config = Config(os.path.join(config.config_path, config.algorithm_type))
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.learn.total_timesteps = config.total_timesteps
    config.algorithm.policy["device"] = config.device
    config.algorithm.policy.method = config.method
    config.algorithm.policy.wandb = config.wandb
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)
    config = bcast_config_vals(config)

    if config.wandb:
        run = wandb.init(config=config)
    else:
        pretty(config)

    if 'n_actions' in config.env_kwargs.keys():
        n = config.env_kwargs.n_actions
    elif 'n_redundancies' in config.env_kwargs.keys():
        n = config.env_kwargs.n_redundancies
    else:
        n = -1
    experiment_name = "SF_Mask_"+ str(config.env_id) + '_' + str(config.method) + '_' + str(config.algorithm_type) + '_' + "n" + str(n) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    log_path = os.path.join("log", experiment_name)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    train(config,log_path,logger)

