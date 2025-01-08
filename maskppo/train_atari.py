import gym
import envs
import numpy as np
import time
import argparse
from common.format_string import pretty
from common.parser_args import get_config
from common.config import Config
import os
import torch
from stable_baselines3.common.env_util import make_vec_env,make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy,CnnPolicy
from torch.nn.modules.activation import Tanh,ReLU
from stable_baselines3.common.evaluation import evaluate_policy
from common.evaluation import evaluate_policy_and_save

# from .maskppo import MaskPPO #origin run in terminal

# from .maskppo import MaskPPO # terminal
from maskppo import MaskPPO  # run in pycharm

from pureppo.rnd import RNDCustomCallback, initialize_rnd

import wandb
torch.set_num_threads(8)

import GPUtil

# add the Pre-training mask to here
Env_mask_dict = {"GoToPositionBonus-v0":"Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "unlockpickupar-v0":"SF_Mask_unlockpickupar-v0_Nill_PPO_n-1_2024-05-11-11-26-20",
        "girdCL-v0": "Mask_girdCL-v0_random_PPO_n6_2024-04-15-10-44-04",
        "GoToR3BlueKeyAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToR3GreenBoxAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToR3BlueKeyAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToR3GreyKeyAddPositionBonus-v0":"Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToDoorOpenR2AddPositionBonus-v0": "SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2GreyKeyAR-v0": "SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2GreenBoxAR-v0": "SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2RedBallAR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2BlueBallAR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2GreenBallAR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "ALE/BeamRider-v5":"BeamRider-v5_Nill_PPO_n-1_2024-11-08-10-43-55",
        "ALE/DemonAttack-v5":"DemonAttack-v5_Nill_PPO_n-1_2025-01-07-02-34-14"

}


def train(config, log_path, mask_path, mask_flag, mask_threshold,seed):
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

    use_rnd_curiosity = config.algorithm.rnd_curiosity if hasattr(config.algorithm, "rnd_curiosity") else False
    if use_rnd_curiosity:
        input_channels = 4  # Grayscale image, should be the same as n_stack.
        output_dim = 512  # Example output dimension
        target_network, predictor_network, optimizer = initialize_rnd(input_channels, output_dim,config.device)
        rnd_callback = RNDCustomCallback(target_network, predictor_network, optimizer,device=config.device)
    else:
        rnd_callback = None

    if len(env.observation_space.shape) >=3:
        policy = 'CnnPolicy'
    else:
        policy = 'MlpPolicy'

    mf_model_path = os.path.join(mask_path, "mfmodel")
    from stable_baselines3.dqn.policies import MultiInputPolicy as ActionModel
    mf_model = ActionModel.load(mf_model_path,device=config.device)

    model = MaskPPO(policy, env, tensorboard_log=log_path, mf_model=mf_model, mask_flag=mask_flag,
            mask_threshold=mask_threshold,**config.algorithm.policy, seed=seed)

    model.learn(**config.algorithm.learn, callback=rnd_callback)
    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model_path = os.path.join(log_path,"model")
        model.save(model_path)
        # test_mf_model = ActionModel.load(mf_model_path)
    if config.play_model:
        save_path = os.path.join(log_path,"eval.npy")
        mean, std = evaluate_policy_and_save(model, env, save_path=save_path, deterministic=False)
        print("mean:" + str(mean) + " std:" + str(std))


def bcast_config_vals(config):
    algorithm_config = Config(os.path.join(config.config_path, config.algorithm_type))
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.learn.total_timesteps = config.total_timesteps
    config.algorithm.policy["device"] = config.device
    if "activation_fn" in config.algorithm.policy.policy_kwargs:
        activation_fn = config.algorithm.policy.policy_kwargs["activation_fn"]
        if activation_fn == "ReLU":
            config.algorithm.policy.policy_kwargs["activation_fn"] = ReLU
        elif activation_fn == "Tanh":
            config.algorithm.policy.policy_kwargs["activation_fn"] = Tanh
        else:
            raise NotImplementedError
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    parser.add_argument('--mask', type=str, default="True")
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    args, extra_args = parser.parse_known_args()

    args.f = "./config/atari"

    # get default parameters in this environment and override with extra_args
    config = get_config(args.f)
    config["config_path"] = "./config"  # added by jin  run in pycharm

    config = bcast_config_vals(config)
    pretty(config)

    if args.mask == "True":
        mask_flag = True
    else:
        mask_flag = False
    mask_threshold = args.mask_threshold
    seed = args.seed

    goalstr = ''
    if 'goal' in config.env_kwargs.keys():
        goal = config.env_kwargs.goal
        goalstr = '_Goal'+str(goal)

    experiment_name = "CEE_Atari_seed_1" + str(config.env_id) + '_' +"mask"+ "seed_1"++str(config.algorithm_type) + goalstr + '_' \
            + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


    log_path = os.path.join("log_formal", experiment_name)

    mask_name = Env_mask_dict[config.env_id]

    mask_path = os.path.join("../log_formal",mask_name)
    # if "wandb" in config:
    #     if config["wandb"]:
    #         # wandb.init(project="muzero-pytorch", entity='jyh', sync_tensorboard=True, name=args.exp_name,
    #         #            config=muzero_config.get_hparams(), dir=str(exp_path))
    #         wandb.init(
    #             # set the wandb project where this run will be logged
    #             project=config.env_id + '_' + config.algorithm_type,
    #
    #             # track hyperparameters and run metadata
    #             config=config.algorithm.policy.policy_kwargs,
    #             sync_tensorboard=True,
    #             name=experiment_name
    #             # dir=str(log_path)
    #         )
    train(config, log_path, mask_path, mask_flag, mask_threshold=mask_threshold,seed=seed)
    # wandb.finish()

