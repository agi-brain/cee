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
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from torch.nn.modules.activation import Tanh, ReLU
# from stable_baselines3.common.evaluation import evaluate_policy
from common.evaluation import evaluate_policy_and_save
from pureppo.ppo_savemodel import SavePPO
from pureppo.rnd import RNDCustomCallback, initialize_rnd
import wandb

torch.set_num_threads(8)


# def eval_policy(env, model):
#     obs = env.reset()
#     traj_rewards = [0]
#     while True:
#         action, _state = model.predict(obs, deterministic=False)
#         next_obs, reward, done, info = env.step(action)
#         obs = next_obs
#         env.render()
#         time.sleep(0.03)
#         traj_rewards[-1] += reward
#         if done:
#             obs = env.reset()
#             m = input('Enter member idx: ')
#             env.member = int(m)
#             print(f"env member: {env.member}, R: {np.mean(traj_rewards)}")
#             traj_rewards.append(0)


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


def train(config, log_path):
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

    # initialize the RND settings
    use_rnd_curiosity = config.algorithm.rnd_curiosity if hasattr(config.algorithm, "rnd_curiosity") else False
    if use_rnd_curiosity:
        input_channels = 4  # Grayscale image, should be the same as n_stack.
        output_dim = 512  # Example output dimension
        target_network, predictor_network, optimizer = initialize_rnd(input_channels, output_dim, config.device)
        rnd_callback = RNDCustomCallback(target_network, predictor_network, optimizer, device=config.device)
    else:
        rnd_callback = None

    if len(env.observation_space.shape) >= 3:
        policy = 'CnnPolicy'
    else:
        policy = 'MlpPolicy'
    # policy = 'MlpPolicy'
    model = SavePPO(policy, env, tensorboard_log=log_path, **config.algorithm.policy)

    model.learn(**config.algorithm.learn, callback=rnd_callback)

    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model_path = os.path.join(log_path, "model")
        model.save(model_path)
        # test_mf_model = ActionModel.load(mf_model_path)
    if config.play_model:
        save_path = os.path.join(log_path, "eval.npy")
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
    # config.algorithm.policy.method = config.method
    # config.algorithm.policy.wandb = config.wandb
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="pureppo/config/atari")
    args, extra_args = parser.parse_known_args()
    # get default parameters in this environment and override with extra_args
    config = get_config(args.f)
    # load default parameters in config_path/algorithm_type and override
    config = bcast_config_vals(config)
    pretty(config)

    config.play_model = False

    if 'n_actions' in config.env_kwargs.keys():
        n = config.env_kwargs.n_actions
    elif 'n_redundancies' in config.env_kwargs.keys():
        n = config.env_kwargs.n_redundancies
    else:
        n = -1

    experiment_name = "PurePPO_" + config.env_id + '_' + config.algorithm_type + '_' + "n" + str(
        n) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = os.path.join("log_formal", experiment_name)
    if "wandb" in config:
        if config["wandb"]:
            # wandb.init(project="muzero-pytorch", entity='jyh', sync_tensorboard=True, name=args.exp_name,
            #            config=muzero_config.get_hparams(), dir=str(exp_path))
            wandb.init(
                # set the wandb project where this run will be logged
                project=config.env_id + '_' + config.algorithm_type,

                # track hyperparameters and run metadata
                config=config.algorithm.policy.policy_kwargs,
                sync_tensorboard=True,
                name=experiment_name
                # dir=str(log_path)
            )
    train(config, log_path)
    wandb.finish()
