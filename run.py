import os
import sys
import time
import argparse
import pprint
from importlib import import_module

import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from mpi4py import MPI

import gym
from env_wrappers import DummyVecEnv, SubprocVecEnv, Monitor, L2M2019EnvBaseWrapper, RandomPoseInitEnv, \
                            ZeroOneActionsEnv, RewardAugEnv, PoolVTgtEnv, SkipEnv, Obs2VecEnv, NoopResetEnv
from logger import save_json

parser = argparse.ArgumentParser()
parser.add_argument('--play', action='store_true')

parser.add_argument('env', type=str, help='Environment id (e.g. HalfCheetah-v2 or L2M2019).')
parser.add_argument('alg', type=str, help='Algorithm name -- module where agent and learn function reside.')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--exp_name', default='exp', type=str)
# training params
parser.add_argument('--n_env', default=1, type=int, help='Number of environments in parallel.')
parser.add_argument('--n_total_steps', default=0, type=int, help='Number of training steps on single or vectorized environment.')
parser.add_argument('--max_episode_length', default=1000, type=int, help='Reset episode after reaching max length.')
# logging
parser.add_argument('--load_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--save_interval', default=1000, type=int)
parser.add_argument('--log_interval', default=1000, type=int)


# --------------------
# Config
# --------------------

def parse_unknown_args(args):
    # construct new parser for the string of unknown args and reparse; each arg is now a list of items
    p2 = argparse.ArgumentParser()
    for arg in args:
        if arg.startswith('--'): p2.add_argument(arg, type=eval, nargs='*')
    # if arg contains only a single value, replace the list with that value
    out = p2.parse_args(args)
    for k, v in out.__dict__.items():
        if len(v) == 1:
            out.__dict__[k] = v[0]
    return out

def get_alg_config(alg, env, extra_args=None):
    alg_args = getattr(import_module('algs.' + alg), 'defaults')(env)
    if extra_args is not None:
        alg_args.update({k: v for k, v in extra_args.items() if k in alg_args})
    return alg_args

def get_env_config(env, extra_args=None):
    env_args = None
    if env == 'L2M2019':
        env_args = {'model': '3D', 'visualize': False, 'integrator_accuracy': 1e-3, 'difficulty': 2, 'stepsize': 0.01}
    if extra_args is not None and env_args is not None:
        env_args.update({k: v for k, v in extra_args.items() if k in env_args})
    return env_args

def print_and_save_config(args, env_args, alg_args):
    print('Building environment and agent with the following config:')
    print(' Run config:\n' + pprint.pformat(args.__dict__))
    print(' Env config: ' + args.env + (env_args is not None)*('\n' + pprint.pformat(env_args)))
    print(' Alg config: ' + args.alg + '\n' + pprint.pformat(alg_args))
    save_json(args.__dict__, os.path.join(args.output_dir, 'config_run.json'))
    if env_args: save_json(env_args, os.path.join(args.output_dir, 'config_env.json'))
    save_json(alg_args, os.path.join(args.output_dir, 'config_alg.json'))


# --------------------
# Environment
# --------------------

def make_single_env(env_name, mpi_rank, subrank, seed, env_args, output_dir):
    # env_kwargs serve to initialize L2M2019Env
    #   L2M2019Env default args are: visualize=True, integrator_accuracy=5e-5, difficulty=2, seed=0, report=None
    #   additionally here: env_kwargs include `model` which can be '2D' or '3D'
    #   NOTE -- L2M2019Env uses seed in reseting the velocity target map in VTgtField.reset(seed) in v_tgt_field.py

    if env_name == 'L2M2019':
        env = L2M2019EnvBaseWrapper(**env_args)
        env = RandomPoseInitEnv(env)
#        env = NoopResetEnv(env)
        env = ZeroOneActionsEnv(env)
#        env = PoolVTgtEnv(env)  # NOTE -- needs to be after RewardAug if RewardAug uses the full vtgt field
        env = RewardAugEnv(env)
        env = SkipEnv(env)
        env = Obs2VecEnv(env)
    else:
        env = gym.envs.make(env_name)
        env.seed(seed + subrank if seed is not None else None)

    # apply wrappers
    env = Monitor(env, os.path.join(output_dir, str(mpi_rank) + '.' + str(subrank)))

    return env

def build_env(args, env_args):
    def make_env(subrank):
        return lambda: make_single_env(args.env, args.rank, subrank, args.seed + 10000*args.rank, env_args, args.output_dir)

    if args.n_env > 1:
        return SubprocVecEnv([make_env(i) for i in range(args.n_env)])
    else:
        return DummyVecEnv([make_env(i) for i in range(args.n_env)])


# --------------------
# Run train and play
# --------------------

def main(args, extra_args):
    # configure args for environment and algorithm; update defaults with extra_args
    env_args = get_env_config(args.env, extra_args.__dict__)
    alg_args = get_alg_config(args.alg, args.env, extra_args.__dict__)

    args.rank = MPI.COMM_WORLD.Get_rank()
    args.world_size = MPI.COMM_WORLD.Get_size()

    # setup logging
    if args.load_path:
        args.output_dir = os.path.dirname(args.load_path)
    if not args.output_dir:  # if not given use results/file_name/time_stamp
        logdir = args.exp_name + '_' + args.env + '_' + time.strftime("%Y-%m-%d_%H-%M")
        args.output_dir = os.path.join('results', logdir)
        if args.rank == 0: os.makedirs(args.output_dir)

    # print and save config
    if args.rank == 0: print_and_save_config(args, env_args, alg_args)

    # build environment
    env = build_env(args, env_args)

    # init session
    tf_config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tf_config)

    # build and train agent
    learn = getattr(import_module('algs.' + args.alg), 'learn')
    agent = learn(env, args.seed, args.n_total_steps, args.max_episode_length, alg_args, args)

    if args.play:
        if env_args: env_args['visualize'] = True
        env = make_single_env(args.env, args.rank, args.n_env + 100, args.seed, env_args, args.output_dir)
        obs = env.reset()
        episode_rewards = 0
        episode_steps = 0
        while True:
            i = input('press key to continue ...')
            action = agent.get_actions(obs)
            next_obs, rew, done, _ = env.step(action.flatten())
            episode_rewards += rew
            episode_steps += 1
            print('q value: {:.4f}; reward: {:.2f}; reward so far: {:.2f}'.format(
                agent.get_action_value(np.atleast_2d(obs), action).squeeze(), rew, episode_rewards))
            obs = next_obs
            env.render()
            time.sleep(0.5)
            if done:
                print('Episode length {}; cumulative reward: {:.2f}'.format(episode_steps, episode_rewards))
                episode_rewards = 0
                episode_steps = 0
                obs = env.reset()

    env.close()
    return agent


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    extra_args = parse_unknown_args(unknown)

    main(args, extra_args)
