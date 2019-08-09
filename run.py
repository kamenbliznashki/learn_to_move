import os
import sys
import time
import argparse
import pprint
import multiprocessing as mp
from importlib import import_module

import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gym
from env_wrappers import DummyVecEnv, SubprocVecEnv, Monitor, L2M2019EnvBaseWrapper
from logger import save_json

parser = argparse.ArgumentParser()
parser.add_argument('--play', action='store_true')

parser.add_argument('env', type=str, help='Environment id (e.g. HalfCheetah-v2 or L2M2019).')
parser.add_argument('alg', type=str, help='Algorithm name -- module where agent and learn function reside.')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--exp_name', default='exp', type=str)
# training params
parser.add_argument('--n_env', default=1, type=int, help='Number of environments in parallel.')
parser.add_argument('--n_total_steps', default=1000, type=int, help='Number of training steps on single or vectorized environment.')
parser.add_argument('--max_episode_length', default=1000, type=int, help='Reset episode after reaching max length.')
# logging
parser.add_argument('--load_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--save_interval', default=1000, type=int)
parser.add_argument('--log_interval', default=1000, type=int)



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



def make_single_env(env_name, subrank, seed, env_args, output_dir):
    # env_kwargs serve to initialize L2M2019Env
    #   L2M2019Env default args are: visualize=True, integrator_accuracy=5e-5, difficulty=2, seed=0, report=None
    #   additionally here: env_kwargs include `model` which can be '2D' or '3D'
    #   NOTE -- L2M2019Env uses seed in reseting the velocity target map in VTgtField.reset(seed) in v_tgt_field.py

    if env_name == 'L2M2019':
        env = L2M2019EnvBaseWrapper(**env_args)
    else:
        env = gym.envs.make(env_name)
        env.seed(seed+subrank if seed is not None else None)

    # apply wrappers
    env = Monitor(env, os.path.join(output_dir, str(subrank)))

    return env

def build_env(args, env_args):
    ncpu = mp.cpu_count()
    if sys.platform == 'darwin': ncpu // 2
    nenv = args.n_env or ncpu

    def make_env(rank):
        return lambda: make_single_env(args.env, rank, args.seed, env_args, args.output_dir)

    if nenv > 1:
        return SubprocVecEnv([make_env(i) for i in range(nenv)])
    else:
        return DummyVecEnv([make_env(i) for i in range(nenv)])




def main(args, extra_args):
    # configure args for environment and algorithm; update defaults with extra_args
    env_args = {'model': '2D', 'visualize': False, 'integrator_accuracy': 1e-3, 'difficulty': 2} if args.env=='L2M2019' else None
    if extra_args is not None and env_args is not None:
        env_args.update({k: v for k, v in extra_args.__dict__.items() if k in env_args})
    alg_args = getattr(import_module('algs.' + args.alg), 'defaults')(args.env)
    alg_args.update({k: v for k, v in extra_args.__dict__.items() if k in alg_args})

    # setup logging
    if args.load_path:
        args.output_dir = os.path.dirname(args.load_path)
    if not args.output_dir:  # if not given use results/file_name/time_stamp
        logdir = args.exp_name + '_' + args.env + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join('results', logdir)
        os.makedirs(args.output_dir)

    # print and save config
    print('Building environment and agent with the following config:')
    print('Run config:\n' + pprint.pformat(args.__dict__))
    print('Env config: ' + args.env + (env_args is not None)*('\n' + pprint.pformat(args.__dict__)))
    print('Alg config: ' + args.alg + '\n' + pprint.pformat(alg_args))
    save_json(args.__dict__, os.path.join(args.output_dir, 'config_run.json'))
    if env_args: save_json(env_args, os.path.join(args.output_dir, 'config_env.json'))
    save_json(alg_args, os.path.join(args.output_dir, 'config_alg.json'))

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
        env = make_single_env(args.env, args.n_env + 1, args.seed, env_args, args.output_dir)
        obs = env.reset()
        episode_rewards = 0
        while True:
            action = agent.get_actions(obs)
            obs, rew, done, _ = env.step(action.flatten())
            episode_rewards += rew
            env.render()
            if done:
                print('Episode rewards: ', episode_rewards)
                episode_rewards = 0
                obs = env.reset()

    env.close()
    return agent


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    extra_args = parse_unknown_args(unknown)

    main(args, extra_args)
