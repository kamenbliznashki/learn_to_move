import os
import sys
import re
import time
import argparse
import pprint
import subprocess
from multiprocessing import Pool
from functools import partial
from itertools import product

import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gym
from run import parse_unknown_args, get_alg_config, get_env_config, print_and_save_config
from logger import save_json


parser = argparse.ArgumentParser()

parser.add_argument('env', type=str, help='Environment id (e.g. HalfCheetah-v2 or L2M2019).')
parser.add_argument('alg', type=str, help='Algorithm name -- module where agent and learn function reside.')
parser.add_argument('--n_workers', default=1, type=int, help='Number of experiments to run in parallel.')
parser.add_argument('--range_args', type=str, nargs='*', help='Names of args for which a range if hyperparams is to be tested \
                                                            (all other args are tested only at discrete values).')
# training params
parser.add_argument('--n_env', default=1, type=int, help='Number of environments in parallel.')
parser.add_argument('--n_total_steps', default=1000, type=int, help='Number of training steps on single or vectorized environment.')
# logging
parser.add_argument('--output_dir', type=str)
parser.add_argument('--save_interval', default=1000, type=int)
parser.add_argument('--log_interval', default=1000, type=int)


def maybe_replace_extra_args_with_samples(range_args, extra_args, n_samples=3):
    if range_args:
        for arg in range_args:
            if arg in extra_args:
                low, high = extra_args.__dict__[arg]
                print('Sampling `{}` uniformly from [{},{})'.format(arg, low, high))
                extra_args.__dict__[arg] = np.random.uniform(low, high, size=(n_samples,))

def dict2str(d):
    out = ' '.join('--{} {}'.format(k, v) for k, v in d.items())
    # remove characters like ( , [
    return  re.sub('[(),\[\]]+', '', out)



def launch_training_job(parent_dir, args, extra_args):
    # create exp folder
    job_name = '_'.join('{}{}'.format(k[:5], str(v)[:5] if not (isinstance(v, list) or isinstance(v, tuple)) else str(v[0])[:5]) for k, v in extra_args.items())

    # create new folder in parent dir with unique name job_name
    args.output_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load defaults and configure args for environment and algorithm; update defaults with extra_args
    env_args = get_env_config(args.env, extra_args)
    alg_args = get_alg_config(args.alg, args.env, extra_args)

    # launch training with this config
    cmd = 'python run.py ' + dict2str(args.__dict__) + ' ' + dict2str(alg_args)
    if env_args is not None: cmd += ' ' + dict2str(env_args)

#    print(cmd)
    subprocess.run(cmd, shell=True, check=True)



if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    extra_args = parse_unknown_args(unknown)
    # replace range_args with uniform samples from the ranges provided
    maybe_replace_extra_args_with_samples(args.range_args, extra_args)

    pprint.pprint(args)
    pprint.pprint(extra_args)

    # combine arg ranges into a dict of products:
    # extra_args = {'a': [1,2,3], 'b': [4,5]}
    # return = [{'a': 1, 'b': 4}, {'a': 1, 'b': 5}, {'a': 2, 'b': 4}, ..., {'a': 3, 'b': 5}]
    extra_args.__dict__ = {k: [v] if isinstance(v, float) or isinstance(v, int) else v for k, v in extra_args.__dict__.items()}
    extra_args_products = list(dict(zip(extra_args.__dict__.keys(), vals)) for vals in list(product(*extra_args.__dict__.values())))
    print('Running configs for the following extra args:\n', pprint.pformat(extra_args_products, indent=2))

    parent_dir = args.output_dir
    n_workers = args.__dict__.pop('n_workers', 1)
    del args.range_args

    with Pool(processes=n_workers) as pool:
        pool.map(partial(launch_training_job, parent_dir, args), extra_args_products)
#    for extra_args in extra_args_products:
#        launch_training_job(parent_dir, job_name, args, extra_args)
