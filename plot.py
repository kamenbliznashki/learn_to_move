import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('logdirs', nargs='*')
parser.add_argument('--recurse', '-r', action='store_true')
parser.add_argument('--metric', '-m', default='episode_length', type=str, help='Metric to plot from log.csv; `avg_` and `std_` are automatically prepended.')


def plot_from_dirs(logdirs, metric='avg_episode_length', metric_std='std_episode_length'):
    # collect data csv's
    data = []
    for logdir in logdirs:
        if not os.path.exists(os.path.join(logdir, 'log.csv')):
            print('No log.csv found in ', logdir)
            continue

        df = pd.read_csv(os.path.join(logdir, 'log.csv'))
        df = df[['timestep', metric, metric_std]]

        x, y, std = df['timestep'], df[metric], np.maximum(df[metric_std], 1)

        label = logdir.strip('/').rpartition('/')[2]
        ax = sns.lineplot(data=df, x='timestep', y=metric, err_style='band', label=label)
        ax.fill_between(x, y - std, y + std, alpha=0.2)

    plt.legend(fontsize=8)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.recurse:
        root = args.logdirs.pop()
        for root, d, files in os.walk(args.logdirs[0]):
            if 'log.csv' in files:
                args.logdirs.append(root)
    # check if logdir passed is for type ./results/abc*
    if '*' in args.logdirs[0]:
        root_dir, _, name = args.logdirs[0].rpartition('/')
        name_key = name.rstrip('*')
        # os.walk returns tuples of (root, dirs, files);
        # take 0th entry output by os.walk ie the root; then take the 1st element of the tuple ie dirs under the root
        subdirs = list(os.walk(root_dir))[0][1]
        args.logdirs = [os.join(root_dir, s) for s in subdirs if name_key in s]
    plot_from_dirs(args.logdirs, 'avg_' + args.metric, 'std_' + args.metric)

