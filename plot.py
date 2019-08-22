import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('logdirs', nargs='*')
parser.add_argument('--recurse', '-r', action='store_true')


def plot_from_dirs(logdirs, metric='avg_episode_length', metric_std='std_episode_length'):
    # collect data csv's
    data = []
    for logdir in logdirs:
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
        logdirs = []
        for root, d, files in os.walk(args.logdirs[0]):
            if 'log.csv' in files:
                logdirs.append(root)
    plot_from_dirs(logdirs if args.recurse else args.logdirs)

