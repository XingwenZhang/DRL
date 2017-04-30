""" This script visualizes the results from the result log
"""

import sys
from Config import Config
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_xy(x, y, xlabel, ylabel, title, dump_file, xy_range=None):
    """
    Plot the relationship between x and y
    """
    fig = plt.figure()
    if title is not None and title != '':
        fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xy_range is not None:
        ax.axis(xy_range)
    ax.plot(x, y, '-')
    plt.savefig(dump_file)


if __name__ == '__main__':
	print('parsing log...')
	file_name = Config.RESULTS_FILENAME
	fd = open(file_name)
	num_frames = []
	rewards = []
	for line in fd:
		fields = line.strip().split()
		assert(len(fields)==4)
		num_frame = int(fields[3])
		reward = int(fields[2][:-1])
		num_frames.append(num_frame)
		rewards.append(reward)

	# smooth
	rewards = np.array(rewards)
	num_frames = np.array(num_frames)
	if len(sys.argv) > 1:
		N = int(sys.argv[1])
		rewards = np.convolve(rewards, np.ones((N,))/N, mode='same')
		num_frames = np.convolve(num_frames, np.ones((N,))/N, mode='same')

	print('ploting...')
	xs = range(len(num_frames))
	plot_xy(xs, rewards, 'episode', 'reward', 'episode_reward', 'episode_reward.png')
	plot_xy(xs, num_frames, 'episode', 'number of frames', 'episode_frames', 'episode_frames.png')

	print('done.')

