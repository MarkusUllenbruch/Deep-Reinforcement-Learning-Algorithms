import numpy as np
import os
import matplotlib.pyplot as plt

def plot_learning_curve(scores, figure_file, Ylabel, color, avg_color , plot_folder='./plots', Xlabel='Episodes'):
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):i + 1])

    plt.figure()
    plt.plot(scores, color=color)
    plt.plot(running_avg, color=avg_color)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.savefig(figure_file)