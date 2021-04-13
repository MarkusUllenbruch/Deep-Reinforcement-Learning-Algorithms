import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores, color='lightgreen')
    plt.plot(x, running_avg, color='green')
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


class RunningStats:
    def __init__(self, shape_states, chkpt_dir='PPO', run=0):
        self.n = 0
        self.old_mean = np.zeros(shape=shape_states)  #Old mean
        self.new_mean = np.zeros(shape=shape_states)  #New mean after single online update of observations
        self.old_ssd = np.zeros(shape=shape_states)   #Old Sum of Squared Distances (SSD)
        self.new_ssd = np.zeros(shape=shape_states)   #New Sum of Squared Distances (SSD)

        self.checkpoint_dir = 'tmp/' + chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_file_mean = os.path.join(self.checkpoint_dir, 'mean' + '_PPO_' + str(run) +'.npy')
        self.checkpoint_file_ssd = os.path.join(self.checkpoint_dir, 'ssd' + '_PPO_' + str(run) +'.npy')
        self.checkpoint_file_n = os.path.join(self.checkpoint_dir, 'n' + '_PPO_' + str(run) +'.npy')

    def clear(self):
        self.n = 0

    def online_update(self, x):
        self.n += 1 # x.shape[0]  # Updates eher auf mini-batch ebene?
        if self.n == 1:
            self.old_mean = x
            self.new_mean = x
        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.n
            self.new_ssd = self.old_ssd + (x - self.old_mean) * (x - self.new_mean)

            self.old_mean = self.new_mean
            self.old_ssd = self.new_ssd

    def mean(self):
        return self.new_mean

    def variance(self):
        return self.new_ssd / (self.n-1) if self.n > 2 else 1.0

    def std(self):
        return np.sqrt(self.variance())

    def __call__(self):
        return self.mean(), self.std()

    def save_stats(self):
        np.save(self.checkpoint_file_mean, self.new_mean)
        np.save(self.checkpoint_file_n, self.n)
        np.save(self.checkpoint_file_ssd, self.new_ssd)

    def load_stats(self):
        self.new_mean = np.load(self.checkpoint_file_mean)
        self.new_ssd = np.load(self.checkpoint_file_ssd)
        self.n = np.load(self.checkpoint_file_n)
