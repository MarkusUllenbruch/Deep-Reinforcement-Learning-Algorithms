import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores, color='lightgreen')
    plt.plot(x, running_avg, color='green')
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

# Eventuell sollte man überlegen ob es sinnvoll ist
# a) Die Running Statistics Werte mean und std irgendwann nach ~ 100.000 timesteps während dem training einzufrieren und konstant zu halten
# oder
# b) ~ Ohne Training 50.000 timesteps im env mit random action samples laufen zu lassen und mean und std davon benutzen für normalization
class RunningStats:
    """ Running Statistics

    Recursive online update of mean, std when new measurement x comes in
    """
    def __init__(self, shape_states):
        self.n = 0
        self.old_mean = np.zeros(shape=shape_states)  #Old mean
        self.new_mean = np.zeros(shape=shape_states)  #New mean after single online update of observations
        self.old_ssd = np.zeros(shape=shape_states)   #Old Sum of Squared Distances (SSD)
        self.new_ssd = np.zeros(shape=shape_states)   #New Sum of Squared Distances (SSD)

    def clear(self):
        self.n = 0

    def online_update(self, x):
        self.n += 1  # x.shape[0]  # Updates eher auf mini-batch ebene?
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


def scale_action(action, action_min, action_max):
    '''Linear Scaling from [-1, +1] bounded action to environment actions '''

    scaled_action = (action_max - action_min) * (action + 1) / 2.0 + action_min
    return scaled_action
