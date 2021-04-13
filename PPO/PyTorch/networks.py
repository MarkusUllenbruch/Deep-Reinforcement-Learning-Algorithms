import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, lr,
                 fc1_dims, fc2_dims, chkpt_dir='PPO', run=0):
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = 'tmp/' + chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'Actor' + '_PPO_' + str(run))

        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_states, fc1_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(fc1_dims, fc2_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(fc2_dims, n_actions), std=0.01)#,
            #nn.Tanh()
        )

        self.log_std = nn.Parameter(T.zeros(1, n_actions),
                                    requires_grad=True)  # log_std learnable, but independent of states

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mean = self.actor(state)
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)

        return dist, mean

    def save_actor(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_actor(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, n_states, lr, fc1_dims, fc2_dims,
                 chkpt_dir='PPO', run=0):
        super(CriticNetwork, self).__init__()

        self.checkpoint_dir = 'tmp/' + chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'Critic' + '_PPO_' + str(run))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_states, fc1_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(fc1_dims, fc2_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(fc2_dims, 1), std=1.0)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_critic(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_critic(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork2(nn.Module):
    def __init__(self, n_states, n_actions, lr,
                 fc1_dims, fc2_dims, chkpt_dir='PPO', run=0):
        super(ActorNetwork2, self).__init__()

        self.checkpoint_dir = 'tmp/' + chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'Actor' + '_PPO_' + str(run))

        self.fc1 = layer_init(nn.Linear(n_states, fc1_dims))
        self.fc2 = layer_init(nn.Linear(fc1_dims, fc2_dims))
        self.mean = layer_init(nn.Linear(fc2_dims, n_actions), std=np.sqrt(2)/100)
        self.log_std = layer_init(nn.Linear(fc2_dims, n_actions), std=np.sqrt(2)/10000)

        #self.log_std = nn.Parameter(T.zeros(1, n_actions), requires_grad=True)  # log_std learnable, but independent of states

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = nn.Tanh()(x)
        x = self.fc2(x)
        x = nn.Tanh()(x)
        log_std = self.log_std(x)
        mean = self.mean(x)

        #std = nn.Softplus()(log_std)
        std = log_std.exp()
        #print('MEAN', mean.detach())
        #print('STD', std.detach())
        #print(std)
        dist = Normal(mean, std)
        #print(dist.sample().detach())
        return dist, mean

    def save_actor(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_actor(self):
        self.load_state_dict(T.load(self.checkpoint_file))