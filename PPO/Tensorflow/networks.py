import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
import tensorflow_probability as tfp
import os
# No sharing weights between Actor and Critic Network


class ActorNetwork(tf.keras.Model):
    def __init__(self, n_states, n_actions,
                 fc1_dims, fc2_dims, actor_activation, directory='tmp/PPO'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(directory, 'Actor_PPO')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.checkpoint_dir = directory

        # Orthogonal Weight initialization + Zero Bias initialization (Implementation Details)
        self.fc1 = Dense(fc1_dims,
                         input_shape=(n_states, ),
                         activation='tanh',
                         kernel_initializer=initializers.orthogonal(np.sqrt(2)),
                         bias_initializer=initializers.constant(0.0))

        self.fc2 = Dense(fc2_dims,
                         activation='tanh',
                         kernel_initializer=initializers.orthogonal(np.sqrt(2)),
                         bias_initializer=initializers.constant(0.0))

        # Last Layer weights ~100 times smaller (Implementation Details)
        self.fc3 = Dense(n_actions,
                         activation=actor_activation,
                         kernel_initializer=initializers.orthogonal(0.01),
                         bias_initializer=initializers.constant(0.0))

        # log_std weights learnable & independent of states (Implementation Detail)
        self.log_std = tf.Variable(tf.zeros((1, n_actions), dtype=tf.float32), trainable=True)


    def call(self, state):
        mean = self.fc1(state)
        mean = self.fc2(mean)
        mean = self.fc3(mean)
        log_std = tf.broadcast_to(self.log_std, shape=mean.shape)
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean, std)
        return dist, mean


class CriticNetwork(tf.keras.Model):
    def __init__(self, n_states, fc1_dims, fc2_dims,
                 directory='tmp/PPO'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(directory, 'Critic_PPO')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.checkpoint_dir = directory

        self.fc1 = Dense(fc1_dims,
                         input_shape=(n_states, ),
                         activation='tanh',
                         kernel_initializer=initializers.orthogonal(np.sqrt(2)),
                         bias_initializer=initializers.constant(0.0))

        self.fc2 = Dense(fc2_dims,
                         activation='tanh',
                         kernel_initializer=initializers.orthogonal(np.sqrt(2)),
                         bias_initializer=initializers.constant(0.0))

        self.fc3 = Dense(1,
                         kernel_initializer=initializers.orthogonal(1.0),
                         bias_initializer=initializers.constant(0.0))


    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        return self.fc3(value)
