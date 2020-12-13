import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
import tensorflow_probability as tfp
import tensorflow.keras.losses as losses
import os


class CriticNetwork(tf.keras.Model):
    def __init__(self, n_states, n_actions, fc1_dims, fc2_dims, network_name, chkpt_dir='tmp/SAC', init_w=3e-3):
        super(CriticNetwork, self).__init__()

        self.network_name = network_name
        self.checkpoint_dir = chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, network_name + '_SAC')

        self.fc1 = Dense(units=fc1_dims, activation='relu', input_shape=(n_states + n_actions, ))
        self.fc2 = Dense(units=fc2_dims, activation='relu')
        self.q = Dense(units=1,
                       kernel_initializer=initializers.RandomUniform(minval=-init_w, maxval=init_w), # Änderung
                       bias_initializer=initializers.RandomUniform(minval=-init_w, maxval=init_w))

    def call(self, state, action):
        inputs = tf.concat([state, action], axis=1)
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.q(x)


class ActorNetwork(tf.keras.Model):
    def __init__(self,
                 n_states,
                 n_actions,
                 fc1_dims,
                 fc2_dims,
                 network_name,
                 chkpt_dir='tmp/SAC',
                 init_w=3e-3,
                 log_std_min=-20,
                 log_std_max=2):
        super(ActorNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network_name = network_name
        self.checkpoint_dir = chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, network_name + '_SAC')

        self.fc1 = Dense(units=fc1_dims, activation='relu', input_shape=(n_states, ))
        self.fc2 = Dense(units=fc2_dims, activation='relu')

        self.mu = Dense(units=n_actions,
                        kernel_initializer=initializers.RandomUniform(minval=-init_w, maxval=init_w),
                        bias_initializer=initializers.RandomUniform(minval=-init_w, maxval=init_w))

        self.log_std = Dense(units=n_actions,
                             kernel_initializer=initializers.RandomUniform(minval=-init_w, maxval=init_w),
                             bias_initializer=initializers.RandomUniform(minval=-init_w, maxval=init_w))


    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)

        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, clip_value_min=self.log_std_min, clip_value_max=self.log_std_max)
        std = tf.exp(log_std)

        normal = tfp.distributions.Normal(mu, std)  # make Gaussian distribution of mu, and sigma for actions

        z = normal.sample()  # sample from distribution with reparam trick
        action = tf.tanh(z)  # bound actions to [-1, +1]

        # correct log_probs because of bounding the actions
        log_prob = normal.log_prob(z) - tf.math.log(1 - tf.math.square(action) + 1e-6)  # Ist dasselbe!
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True) # tf.reduce_sum() or tf.reduce_mean()

        return action, log_prob