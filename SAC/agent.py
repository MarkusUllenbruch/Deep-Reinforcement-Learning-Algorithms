import numpy as np
import tensorflow as tf
from networks import CriticNetwork, ActorNetwork
from replay_buffer import ReplayBuffer


class Agent:
    """ 2019 State-of-the-Art Implementation of SAC with optimized temperature

    """
    def __init__(self,
                 env,
                 lr_Q = 3e-4,
                 lr_actor = 3e-4,
                 lr_a = 3e-4,
                 gamma=0.99,
                 tau=0.005,
                 layer1_size=256,
                 layer2_size=256,
                 batch_size=256,
                 max_size=1000000,
                 warmup=1000,
                 policy_delay=1,
                 minimum_entropy=None):

        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]

        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.min_action = env.action_space.low
        self.max_action = env.action_space.high

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup = warmup
        self.time_step = 0
        self.update_step = 0
        self.policy_delay = policy_delay

        self.policy_net = ActorNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                       fc1_dims=layer1_size, fc2_dims=layer2_size, network_name='Actor')

        self.q_net1 = CriticNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                    hidden_neurons_1=layer1_size, hidden_neurons_2=layer2_size, network_name='Critic_1')

        self.q_net2 = CriticNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                    hidden_neurons_1=layer1_size, hidden_neurons_2=layer2_size, network_name='Critic_2')

        self.target_q_net1 = CriticNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                           hidden_neurons_1=layer1_size, hidden_neurons_2=layer2_size, network_name='Target_Critic_1')

        self.target_q_net2 = CriticNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                           hidden_neurons_1=layer1_size, hidden_neurons_2=layer2_size, network_name='Target_Critic_2')

        self.replay_buffer = ReplayBuffer(n_actions=self.n_actions,
                                          n_states=self.n_states,
                                          memory_size=max_size)

        self.policy_net.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_actor))
        self.q_net1.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_Q))
        self.q_net2.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_Q))

        self.update_target_networks(tau=1)  # copy parameters to target networks

        # entropy temperature parameter alpha
        # self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        print(-tf.constant(env.action_space.shape[0], dtype=tf.float32))

        self.log_alpha = tf.Variable(tf.zeros(1), trainable=True)
        self.minimum_entropy = -tf.reduce_prod(tf.convert_to_tensor(env.action_space.shape, dtype=tf.float32))
        self.minimum_entropy = -tf.reduce_prod(tf.convert_to_tensor(env.action_space.shape, dtype=tf.float32)) if minimum_entropy is None else minimum_entropy
        print('Minimum Entropy set to: ', self.minimum_entropy)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_a)
        self.alpha = tf.exp(self.log_alpha).numpy()
        print('alpha: ', self.alpha)


    def choose_action(self, state):
        if self.time_step < self.warmup:
            actions = np.random.uniform(low=-1.0, high=1.0, size=self.n_actions)  # "random uniform distribution over all valid actions"
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        else:
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            state = tf.expand_dims(state, axis=0)
            actions, _ = self.policy_net(state)

        self.time_step += 1
        if self.time_step == self.warmup:
            print('No warmup anymore!')
        a = self.rescale_action(actions[0].numpy())
        return a

    def scale_action(self, action):
        """ Scale all actions to [-1., +1.]

        :param action: unscaled actions
        :return: scaled actions all in range -1. .. +1.
        """
        # old = 2 * (action - self.min_action) / (self.max_action - self.min_action) - 1.0
        scale = (2 * action - (self.action_range[1] + self.action_range[0])) / \
                    (self.action_range[1] - self.action_range[0])
        return scale

    def rescale_action(self, action):
        """ Rescale all scaled actions to environment actionspace values

        :param action: scaled actions
        :return: rescaled actions all in range min_action .. max_action
        """
        # old = (action + 1.0) * (self.max_action - self.min_action) / 2.0 + self.min_action
        rescale = action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
                  (self.action_range[1] + self.action_range[0]) / 2.0
        return rescale

    def remember(self, state, action, reward, new_state, done):
        action = self.scale_action(action)  # ÄNDERUNG! Funktioniert das mit?
        self.replay_buffer.store_environment_transition(state, action, reward, new_state, done)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        for theta_target, theta in zip(self.target_q_net1.get_weights(),
                                       self.q_net1.get_weights()):
            theta_target = tau * theta + (1 - tau) * theta_target
            weights.append(theta_target)
        self.target_q_net1.set_weights(weights)

        weights = []
        for theta_target, theta in zip(self.target_q_net2.get_weights(),
                                       self.q_net2.get_weights()):
            theta_target = tau * theta + (1 - tau) * theta_target
            weights.append(theta_target)
        self.target_q_net2.set_weights(weights)

        # weights = []
        # theta_target = self.target_q_net1.weights
        # for i, theta in enumerate(self.q_net1.weights):
        #    weights.append(tau*theta + (1-tau)*theta_target[i])
        # self.target_q_net1.set_weights(weights)
        #
        # weights = []
        # theta_target = self.target_q_net2.weights
        # for i, theta in enumerate(self.q_net2.weights):
        #    weights.append(tau*theta + (1-tau)*theta_target[i])
        # self.target_q_net2.set_weights(weights)

    def save_models(self):
        print('models saved')  # To Do!

    def load_models(self):
        print('models loaded')  # To Do!

    def learn(self):
        if self.replay_buffer.count < self.batch_size:
            return
        elif self.replay_buffer.count == self.batch_size:
            print('Buffer Size equals batch Size! - Learning begins!')
            return

        # sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_from_buffer(batch_size=self.batch_size)

        # convert batchs from 2D numpy arrays to tensorflow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # expand rewards and dones from 1D numpy arrays to 2D tensors and reshape them
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.expand_dims(rewards, axis=0)
        rewards = tf.reshape(rewards, [self.batch_size, 1])
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        dones = tf.expand_dims(dones, axis=0)
        dones = tf.reshape(dones, [self.batch_size, 1])

        ## Update critic networks Q1 & Q2
        with tf.GradientTape(persistent=True) as tape_Q:
            next_actions, next_log_pi = self.policy_net(next_states)
            Q1_next = self.target_q_net1(next_states, next_actions)
            Q2_next = self.target_q_net2(next_states, next_actions)
            next_q_target = tf.minimum(Q1_next, Q2_next) - self.alpha * next_log_pi
            expected_q = tf.stop_gradient(rewards + (1 - dones) * self.gamma * next_q_target)

            curr_q1 = self.q_net1(states, actions)
            curr_q2 = self.q_net2(states, actions)

            q1_loss = tf.reduce_mean((curr_q1 - expected_q) ** 2)
            q2_loss = tf.reduce_mean((curr_q2 - expected_q) ** 2)  # tf.square()
            q_loss = q1_loss + q2_loss

        grad_Q1 = tape_Q.gradient(q_loss, self.q_net1.trainable_variables)
        grad_Q2 = tape_Q.gradient(q_loss, self.q_net2.trainable_variables)

        self.q_net1.optimizer.apply_gradients(zip(grad_Q1, self.q_net1.trainable_variables))
        self.q_net2.optimizer.apply_gradients(zip(grad_Q2, self.q_net2.trainable_variables))

        ## Update policy network and polyak update target Q networks less frequently (like in TD3 --> "Delayed SAC")
        if self.update_step % self.policy_delay == 0:
            with tf.GradientTape() as tape_policy:
                new_actions, log_pi = self.policy_net(states)
                Q1 = self.q_net1(states, new_actions)
                Q2 = self.q_net2(states, new_actions)
                Q_min = tf.minimum(Q1, Q2)
                loss_policy = tf.reduce_mean(self.alpha * log_pi - Q_min)

            grad_policy = tape_policy.gradient(loss_policy, self.policy_net.trainable_variables)
            self.policy_net.optimizer.apply_gradients(zip(grad_policy, self.policy_net.trainable_variables))

            self.update_target_networks()  # update target networks with polyak averaging

        ## Update temperature parameter alpha
        with tf.GradientTape() as tape:
            _, log_pi_a = self.policy_net(states)
            alpha_loss = tf.reduce_mean(self.log_alpha * (-log_pi_a - self.minimum_entropy))

        grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(grads, [self.log_alpha]))
        self.alpha = tf.exp(self.log_alpha).numpy()

        self.update_step += 1  # Keep track of the number of network updates
