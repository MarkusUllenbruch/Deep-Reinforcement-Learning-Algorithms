import tensorflow as tf
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork
from utils import RunningStats


class Agent:
    """ Proximal Policy Optimization Algorithm
    - https://arxiv.org/abs/1707.06347

    Implementation Details:
    - https://costa.sh/blog-the-32-implementation-details-of-ppo.html
    - https://arxiv.org/abs/2005.12729
    """
    def __init__(self,
                 n_actions,
                 n_states,
                 obs_shape,
                 actor_activation,
                 gamma=0.99,
                 lr_actor=0.0003,
                 lr_critic=0.0003,
                 gae_lambda=0.95,
                 entropy_coef=0.0005,
                 ppo_clip=0.2,
                 mini_batch_size=64,
                 n_epochs=10,
                 fc1=64,
                 fc2=64,
                 clip_value_loss=True,
                 clip_gradients=True,
                 normalize_obs=False):

        self.entropy_coef = entropy_coef
        self.clip_value_loss = clip_value_loss
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.normalize_obs = normalize_obs
        self.clip_gradients = clip_gradients
        self.learning_step = 0

        ## Actor Network
        self.actor = ActorNetwork(n_states=n_states,
                                  n_actions=n_actions,
                                  actor_activation=actor_activation,
                                  fc1_dims=fc1,
                                  fc2_dims=fc2)
        self.actor.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_actor))

        ## Critic Network
        self.critic = CriticNetwork(n_states=n_states,
                                    fc1_dims=fc1,
                                    fc2_dims=fc2)
        self.critic.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_critic))

        ## Memory Initialization
        self.memory = PPOMemory(mini_batch_size, gamma, gae_lambda)

        ## Running statistics Initialization for observation normalization
        self.running_stats = RunningStats(shape_states=obs_shape)



    def remember(self, state, action, log_probs, value, reward, done):
        self.memory.store_memory(state, action, log_probs, value, reward, done)

    def remember_adv(self, advantage_list):
        self.memory.store_advantage(advantage_list)

    def save_networks(self):
        print('--saving networks--')
        #if self.learn_step >= 1:
        #    self.actor.save(self.actor.checkpoint_dir)  # Klappt nicht zu speichern mit Subclassing Model
        #    self.critic.save(self.critic.checkpoint_dir)

    def load_networks(self):
        print('--loading networks--')
        # -- TO DO --

    def normalize_observation(self, obs):
        mean, std = self.running_stats()
        obs_norm = (obs - mean) / (std + 1e-6)
        return obs_norm

    def choose_action(self, observation):
        if self.normalize_obs:
            self.running_stats.online_update(observation)
            observation = self.normalize_observation(observation)  # Normalize Observations

        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        dist, mean = self.actor(state)

        value = self.critic(state)
        value = tf.squeeze(value)

        action = dist.sample()
        log_probs = dist.log_prob(action)
        log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
        log_probs = tf.squeeze(log_probs).numpy()

        if action.shape[0] == 1 and action.shape[1] == 1:
            action = action.numpy()[0].reshape(1, )
        else:
            action = tf.squeeze(action).numpy()

        return action, log_probs, value

    def learn(self):
        """Implementing the PPO Surrogate Loss Learning Algorithm"""

        # Iterating over whole Batch training self.n_epochs times
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, advantage_arr, minibatches = self.memory.generate_batches()

            # Normalize Observations of whole Batch
            if self.normalize_obs:
                state_arr = self.normalize_observation(state_arr)

            # PPO weight update over all minibatches
            for minibatch in minibatches:

                states = tf.convert_to_tensor(state_arr[minibatch], dtype=tf.float32)
                old_log_probs = tf.convert_to_tensor(old_prob_arr[minibatch], dtype=tf.float32)
                actions = tf.convert_to_tensor(action_arr[minibatch], dtype=tf.float32)
                critic_value_old = tf.convert_to_tensor(vals_arr[minibatch], dtype=tf.float32)
                advantage = tf.convert_to_tensor(advantage_arr[minibatch], dtype=tf.float32)

                # Advantage Normalization per Mini-Batch Level (Implementation Detail)
                advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

                ## Actor-Loss
                with tf.GradientTape() as tape_actor:
                    dist, _ = self.actor(states)

                    new_log_probs = dist.log_prob(actions)
                    new_log_probs = tf.squeeze(tf.reduce_sum(new_log_probs, axis=1, keepdims=True))

                    prob_ratio = tf.exp(new_log_probs - old_log_probs)

                    surr_1 = advantage * prob_ratio
                    surr_2 = advantage * tf.clip_by_value(prob_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
                    ppo_surr_loss = - tf.reduce_mean(tf.minimum(surr_1, surr_2))
                    ppo_entropy_loss = - tf.reduce_mean(self.entropy_coef * dist.entropy())
                    actor_loss = ppo_surr_loss + ppo_entropy_loss
                grad_actor = tape_actor.gradient(actor_loss, self.actor.trainable_variables)

                # Global Gradient Clipping by L2-Norm (Implementation Detail)
                if self.clip_gradients:
                    grad_actor, _ = tf.clip_by_global_norm(grad_actor, clip_norm=0.5)

                # Apply Weight Updates by Optimizer
                self.actor.optimizer.apply_gradients(zip(grad_actor, self.actor.trainable_variables))


                ## Critic-Loss
                with tf.GradientTape() as tape_critic:
                    critic_value_new = self.critic(states)
                    critic_value_new = tf.squeeze(critic_value_new)
                    returns = advantage + critic_value_old

                    if self.clip_value_loss:
                        # Clipping Value Loss (Implementation Detail)
                        v_loss_unclipped = ((critic_value_new - returns) ** 2)
                        v_clipped = critic_value_old + tf.clip_by_value(
                            critic_value_new - critic_value_old, -self.ppo_clip, self.ppo_clip
                        )
                        v_loss_clipped = (v_clipped - returns) ** 2
                        v_loss_max = tf.maximum(v_loss_unclipped, v_loss_clipped)
                        critic_loss = 0.5 * tf.reduce_mean(v_loss_max)
                    else:
                        # Standard MSE Loss of Critic (PPO Paper Style)
                        critic_loss = 0.5 * tf.reduce_mean((critic_value_new - returns) ** 2)
                grad_critic = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

                # Global Gradient Clipping by L2-Norm (Implementation Detail)
                if self.clip_gradients:
                    grad_critic, _ = tf.clip_by_global_norm(grad_critic, clip_norm=0.5)

                # Apply Weight Updates by Optimizer
                self.critic.optimizer.apply_gradients(zip(grad_critic, self.critic.trainable_variables))

                self.learning_step += 1

        self.memory.clear_memory()  # Clear Memory for new sample Rollouts for next PPO iteration
