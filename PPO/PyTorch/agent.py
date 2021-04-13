import torch as T
import torch.nn as nn
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork, ActorNetwork2
import numpy as np
from utils import RunningStats

class Agent:
    def __init__(self,
                 n_actions,
                 n_states,
                 obs_shape,
                 gamma=0.99,
                 lr=0.0003,
                 gae_lambda=0.95,
                 entropy_coeff=0.0005,
                 ppo_clip=0.2,
                 mini_batch_size=64,
                 n_epochs=10,
                 clip_value_loss=True,
                 normalize_observation=False,
                 stop_normalize_obs_after_timesteps=50000,
                 fc1=64,
                 fc2=64,
                 environment='None',
                 run=0):

        self.entropy_coeff = entropy_coeff
        self.clip_value_loss = clip_value_loss
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.normalize_observation = normalize_observation
        self.stop_obs_timesteps = stop_normalize_obs_after_timesteps
        self.timestep = 0

        self.actor = ActorNetwork(n_states=n_states, n_actions=n_actions, lr=lr, fc1_dims=fc1, fc2_dims=fc2, chkpt_dir=environment, run=run)
        self.critic = CriticNetwork(n_states=n_states, lr=lr, fc1_dims=fc1, fc2_dims=fc2, chkpt_dir=environment, run=run)
        self.memory = PPOMemory(mini_batch_size, gamma, gae_lambda)
        self.running_stats = RunningStats(shape_states=obs_shape, chkpt_dir=environment, run=run)
        # self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=1e-5)

    def remember(self, state, action, log_probs, value, reward, done):
        self.memory.store_memory(state, action, log_probs, value, reward, done)

    def remember_adv(self, advantage_list):
        self.memory.store_advantage(advantage_list)

    def save_networks(self):
        print('--saving networks--')
        self.actor.save_actor()
        self.critic.save_critic()
        if self.normalize_observation:
            self.running_stats.save_stats()

    def load_networks(self):
        print('--loading networks--')
        self.actor.load_actor()
        self.critic.load_critic()
        if self.normalize_observation:
            self.running_stats.load_stats()

    def normalize_obs(self, obs):
        mean, std = self.running_stats()
        obs_norm = (obs - mean) / (std + 1e-6)
        return obs_norm

    def choose_action(self, observation):
        if self.normalize_observation:
            if self.timestep < self.stop_obs_timesteps:
                self.running_stats.online_update(observation)
            elif self.timestep == self.stop_obs_timesteps:
                print('No online update for obs Normalization anymore')
            observation = self.normalize_obs(observation)  #Normalize Observations

        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist, _ = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        log_probs = dist.log_prob(action)
        log_probs = T.sum(log_probs, dim=1, keepdim=True).squeeze().detach().cpu().numpy()

        value = T.squeeze(value).item()

        # action = T.squeeze(action).detach().numpy()
        if action.shape[0] == 1 and action.shape[1] == 1:
            action = action.detach().cpu().numpy()[0].reshape(1, )
        else:
            action = T.squeeze(action).detach().cpu().numpy()
        self.timestep += 1

        return action, log_probs, value

    def choose_deterministic_action(self, observation):
        if self.normalize_observation:
            observation = self.normalize_obs(observation)  #Normalize Observations

        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        _, mean = self.actor(state)
        action = T.squeeze(mean).detach().cpu().numpy() #.reshape(1, )
        return action

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, advantage_arr, batches = \
                self.memory.generate_batches()

            if self.normalize_observation:
                #print(state_arr[0:5,:])
                state_arr = self.normalize_obs(state_arr)
                #print(state_arr[0:5,:])

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_log_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device).detach()
                actions = T.tensor(action_arr[batch]).to(self.actor.device).detach()
                critic_value_old = T.tensor(vals_arr[batch]).to(self.actor.device).detach()
                advantage = T.tensor(advantage_arr[batch]).to(self.actor.device)
                #returns = T.tensor(reward_arr[batch]).to(self.actor.device)
                #advantage = returns - critic_value_old

                # Advantage Normalization per Mini-Batch
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                advantage = advantage.detach()

                ## Actor-Loss
                dist, _ = self.actor(states)
                critic_value_new = self.critic(states)
                critic_value_new = T.squeeze(critic_value_new)

                new_log_probs = dist.log_prob(actions)
                new_log_probs = T.sum(new_log_probs, dim=1, keepdim=True).squeeze()

                prob_ratio = (new_log_probs - old_log_probs).exp()

                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.ppo_clip,
                                                 1 + self.ppo_clip) * advantage
                ppo_surr_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                entropy_loss = - self.entropy_coeff * dist.entropy().mean()
                actor_loss = ppo_surr_loss + entropy_loss

                ## Critic-Loss
                returns = advantage + critic_value_old
                # Clipping Value Loss
                if self.clip_value_loss:
                    v_loss_unclipped = ((critic_value_new - returns) ** 2)
                    v_clipped = critic_value_old + T.clamp(critic_value_new - critic_value_old, -self.ppo_clip,
                                                           self.ppo_clip)
                    v_loss_clipped = (v_clipped - returns) ** 2
                    v_loss_max = T.max(v_loss_unclipped, v_loss_clipped)
                    critic_loss = 0.5 * v_loss_max.mean()
                else:
                    critic_loss = 0.5 * ((critic_value_new - returns) ** 2).mean()

                ## Backprop Actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.actor.parameters(),
                                         max_norm=0.5,
                                         norm_type=2)
                self.actor.optimizer.step()

                ## Backprop Critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.critic.parameters(),
                                         max_norm=0.5,
                                         norm_type=2)
                self.critic.optimizer.step()

                # loss = critic_loss + actor_loss
                # self.optimizer.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(parameters=list(self.actor.parameters()) + list(self.critic.parameters()),
                #                        max_norm=0.8,
                #                        norm_type=2)
                # self.optimizer.step()

        self.memory.clear_memory()  # Clear Memory to save new samples for next iteration


