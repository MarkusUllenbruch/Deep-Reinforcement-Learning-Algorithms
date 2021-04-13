import numpy as np
import gym
from agent import Agent
from utils import scale_action

# Breaking Conditions
max_timesteps = 500000
max_episodes = 1000

environment = 'Pendulum-v0'

## Hyperparameter
batch_size = 2000
mini_batch_size = 200
n_epochs = 8
lr_critic = 0.0004
lr_actor = 0.0004


for run in range(1):
    env = gym.make(environment)
    action_min = env.action_space.low
    action_max = env.action_space.high
    agent = Agent(n_actions=env.action_space.shape[0],
                  n_states=env.observation_space.shape[0],
                  obs_shape=env.observation_space.shape,
                  actor_activation='tanh',  # 'tanh' or None
                  mini_batch_size=mini_batch_size,
                  n_epochs=n_epochs,
                  ppo_clip=0.2,
                  entropy_coef=0.0015,  # 0.015
                  lr_actor=lr_actor,
                  lr_critic=lr_critic,
                  normalize_obs=False,
                  clip_value_loss=True,
                  fc1=300,
                  fc2=200)

    filename = 'PPO_' + environment + '_' + str(run)

    best_score = env.reward_range[0]
    score_history = []
    timesteps_history = []

    n_learn_updates = 0
    avg_score = 0
    t_batch = 0
    t_total = 0
    n_episode = 1

    while t_total < max_timesteps and n_episode < max_episodes:
        state = env.reset()
        done = False
        score = 0
        t_episode = 0
        rewards_ep = []  # Rewards per episode
        values_ep = []  # values per episode
        dones_ep = []  # done flags per episode
        while not done:
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(
                scale_action(action, action_min=action_min, action_max=action_max)  # Scale agent action to env needs
            )

            score += reward
            agent.remember(state, action, log_prob, value, reward, done)

            rewards_ep.append(reward)
            values_ep.append(value)
            dones_ep.append(done)

            # Calc advantages per episode and save them when epsiode is over or batch size is achieved for learning
            if done or t_batch >= batch_size-1:
                _, _, next_value = agent.choose_action(next_state)
                advantage = agent.memory.calc_advantage(rewards_ep, values_ep, dones_ep, next_value)
                agent.remember_adv(advantage)

            # PPO learning Update when batch size fully filled with examples in memory
            if t_batch >= batch_size-1:
                agent.learn()
                t_batch = 0  # Set batch timesteps to zero
                n_learn_updates += 1
                break

            t_batch += 1
            t_episode += 1
            t_total += 1
            state = next_state
        score_history.append(score)
        timesteps_history.append(t_total)
        avg_score = np.mean(score_history[-100:])
        n_episode += 1

        if avg_score > best_score:
            best_score = avg_score
            agent.save_networks()

        print('episode', n_episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', t_total, 'learning_steps', n_learn_updates)

    score_timesteps_array = np.array((score_history, timesteps_history))
    np.save('Scores_' + filename, score_timesteps_array)