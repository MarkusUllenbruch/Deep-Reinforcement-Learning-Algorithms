import gym
import numpy as np
from agent import Agent

max_episodes = 8000
environment = 'BipedalWalker-v3'

batch_size = 10000          #Pendulum 2200   10k
mini_batch_size = 250      #Pendulum 200 250 hat gelöst
n_epochs = 10               #Pendulum 9
learning_rate = 3.0e-4       #Pendulum 0.0004 # Walker 3.4


for run in range(10):
    env = gym.make(environment)
    agent = Agent(n_actions=env.action_space.shape[0],
                  n_states=env.observation_space.shape[0],
                  obs_shape=env.observation_space.shape,
                  mini_batch_size=mini_batch_size,
                  n_epochs=n_epochs,
                  ppo_clip=0.2,
                  entropy_coeff=0.005,  # 0.005
                  lr=learning_rate,
                  clip_value_loss=False,
                  normalize_observation=True,
                  stop_normalize_obs_after_timesteps=800000, # 600000
                  fc1=600,   # Pendulum 400
                  fc2=400,   # Pendulum 300
                  environment=environment,
                  run=run)

    filename = 'PPO_' + environment + '_' + str(run)

    best_score = env.reward_range[0]
    score_history = []
    timesteps_history = []

    n_learn_updates = 0
    avg_score = 0
    t_batch = 0
    t_total = 0
    n_episode = 1

    while n_episode < max_episodes:
        state = env.reset()
        done = False
        score = 0
        t_episode = 0
        rewards_ep = []
        values_ep = []
        dones_ep = []
        while not done:
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            #env.render()
            score += reward
            agent.remember(state, action, log_prob, value, reward, done)

            rewards_ep.append(reward)
            values_ep.append(value)
            dones_ep.append(done)

            if done or t_batch >= batch_size-1:
                _, _, next_value = agent.choose_action(next_state)
                advantage = agent.memory.calc_advantage(rewards_ep, values_ep, dones_ep, next_value)
                agent.remember_adv(advantage)


            if t_batch >= batch_size-1:  #PPO learn Update after -time_horizon- timesteps of environment steps
                agent.learn()
                t_batch = 0
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

        print('episode', n_episode, ' score %.1f' % score, ' avg score %.1f' % avg_score,
                ' episode steps', t_episode, ' time_steps', t_total, ' learning_steps', n_learn_updates)

    score_timesteps_array = np.array((score_history, timesteps_history))
    np.save('Scores_' + filename, score_timesteps_array)