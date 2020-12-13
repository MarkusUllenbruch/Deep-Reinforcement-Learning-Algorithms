import numpy as np
import gym

from agent import Agent
from plot_learning_curve import plot_learning_curve


learnID = '1'
n_games = 300  # 1500

# RANDOM_SEED = 90

# env_id = 'InvertedPendulumBulletEnv-v0'
# env_id = 'Pendulum-v0'
env_id = 'LunarLanderContinuous-v2'
# env_id = 'Pendulum-v0'
# env_id = 'BipedalWalker-v3'

env = gym.make(env_id)

# tf.random.set_seed(RANDOM_SEED)
# env.seed(RANDOM_SEED)
# env.action_space.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

agent = Agent(env=env)

filename_return = env_id + '_SAC_' + 'return_' + learnID
filename_alpha = env_id + '_SAC_' + 'alpha_' + learnID
figure_file_return = 'plots/' + filename_return
figure_file_alpha = 'plots/' + filename_alpha
print('ENVIRONMENT: ', env_id)

best_score = env.reward_range[0]
score_history = []
alpha_history = []
steps = 0
for i in range(n_games):
    score = 0.0
    alpha = []
    done = False
    observation = env.reset()
    while not done:
        steps += 1
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
        score += reward
        alpha.append(agent.alpha)

    score_history.append(score)
    alpha_history.append(np.mean(alpha))
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    print('episode ', i, 'score %.1f' % score, 'avg score %1.f' % avg_score, 'steps ', steps, 'alpha ', np.mean(alpha))

plot_learning_curve(score_history, figure_file_return +'.png', color='lightgreen', avg_color='green', Ylabel='Return')
plot_learning_curve(alpha_history, figure_file_alpha +'.png', color='blue', Ylabel='Temperature alpha')

np.save(figure_file_return, score_history)