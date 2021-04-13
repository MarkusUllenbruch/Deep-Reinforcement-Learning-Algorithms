import numpy as np
import gym
from agent import Agent

num_envs_iterations = 200
environment = 'BipedalWalker-v3'
array = np.zeros(shape=(10, num_envs_iterations))

batch_size = 5 #2*2048         #Pendulum 2000
mini_batch_size = 5#  2*64     #Pendulum 200
n_epochs = 10              #Pendulum 8
learning_rate = 1.5e-4    #Pendulum 0.0004

for run in range(10):
    print(run)
    env = gym.make(environment)

    agent = Agent(n_actions=env.action_space.shape[0],
                  n_states=env.observation_space.shape[0],
                  obs_shape=env.observation_space.shape,
                  mini_batch_size=mini_batch_size,
                  n_epochs=n_epochs,
                  ppo_clip=0.2,
                  entropy_coeff=3e-3,  # 0.0014
                  lr=learning_rate,
                  clip_value_loss=False,
                  normalize_observation=True,
                  fc1=600,  # 500
                  fc2=400,
                  environment=environment,
                  run=run)  # 300

    agent.load_networks()
    score_history = []
    for i in range(num_envs_iterations):
        done = False
        score = 0.0
        obs = env.reset()
        while not done:
            action = agent.choose_deterministic_action(obs)
            obs_, reward, done, _ = env.step(action)
            score += reward
            obs = obs_
        score_history.append(score)

    print(np.mean(score_history))
    array[run, :] = score_history
print(array, array.shape)
np.savetxt("PPO_Deterministic"+ environment +".csv", array, delimiter=",")
np.save("PPO_Deterministic"+ environment +".npy", array)
print(array.shape, np.mean(array), np.std(array))
print(np.mean(array, axis=1, keepdims=True))

