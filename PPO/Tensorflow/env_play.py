import gym
from agent import Agent

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')

    agent = Agent(n_actions=env.action_space.shape[0],
                  n_states=env.observation_space.shape[0],
                  obs_shape=env.observation_space.shape,
                  fc1=300,
                  fc2=200)

    agent.load_networks()

    for i in range(10):
        observation = env.reset()
        env.render()
        done = False
        score = 0
        while not done:
            action, _, _ = agent.choose_action(observation, training=False)
            observation_, reward, done, _ = env.step(action)
            env.render()
            observation = observation_
            score += reward
        print(score)
        score = 0
    env.close()