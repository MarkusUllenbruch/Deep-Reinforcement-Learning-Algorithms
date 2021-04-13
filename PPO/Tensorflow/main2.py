import gym
import numpy as np
from utils import scale_action


env = gym.make('Pendulum-v0')

high = env.action_space.high
low = env.action_space.low

for i in range(20):
    action = np.tanh(env.action_space.sample())
    a = scale_action(action, action_min=low, action_max=high)
    print(action, 'to' ,a)
