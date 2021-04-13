import numpy as np
import torch as T

class PPOMemory:
    def __init__(self, batch_size, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.T = T
        self.states = []
        self.log_probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.batch_size = batch_size

    def calc_advantage(self, rewards, values, dones, next_value):
        '''Advantage calculation of one Trajectory'''
        #print(dones[-4:])
        advantage = np.zeros(len(rewards), dtype=np.float32)
        values = values + [next_value]
        for t in range(len(rewards) ):
            discount = 1
            delta = 0
            for k in range(t, len(rewards) ):
                delta += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = delta
        return list(advantage)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        ###################################

        #test = [0]
        #for j in range(len(self.dones)-1):
        #    if self.dones[j] == False and self.dones[j+1] == True:
        #        test.append(j+1)
        #        print(self.rewards[j+1])
        #        print(self.dones[j+1])
        #test.append(len(self.dones)-1)
        #print(test)
        #print(self.termination_step)
        #print(len(self.rewards))
        #for j in self.termination_step:
        #    print(self.rewards[j])
        ##################################

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                np.array(self.advantages),\
                batches

    def store_memory(self, state, action, log_probs, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def store_advantage(self, adv):
        self.advantages = self.advantages + adv

    def clear_memory(self):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []

