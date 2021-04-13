import numpy as np

class ReplayBuffer:
    def __init__(self, n_actions, n_states, memory_size=1000000):

        self.memory_size = memory_size

        self.state_memory = np.zeros((self.memory_size, n_states), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, n_states), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size)
        self.done_memory = np.zeros(self.memory_size, dtype=np.bool)

        self.count = 0

    def store_environment_transition(self, state, action, reward, state_, done):
        i = self.count % self.memory_size

        self.state_memory[i] = state
        self.next_state_memory[i] = state_
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.done_memory[i] = done

        self.count += 1

    def sample_from_buffer(self, batch_size=64):
        batch_elements = min(self.count, self.memory_size)
        batch_indexes = np.random.choice(batch_elements, batch_size)

        states = self.state_memory[batch_indexes]
        actions = self.action_memory[batch_indexes]
        rewards = self.reward_memory[batch_indexes]
        next_states = self.next_state_memory[batch_indexes]
        dones = self.done_memory[batch_indexes]

        return states, actions, rewards, next_states, dones