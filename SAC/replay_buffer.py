import numpy as np

class ReplayBuffer:
    def __init__(self, n_actions, n_states, max_size=1000000):
        # self.n_actions = n_actions
        # self.n_states = n_states
        self.max_memory_size = max_size

        self.state_memory = np.zeros((self.max_memory_size, n_states), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_memory_size, n_states), dtype=np.float32)
        self.action_memory = np.zeros((self.max_memory_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.max_memory_size)
        self.terminal_memory = np.zeros(self.max_memory_size, dtype=np.bool)

        self.n_count = 0

    def store_transition(self, state, action, reward, state_, done):
        idx = self.n_count % self.max_memory_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.n_count += 1

    def sample_buffer(self, batch_size=64):
        max_mem = min(self.n_count, self.max_memory_size)
        batch_idx = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch_idx]
        actions = self.action_memory[batch_idx]
        rewards = self.reward_memory[batch_idx]
        states_ = self.new_state_memory[batch_idx]
        dones = self.terminal_memory[batch_idx]

        return states, actions, rewards, states_, dones