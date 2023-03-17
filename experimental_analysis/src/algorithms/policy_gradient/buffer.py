import numpy as np
import scipy.signal

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, max_size, action_dimension, gamma=0.99, lam=0.97):
        # Buffer initialization
        if isinstance(observation_dimensions, list):
            self.observation_list = True
            self.observation_buffer = []
            for l in observation_dimensions:
                self.observation_buffer.append(np.zeros((max_size, l), dtype=np.float32))
        else:
            self.observation_list = False
            self.observation_buffer = np.zeros(
                (max_size, observation_dimensions), dtype=np.float32
            )
        self.action_buffer = np.zeros(max_size, dtype=np.int32)
        self.advantage_buffer = np.zeros(max_size, dtype=np.float32)
        self.reward_buffer = np.zeros(max_size, dtype=np.float32)
        self.return_buffer = np.zeros(max_size, dtype=np.float32)
        self.value_buffer = np.zeros(max_size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(max_size, dtype=np.float32)
        self.mask_buffer = np.zeros((max_size, action_dimension), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value=None, logprobability=None, mask=None):
        # Append one step of agent-environment interaction
        if self.observation_list:
            for i, o in enumerate(observation):
                self.observation_buffer[i][self.pointer] = o
        else:
            self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward

        if value is not None:
            self.value_buffer[self.pointer] = value
        if logprobability is not None:
            self.logprobability_buffer[self.pointer] = logprobability
        if mask is not None:
            self.mask_buffer[self.pointer] = mask
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        elems_in_buffer = self.pointer
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
            elems_in_buffer,
            self.mask_buffer,
        )
