import torch
import numpy as np
        
class TrajectoryReplayBuffer:
    def __init__(self, feature_size, max_time, size, device):
        self.size = size
        self.feature_size = feature_size
        self.max_time = max_time
        # Initializing the list to store the transitions
        self.reset_buffer(self.size)
        self.device = device

    def reset_buffer(self, size):
        self.states_buffer = np.zeros(shape=(size, self.max_time, self.feature_size), dtype=np.float32)
        self.next_states_buffer = np.zeros(shape=(size, self.max_time, self.feature_size), dtype=np.float32)
        self.actions_buffer = np.zeros(shape=(size, self.max_time), dtype=np.float32)
        self.rewards_buffer = np.zeros(shape=(size, self.max_time), dtype=np.float32)
        self.lens_buffer = np.zeros(shape=(size, 1), dtype=np.int32)
        self.next_spot_to_add = 0
        self.next_ind = 0
        self.buffer_is_full = False
        self.num_trajectories = 0

    # LSTM training does only make sense, if there are sequences in the buffer which have different returns.
    # LSTM could otherwise learn to ignore the input and just use the bias units.
    def different_returns_encountered(self):
        if self.buffer_is_full:
            return len(np.unique(np.sum(self.rewards_buffer, axis=-1))) > 1
        else:
            return len(np.unique(np.sum(self.rewards_buffer[:self.next_spot_to_add], axis=-1))) > 1

    # Add a new episode to the buffer
    def add(self, trajectory):
        previous_state, actions, rewards, next_state = trajectory
        traj_length = len(previous_state)
        self.next_ind = self.next_spot_to_add
        self.num_trajectories += 1
        self.next_spot_to_add = self.next_spot_to_add + 1
        self.next_spot_to_add = self.next_spot_to_add % self.size
        if self.num_trajectories >= self.size:
            self.buffer_is_full = True
        self.states_buffer[self.next_ind, :traj_length] = previous_state
        self.states_buffer[self.next_ind, traj_length:] = 0
        self.next_states_buffer[self.next_ind, :traj_length] = next_state
        self.next_states_buffer[self.next_ind, traj_length:] = 0
        self.actions_buffer[self.next_ind, :traj_length] = actions
        self.actions_buffer[self.next_ind, traj_length:] = 0
        self.rewards_buffer[self.next_ind, :traj_length] = rewards
        self.rewards_buffer[self.next_ind, traj_length:] = 0
        self.lens_buffer[self.next_ind] = traj_length
        
    def sample(self, batch_size):
        indices = np.random.choice(range(min(self.num_trajectories, self.size)), size=batch_size)

        states = torch.from_numpy(self.states_buffer[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states_buffer[indices]).to(self.device)
        actions = torch.from_numpy(self.actions_buffer[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards_buffer[indices]).to(self.device)
        lengths = torch.from_numpy(self.lens_buffer[indices]).to(self.device)
        indices = torch.from_numpy(indices).to(self.device)

        return states, next_states, actions, rewards, lengths, indices
        
    def __len__(self):
        return min(self.num_trajectories, self.size)


def nograd(t):
    return t.detach()
