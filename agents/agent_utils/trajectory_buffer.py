import os
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
        self.actions_buffer = np.zeros(shape=(size, self.max_time), dtype=np.float32)
        self.label_buffer = np.zeros(shape=(size, self.max_time), dtype=np.float32)
        self.lens_buffer = np.zeros(shape=(size,), dtype=np.int32)
        self.next_ind = 0
        self.num_trajectories = 0

    # LSTM training does only make sense, if there are sequences in the buffer which have different returns.
    # LSTM could otherwise learn to ignore the input and just use the bias units.
    def different_returns_encountered(self):
        if self.num_trajectories == 0:
            return False
        labels = self.labels_buffer[:min(self.num_trajectories, self.size)]
        return len(np.unique(labels)) > 1

    # Add a new episode to the buffer
    def add(self, trajectory):
        previous_state, actions, label = trajectory
        traj_length = len(previous_state)
        
        idx = self.next_ind
        self.states_buffer[idx, :traj_length] = previous_state
        self.actions_buffer[idx, :traj_length] = actions
        self.label_buffer[idx, :traj_length] = label
        self.lens_buffer[idx] = traj_length
        
        # advance index
        self.next_ind = (self.next_ind + 1) % self.size
        self.num_trajectories = min(self.num_trajectories + 1, self.size)
        
    def sample(self, batch_size):
        indices = np.random.choice(range(min(self.num_trajectories, self.size)), size=batch_size)

        states = torch.from_numpy(self.states_buffer[indices]).to(self.device)
        actions = torch.from_numpy(self.actions_buffer[indices]).to(self.device)
        rewards = torch.from_numpy(self.label_buffer[indices]).to(self.device)
        lengths = torch.from_numpy(self.lens_buffer[indices]).to(self.device)
        indices = torch.from_numpy(indices).to(self.device)

        return states, actions, rewards, lengths, indices
    
    def save_buffer_data(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        N = self.num_trajectories   # how many trajectories are filled

        np.save(os.path.join(save_dir, "states_buffer.npy"), self.states_buffer[:N], allow_pickle=True)
        np.save(os.path.join(save_dir, "actions_buffer.npy"), self.actions_buffer[:N], allow_pickle=True)
        np.save(os.path.join(save_dir, "lens_buffer.npy"), self.lens_buffer[:N], allow_pickle=True)
        np.save(os.path.join(save_dir, "labels_buffer.npy"), self.label_buffer[:N], allow_pickle=True)

        print(f"[INFO] Trajectory buffer with {N} trajectories saved to {save_dir}")
        
    def __len__(self):
        return min(self.num_trajectories, self.size)

