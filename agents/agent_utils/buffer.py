import os
import pickle
import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "truncated", "terminated"])
    
    def add(self, transition):
        """Add a new experience to memory."""
        state, action, reward, next_state, truncated, terminated = transition
        e = self.experience(state, action, reward, next_state, truncated, terminated)
        self.memory.append(e)

    def dump_data(self, dir):
        states = [exp.next_state for exp in self.memory]
        data_path = os.path.join(dir, 'data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(states, f)
        print(f"Collected {len(states)} transitions and saved to {data_path}")
        data_path = os.path.join(dir, 'class_prob.pkl')

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        if isinstance(experiences[0].state, np.ndarray):
            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.stack([e.state for e in experiences if e is not None]).float().squeeze().to(self.device)
            next_states = torch.stack([e.next_state for e in experiences if e is not None]).squeeze().float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        truncated = torch.from_numpy(np.vstack([e.truncated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        terminated = torch.from_numpy(np.vstack([e.terminated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, truncated, terminated)
    
    def dump_data(self, dir):
        states = [exp.next_state for exp in self.memory]
        class_probs = [exp.class_prob for exp in self.memory]
        data_path = os.path.join(dir, 'data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(states, f)
        print(f"Collected {len(states)} transitions and saved to {data_path}")
        data_path = os.path.join(dir, 'class_prob.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(class_probs, f)
        print(f"Collected {len(class_probs)} transitions and saved to {data_path}")

    def get_full_observation_reward(self):
        states = torch.from_numpy(np.stack([e.next_state for e in self.memory if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in self.memory if e is not None])).float().to(self.device)
        return (states, rewards)
    
    def clear(self):
        """Clear all experiences from memory."""
        self.memory.clear()
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class TrajectoryReplyBuffer:
    """Buffer to store the trajectory data"""
    def __init__(self, max_episode_len, feature_dim, buffer_size, encoded_dim):
        self.max_episode_len = max_episode_len
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        self.encoded_dim = encoded_dim
        self.next_spot_to_add = 0
        self.buffer_is_full = False
        
        self.states_buffer = np.zeros(shape=(self.buffer_size, self.max_episode_len+1, self.feature_dim), dtype=np.float32)
        self.caption_buffer = np.zeros(shape=(self.buffer_size, self.max_episode_len+1, self.encoded_dim), dtype=np.float32)
        self.ep_len = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)
        
    def dump_buffer_data(self, dump_dir):
        
        data_path = os.path.join(dump_dir, 'states.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self.states_buffer, f)
        print(f"Collected {len(self.states_buffer)} states and saved to {data_path}")
        
        data_path = os.path.join(dump_dir, 'caption_encode.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self.caption_buffer, f)
        print(f"Collected {len(self.caption_buffer)} caption encoding and saved to {data_path}")
        
        data_path = os.path.join(dump_dir, 'episode_len.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self.ep_len, f)
        print(f"Collected {len(self.ep_len)} episode length and saved to {data_path}")
            
    def add(self, state, caption, language):
        traj_length = len(state)
        self.next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + 1
        if self.next_spot_to_add >= self.buffer_size:
            self.buffer_is_full = True
        self.states_buffer[self.next_ind, :traj_length] = state
        self.caption_buffer[self.next_ind, :traj_length] = caption
        self.ep_len[self.next_ind] = traj_length
        
    def sample(self, batch_size):
        indices = np.random.choice(range(self.buffer_size), size = batch_size)
        return (self.states_buffer[indices, :, :], self.caption_buffer[indices, :, :], indices)
    
class StateCaptionEncodeBuffer():
    """Buffer to store the states and encoded captions"""
    def __init__(self, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.states = []
        self.next_states = []
        self.caption_encoded = []
            
    def add(self, state, next_state, caption_encode):
        """Add a new experience to memory."""
        self.states.append(state)
        self.next_states.append(next_state)
        self.caption_encoded.append(caption_encode)
        
    def dump_buffer_data(self, dump_dir):
        
        data_path = os.path.join(dump_dir, 'states.pkl')
        with open(data_path, "wb") as file:
            pickle.dump(self.states, file)
            
        data_path = os.path.join(dump_dir, 'next_states.pkl')
        with open(data_path, "wb") as file:
            pickle.dump(self.next_states, file)
            
        data_path = os.path.join(dump_dir, 'caption_encoded.pkl')
        with open(data_path, "wb") as file:
            pickle.dump(self.caption_encoded, file)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)