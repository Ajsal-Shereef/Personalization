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