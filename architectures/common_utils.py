import os
import cv2
import random
import string
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from datetime import datetime
from scipy.special import softmax
from itertools import zip_longest
from sklearn.manifold import TSNE
from collections.abc import Iterable
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def identity(x, dim=0):
    """
    Return input without any change.
    x: torch.Tensor
    :return: torch.Tensor
    """
    return x

import torch

def snip_trajectories(
    is_snip_trajectory: bool,
    snipping_window : int,
    train_observations: torch.Tensor,
    train_action: torch.Tensor,
    train_len: torch.Tensor,
    labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Snips trajectories to a fixed length of snipping_window, starting from a random index.

    Args:
        train_observations (torch.Tensor): Tensor of observations with shape
                                           (batch_size, max_seq_len, obs_dim).
        train_action (torch.Tensor): Tensor of actions with shape
                                     (batch_size, max_seq_len, ...).
        labels (torch.Tensor): Tensor of labels with shape
                                (batch_size, max_seq_len).
        train_len (torch.Tensor): 1D Tensor of original trajectory lengths
                                  with shape (batch_size,).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A tuple containing the snipped observations, actions, labels, and the
        new trajectory lengths.
    """
    if not is_snip_trajectory:
        # If snipping is not required, return the original tensors.
        return [train_observations, train_action, train_len, torch.sum(labels, dim=-1)]
    # Ensure all operations happen on the same device as the input tensors
    device = train_observations.device
    batch_size = train_observations.shape[0]

    # 1. Determine the random start index for each trajectory in the batch
    # The upper bound for randint is clipped to ensure we can always snip snipping_window steps.
    # If a trajectory is shorter than snipping_window steps, we must start at index 0.
    # The `min=1` ensures that the argument to randint is at least 1.
    max_start_index = torch.clamp(train_len - snipping_window, min=1).float()
    
    # Generate random floats in [0, 1) and scale them to get integer start indices
    lower_bound = (torch.rand(batch_size, device=device) * max_start_index).long()

    # 2. Create the indices for the snipping_window-step snippets for the whole batch
    # `lower_bound.unsqueeze(1)` has shape (batch_size, 1)
    # `torch.arange(snipping_window)` has shape (snipping_window,)
    # Broadcasting results in `indices` with shape (batch_size, snipping_window)
    indices = lower_bound.unsqueeze(1) + torch.arange(snipping_window, device=device)

    # 3. Gather the snippets from actions and labels
    # Assumes train_action and labels are 2D: (batch_size, max_seq_len)
    # If train_action has more dimensions, its gather logic should match observations
    if train_action.dim() > 2:
        action_indices = indices.unsqueeze(-1).expand(*indices.shape, train_action.shape[-1])
        train_action = torch.gather(train_action, 1, action_indices)
    else:
        train_action = torch.gather(train_action, 1, indices)
    
    labels = torch.gather(labels, 1, indices)

    # 4. Gather the snippets from observations
    # The indices tensor needs an extra dimension to match the rank of train_observations
    obs_indices = indices.unsqueeze(-1).expand(-1, -1, train_observations.shape[-1])
    train_observations = torch.gather(train_observations, 1, obs_indices)
    
    # 5. Update the trajectory lengths
    train_len = torch.clamp(train_len, max=snipping_window)
    
    return [train_observations, train_action, train_len, torch.sum(labels, dim=-1)]

def custom_action_encoding(action_tensor: torch.Tensor, num_actions: int, dim: int):
    """
    Encodes a tensor of actions into a high-dimensional one-hot representation over segments.

    This version uses a robust broadcasting method to avoid potential indexing bugs
    in some PyTorch environments.
    """
    if dim % num_actions != 0:
        raise ValueError(f"Encoding dimension '{dim}' must be divisible by the number of actions '{num_actions}'.")

    input_shape = action_tensor.shape
    output_shape = input_shape + (dim,)
    
    # Create the final output tensor filled with zeros.
    encoded_tensor = torch.zeros(
        output_shape,
        device=action_tensor.device,
        dtype=torch.float32
    )
    
    slice_dim = dim // num_actions

    # Iterate through each possible action value.
    for i in range(num_actions):
        # Create a boolean mask with the same shape as the input tensor.
        mask = (action_tensor == i)
        
        # 1. Create a one-hot vector for the slice itself.
        slice_encoding = torch.zeros(dim, device=action_tensor.device, dtype=torch.float32)
        start_idx = i * slice_dim
        end_idx = (i + 1) * slice_dim
        slice_encoding[start_idx:end_idx] = 1.0

        # 2. Reshape the mask and the slice for broadcasting.
        #    mask shape: (300, 50) -> (300, 50, 1)
        #    slice_encoding shape: (25) -> (1, 1, 25)
        # This allows their values to be multiplied together correctly.
        update_values = mask.unsqueeze(-1) * slice_encoding.view(1, 1, -1)
        
        # 3. Add the values to the main tensor. Since the tensor is zeros, this is equivalent to setting them.
        encoded_tensor += update_values
        
    return encoded_tensor

def Boltzmann(q_values, T):
    """
    Computes the Boltzmann distribution over a set of Q-values.

    Args:
        q_values (np.ndarray): A numpy array of Q-values.
        T (float): The temperature parameter.

    Returns:
        np.ndarray: A probability distribution over the actions.
    """
    # Ensure T is not zero to avoid division by zero
    if T == 0:
        raise ValueError("Temperature can't be zero")
    
    # Subtracting the max Q-value for numerical stability to prevent overflow
    q_values = q_values - np.max(q_values)
    
    # Calculate the exponential of the Q-values divided by the temperature
    exp_values = np.exp(q_values / T)
    
    # Normalize to get the probability distribution
    return exp_values / np.sum(exp_values)
    
def get_activation(activation):
    # initialize activation
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'identity':
        return identity
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'none':
        return None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
 
class LSTMDataset(Dataset):
    def __init__(self, data):
        self.states = torch.tensor(data[0], dtype=torch.float32)
        self.actions = torch.tensor(data[1], dtype=torch.float32)
        self.lengths = torch.tensor(data[2], dtype=torch.int32)
        self.labels = torch.tensor(data[3], dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        lengths = self.lengths[idx]
        labels = self.labels[idx]
        return states, actions, lengths, labels
    
def collect_random(env, dataset, num_samples=200):
    episode = 0
    state, info = env.reset()
    # state = np.transpose(state['image'], (2, 0, 1))/255
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, truncated, terminated, _ = env.step(action)
        # next_state = np.transpose(next_state['image'], (2, 0, 1))/255
        done = truncated + terminated
        dataset.add((state, action, reward, next_state, truncated, terminated))
        state = next_state
        if done:
            episode + 1
            state, info = env.reset()
    return episode
            
def create_dump_directory(path):
    str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    dump_dir = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_{}'.format(str))
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir

def write_video(frames, episode, dump_dir, frameSize=(224, 224)):
    os.makedirs(dump_dir, exist_ok=True)
    video_path = os.path.join(dump_dir, f'{episode}.mp4')
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, frameSize, isColor=True)
    for img in frames:
        video.write(img)
    video.release()
    
def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo
    
def save_gif(frames, episode, dump_dir, fps):
    os.makedirs(dump_dir, exist_ok=True)
    gif_path = os.path.join(dump_dir, f'{episode}.gif')
    
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

def soft_update(local, target, tau):
    """
    Soft-update: target = tau*local + (1-tau)*target.
    local: nn.Module
    target: nn.Module
    tau: float
    """
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def hard_update(local, target):
    """
    Hard update: target <- local.
    local: nn.Module
    target: nn.Module
    """
    target.load_state_dict(local.state_dict())


def set_random_seed(seed, env):
    """
    Set random seed
    seed: int
    env: gym.Env
    """
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_one_hot(labels, c):
    """
    Converts an integer label to a one-hot Variable.
    labels (torch.Tensor): list of labels to be converted to one-hot variable
    c (int): number of possible labels
    """
    y = torch.eye(c).to(device)
    labels = labels.type(torch.LongTensor)
    return y[labels]


def one_hot_to_discrete_action(action, is_softmax=False):
    """
    convert the discrete action representation to one-hot representation
    action: in the format of a vector [one-hot-selection]
    """
    flatten_action = action.flatten()
    if not is_softmax:
        return np.argmax(flatten_action)
    else:
        return np.random.choice(flatten_action.shape[0], size=1, p=softmax(flatten_action)).item()


def discrete_action_to_one_hot(action_id, action_dim):
    """
    return one-hot representation of the action in the format of np.ndarray
    """
    action = np.array([0 for _ in range(action_dim)]).astype(np.float)
    action[action_id] = 1.0
    # in the format of one-hot-vector
    return action

