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
    
def save_gif(frames, episode, dump_dir, duration=100):
    os.makedirs(dump_dir, exist_ok=True)
    gif_path = os.path.join(dump_dir, f'{episode}.gif')
    
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
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

