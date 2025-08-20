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
    
class VGGLoss(nn.Module):
    """
    Custom loss function for VGG-based perceptual loss.
    """
    def __init__(self, device=device):
        """
        Initializes the loss function.
        Args:
            device (str): The device to run the VGG model on ('cpu' or 'cuda').
        """
        super(VGGLoss, self).__init__()
        
        # Load pre-trained VGG16 model and set to evaluation mode
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        
        # We don't need to train the VGG model, so we freeze its parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Use an intermediate layer for perceptual loss. relu3_3 (layer 15) is a common choice.
        self.vgg_feature_extractor = nn.Sequential(*list(vgg.children())[:16])

        # VGG was trained on ImageNet, which has specific normalization values.
        # We store these as non-trainable parameters.
        self.normalize = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False).to(device)
        self.normalize_std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False).to(device)


    def forward(self, y_pred, y_true):
        """
        Calculates the VGG perceptual loss.
        Args:
            y_pred (torch.Tensor): The predicted images from the model (range [0,1]).
            y_true (torch.Tensor): The ground truth images (range [0,1]).
        Returns:
            torch.Tensor: The VGG perceptual loss.
        """
        # Normalize images before passing them to VGG
        y_pred_norm = (y_pred - self.normalize) / self.normalize_std
        y_true_norm = (y_true - self.normalize) / self.normalize_std

        # Extract features from the intermediate VGG layer
        pred_features = self.vgg_feature_extractor(y_pred_norm)
        true_features = self.vgg_feature_extractor(y_true_norm)

        # Calculate perceptual loss as the mean squared error between the feature maps
        perceptual_loss = F.l1_loss(pred_features, true_features)
        
        return perceptual_loss

def normalize_01(x, dim=1):
    """
    Normalize the elements in x to [0, 1]
    x: torch.Tensor, the shape should be (batch size, flatten vector)
    :return: torch.Tensor
    """

    min_x = torch.min(x, dim=1, keepdim=True)[0]
    max_x = torch.max(x, dim=1, keepdim=True)[0]

    delta = max_x - min_x
    zero_idxs = (delta[:, 0] == 0)
    delta[zero_idxs, :] = 1.0

    x = x - min_x
    x = x/delta
    x[zero_idxs, :] = 0

    return x

def random_mask(image, min_size=8, max_size=16):
    """
    Applies a random rectangular mask to a single image.
    """
    c, h, w = image.shape
    mask_h = np.random.randint(min_size, max_size)
    mask_w = np.random.randint(min_size, max_size)

    top = np.random.randint(0, h - mask_h)
    left = np.random.randint(0, w - mask_w)

    masked = image.clone()
    masked[:, top:top+mask_h, left:left+mask_w] = 0.0

    return masked

def sample_gumbel_softmax(c_logit, training=True, tau=0.1):
    if training:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(c_logit) + 1e-10) + 1e-10)
        y = F.softmax((c_logit + gumbel_noise) / tau, dim=-1)
        return y
    else:
        return F.softmax((c_logit)/tau, dim=-1)
    
def sample_binary_concrete(logits, training=True, tau=0.1, hard=False, eps=1e-6):
    """
    Sample relaxed Bernoulli (Binary Concrete) during training;
    return deterministic output during evaluation.

    Args:
        logits (Tensor): Logits (pre-sigmoid), shape [batch_size, ...].
        tau (float): Temperature of relaxation (lower → sharper).
        training (bool): If True, sample; else deterministic.
        hard (bool): If True and not training, return binary 0/1 instead of sigmoid output.
        eps (float): Small constant to avoid numerical issues.

    Returns:
        Tensor: Sampled or deterministic tensor in [0, 1] or {0,1}.
    """
    if training:
        # Sample from logistic distribution. This is the Binary Concrete distribution.
        # g1-g2 follows logistic distribution with CDF: F(x) = 1 / (1 + exp(-x))
        # We use the reparameterization trick to sample from this distribution.
        u = torch.rand_like(logits).clamp(min=eps, max=1 - eps)
        logistic_noise = torch.log(u) - torch.log(1 - u)
        return torch.sigmoid((logits + logistic_noise) / tau)
    else:
        # Evaluation mode
        if hard:
            return (logits > 0).float()  # Hard threshold
        else:
            return torch.sigmoid(logits)  # Soft sigmoid output
        
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

class Visualiser:
    
    def __init__(self, vision_model, dae_model, save_dir):
        self.vision_model = vision_model
        self.dae_model = dae_model
        self.save_dir = save_dir
        self.vision_model.eval()
        self.dae_model.eval()
        
    def plot_img(self, ax, img_tensor, title=""):
        img_np = img_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    def latent_traversal(self, x, img_number, mean_of_mus, std_of_mus, n_traversal=30, n_std_dev=10):
        """
        Performs traversal using data-driven ranges based on pre-calculated statistics.
        Visualizes ONLY the final output from the DAE.

        Args:
            x (torch.Tensor): The input image tensor [C, H, W].
            img_number (int): An identifier for the saved image file.
            mean_of_mus (torch.Tensor): The mean of each latent dimension from the diagnostic tool.
            std_of_mus (torch.Tensor): The standard deviation of each latent dimension.
            n_traversal (int): The number of steps to generate.
            n_std_dev (float): How many standard deviations to traverse from the mean.
        """
        # Ensure models are in evaluation mode
        self.vision_model.eval()
        self.dae_model.eval()
        device = next(self.vision_model.parameters()).device
        x = x.to(device).unsqueeze(0)

        with torch.no_grad():
            latent_list = self.vision_model.get_latent_list(x)

        # Set up the plot grid
        sampled_dims_list = [min(32, l.shape[1]) for l in latent_list]
        total_sampled_dims = sum(sampled_dims_list)
        if total_sampled_dims == 0: return

        n_rows, n_cols = total_sampled_dims, n_traversal + 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        if n_rows == 1: axs = np.expand_dims(axs, 0)
        if n_cols == 1: axs = np.expand_dims(axs, 1)
        current_row = 0
        for idx, base_latent in enumerate(latent_list):
            latent_dim = base_latent.shape[1]
            sampled_dims = range(min(32, latent_dim))

            for i, dim in enumerate(sampled_dims):
                plot_row = current_row + i

                # Plot original reconstruction
                with torch.no_grad():
                    original_x_hat = self.vision_model.decoder(torch.cat(latent_list, dim=1)) if len(latent_list) > 1 else self.vision_model.decoder(latent_list[0])
                    original_x_dae = self.dae_model(original_x_hat)
                self.plot_img(axs[plot_row, 0], original_x_dae, "Original")
                axs[plot_row, 0].set_ylabel(f"DAE z[{idx}][{dim}]", fontsize=9, rotation=0, labelpad=35, va='center')

                # --- DATA-DRIVEN TRAVERSAL LOGIC ---
                # Get the pre-computed stats for this specific dimension
                mean_val = mean_of_mus[dim].item()
                std_val = std_of_mus[dim].item()

                # Create a range of +/- N standard deviations around the mean for this dimension
                # start_val = mean_val - (n_std_dev * std_val)
                # end_val = mean_val + (n_std_dev * std_val)
                
                start_val = base_latent.squeeze()[dim].item() - (n_std_dev * std_val)
                end_val = base_latent.squeeze()[dim].item() + (n_std_dev * std_val)
                
                values = torch.linspace(start_val, end_val, n_traversal, device=device)

                # --- BATCHED TRAVERSAL (Same as before) ---
                traversal_latent = base_latent.clone().repeat(n_traversal, 1)
                traversal_latent[:, dim] = values

                modified_latents_batch = []
                for l_idx, l_base in enumerate(latent_list):
                    modified_latents_batch.append(traversal_latent if l_idx == idx else l_base.clone().repeat(n_traversal, 1))

                # Decode the batch
                with torch.no_grad():
                    if len(modified_latents_batch) == 2:
                        x_hat_batch = self.vision_model.decoder(modified_latents_batch[0], modified_latents_batch[1])
                    else:
                        x_hat_batch = self.vision_model.decoder(torch.cat(modified_latents_batch, dim=1))
                    x_dae_batch = self.dae_model(x_hat_batch)

                # Plot the results
                for j in range(n_traversal):
                    self.plot_img(axs[plot_row, j + 1], x_dae_batch[j])

            current_row += len(sampled_dims)

        # Save the final figure
        plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.1)
        save_folder = f"{self.save_dir}/{self.vision_model.name}"
        os.makedirs(save_folder, exist_ok=True)
        save_path = f"{save_folder}/traversal_{img_number}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=250)
        plt.close(fig)
        print(f"[INFO] Data-driven traversal grid saved to: {save_path}")

    def diagnose_latent_space(self, dataloader, num_batches=20):
        """
        Analyzes the VAE's latent space to check for inactive (dead) dimensions.

        Args:
            dataloader: A DataLoader for your dataset.
            num_batches (int): The number of batches to average over.
        """
        self.vision_model.eval()
        device = next(self.vision_model.parameters()).device

        all_mus = []
        print("Running latent space diagnosis...")

        with torch.no_grad():
            for i, images in enumerate(dataloader):
                if i >= num_batches:
                    break
                images = images.to(device)

                # This assumes get_latent_list returns the raw output of the encoder
                # which might be [mu, logvar] or just a single latent z.
                # We need to get the `mu` vector. Let's assume you have an `encode` method.
                # If not, adapt this line to get the mean vector.
                mu = self.vision_model.get_latent_list(images) # Shape: [batch_size, latent_dim]

                # If your model returns a list of latents, focus on the first for diagnosis
                if isinstance(mu, list):
                    mu = mu[0]

                all_mus.append(mu)

        if not all_mus:
            print("Diagnosis failed: Could not retrieve latent vectors.")
            return

        all_mus_cat = torch.cat(all_mus, dim=0) # Shape: [total_images, latent_dim]

        # Calculate the std of each latent dimension across the dataset
        mean_of_mus = all_mus_cat.mean(dim=0)
        std_of_mus = all_mus_cat.std(dim=0)

        print(f"\n--- Latent Space Health Report (based on {all_mus_cat.shape[0]} samples) ---")
        print(f"Number of latent dimensions: {std_of_mus.shape[0]}")

        # Identify dead dimensions (std close to zero)
        dead_dims = (std_of_mus < 1e-4).sum().item()

        print(f"\nMean of each dimension:\n{mean_of_mus.cpu().numpy().round(3)}")
        print(f"\nstandard deviation of each dimension:\n{std_of_mus.cpu().numpy().round(3)}")

        print(f"\nResult: Found {dead_dims} 'dead' or inactive dimensions (std < 0.0001).")

        if dead_dims == std_of_mus.shape[0]:
            print("\n[CRITICAL] All latent dimensions are dead. This is a classic case of posterior collapse.")
            print("The VAE is ignoring the latent code. This is why traversal has no effect.")
        elif dead_dims > 0:
            print(f"\n[WARNING] {dead_dims} dimensions are inactive. Your traversal might be sampling only these.")
        else:
            print("\n[GOOD] Latent space seems active. All dimensions have significant std.")
        return mean_of_mus, std_of_mus

    def save_image(self, grid, filename, resize=False):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        if resize:
            im = im.resize((im.size[0] // 3, im.size[1] // 3), Image.Resampling.LANCZOS)
        im.save(filename)
        print("[INFO] Visualisation saved at: ", filename)

    def visualize_latents(self, data, labels=None, title="Latent Space", threshold=0.5):
        z_mu = [self.vision_model.get_latent_list(i.unsqueeze(0).to(device).float())[0].detach().cpu().numpy() for i in data]
        z_vals = np.stack(z_mu).squeeze()
        tsne = TSNE(n_components=2)
        z_2d = tsne.fit_transform(z_vals)

        plt.figure(figsize=(7, 6))

        if labels is not None:
            labels = np.array(labels)
            # Binarize sigmoid outputs using threshold
            if labels.ndim == 2:
                labels_bin = (labels >= threshold).astype(int)
                label_dims = labels.shape[1]
                for i in range(label_dims):
                    idx = labels_bin[:, i] == 1
                    if np.sum(idx) > 0:
                        plt.scatter(z_2d[idx, 0], z_2d[idx, 1], s=10, label=f"Label {i}", alpha=0.7)
            elif labels.ndim == 1:
                labels_bin = (labels >= threshold).astype(int)
                for i in np.unique(labels_bin):
                    idx = labels_bin == i
                    plt.scatter(z_2d[idx, 0], z_2d[idx, 1], s=10, label=f"Class {i}", alpha=0.7)

            plt.legend()
        else:
            plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.6)

        plt.title(title)
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(f"{self.save_dir}/{self.vision_model.name}", exist_ok=True)
        plt.savefig(f"{self.save_dir}/{self.vision_model.name}/{title}.png")
        print("[INFO] Latent space visualization saved at: ", f"{self.save_dir}/{self.vision_model.name}/{title}.png")

    def swap(self, images, device):
        if self.vision_model.name != "MAGIK":
            return
        count = len(images)
        image_shape = images.size()[1:]
        z_mu, c = self.vision_model.get_latent_list(images.to(device).float())
        z = z_mu
        u = c
        z_dim = z_mu[0].shape[0]
        latents = list()
        for j in range(count):
            for i in range(count):
                latents.append(torch.cat([z[j], u[i]]).unsqueeze(0))
        # Convert latents to a tensor
        latents_tensor = torch.stack(latents).squeeze(1)
        z_tensor = latents_tensor[:, :z_dim]  
        u_tensor = latents_tensor[:, z_dim:]  
        all_images = torch.zeros(count+1, count+1, *image_shape)
        all_images[0, 1:] = images
        all_images[1:, 0] = images
        # all_images[torch.arange(count+1, (count+1)*(count+1), count+1)] = images
        all_images[1:, 1:] = self.dae_model(self.vision_model.decoder(z_tensor, u_tensor)).view(count, count, *image_shape)
        grid = make_grid(all_images.view(-1, *image_shape), nrow=count+1, pad_value=0)
        self.save_image(grid, f"{self.save_dir}/{self.vision_model.name}" + '/swap.png', resize=False)
        
    def interpolate_images(self, dataloader, num_pairs=20, n_steps=30, save_name="interpolation.png"):
        """
        Samples image pairs from a dataloader, performs latent space interpolation for each,
        and consolidates all results into a single output image grid.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
            num_pairs (int): The number of random pairs to interpolate (becomes rows in the grid).
            n_steps (int): The number of steps for each interpolation sequence.
            save_name (str): The filename for the final consolidated image.
        """
        # --- 1. Initial Setup ---
        device = next(self.vision_model.parameters()).device

        # --- 2. Collect all images from the dataloader for easy sampling ---
        print("[INFO] Collecting all images from the dataloader...")
        try:
            all_images = [item[0] for item in dataloader]
        except IndexError:
            all_images = [item for item in dataloader]

        if len(all_images) < 2:
            print("[ERROR] Dataloader contains fewer than 2 images. Cannot create pairs.")
            return

        print(f"[INFO] Collected {len(all_images)} images. Generating consolidated grid for {num_pairs} pairs.")

        # --- 3. Create a single, large figure grid OUTSIDE the loop ---
        n_rows = num_pairs
        n_cols = n_steps + 2  # +2 for the original images
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))

        # Ensure axs is always a 2D array for consistent indexing
        if n_rows == 1:
            axs = np.expand_dims(axs, 0)

        # --- 4. Loop to process each pair and populate the grid ---
        for i in range(num_pairs):
            print(f"--- Processing and plotting Pair {i+1}/{num_pairs} ---")

            # Get the axes for the current row
            current_row_axs = axs[i]

            # Randomly sample a pair of images
            idx_a, idx_b = torch.randperm(len(all_images))[:2]
            img_a_orig = all_images[idx_a]
            img_b_orig = all_images[idx_b]

            # --- Core interpolation logic (same as before) ---
            with torch.no_grad():
                img_a_tensor = img_a_orig.to(device).unsqueeze(0)
                img_b_tensor = img_b_orig.to(device).unsqueeze(0)

                z_a = self.vision_model.get_latent_list(img_a_tensor)[0]
                z_b = self.vision_model.get_latent_list(img_b_tensor)[0]

                alphas = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
                interpolated_latents = (1 - alphas) * z_a + alphas * z_b

                x_hat_batch = self.vision_model.decoder(interpolated_latents)
                x_dae_batch = self.dae_model(x_hat_batch)

            # --- 5. Plot the results on the pre-made axes for the current row ---

            # Add a Y-label to identify the row
            current_row_axs[0].set_ylabel(f"Pair {i+1}", rotation=0, size='large', labelpad=40, va='center')

            # Plot Original Image A
            self.plot_img(current_row_axs[0], img_a_orig, "Original A")

            # Plot Interpolated Sequence
            for j in range(n_steps):
                title = f"α={alphas[j].item():.2f}" if i == 0 else "" # Only show alpha on the top row
                self.plot_img(current_row_axs[j + 1], x_dae_batch[j], title)

            # Plot Original Image B
            self.plot_img(current_row_axs[n_cols - 1], img_b_orig, "Original B")

        # --- 6. Finalize and Save the single figure AFTER the loop ---
        plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.1)

        save_folder = f"{self.save_dir}/{self.vision_model.name}"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_name)

        print(f"\n[INFO] Saving consolidated grid to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=250)
        plt.close(fig) # Close the single, large figure
        print("[INFO] Interpolations done.")

class StatesDataset(Dataset):
    def __init__(self, data, labels=None, split_ratio=0.5):
        self.data = torch.tensor(data, dtype=torch.float32)

        if labels is None:
            # Unsupervised mode
            self.labels = None
            self.supervised = False
        else:
            # Supervised mode with split and pairing
            self.labels = torch.tensor(labels).float()
            assert len(self.data) == len(self.labels), "Data and labels must be the same length"
            assert 0.0 < split_ratio < 1.0, "Split ratio must be between 0 and 1"

            split_idx = int(len(self.data) * split_ratio)
            self.data1 = self.data[:split_idx]
            self.label1 = self.labels[:split_idx]
            self.data2 = self.data[split_idx:]
            self.label2 = self.labels[split_idx:]

            self.supervised = True
            self.dataset_len = max(len(self.data1), len(self.data2))

    def __len__(self):
        if self.supervised:
            return self.dataset_len
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.supervised:
            i1 = random.randint(0, len(self.data1) - 1)
            i2 = random.randint(0, len(self.data2) - 1)
            x1, y1 = self.data1[i1], self.label1[i1]
            x2, y2 = self.data2[i2], self.label2[i2]
            return x1, y1, x2, y2
        else:
            return self.data[idx]

class NormalDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx]
        return state, state

class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.H, self.W = data.shape[-2:] 
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean = self.data[idx]
        masked = random_mask(clean, 0, min(self.W, self.H))
        return masked, clean

class NoisyDataset(Dataset):
    def __init__(self, data, noise_factor=0.3):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean = self.data[idx]
        noisy = clean + self.noise_factor * torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0., 1.)
        return noisy, clean
    
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

