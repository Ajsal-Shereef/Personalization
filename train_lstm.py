import torch
import wandb
import hydra
import pickle
import warnings
import numpy as np

from PIL import Image
from tqdm import tqdm
from collections import deque
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_model(env, cfg):
    cfg.Network.n_actions = int(env.action_space.n)
    cfg.Network.feature_size = int(env.observation_space.shape[0])
    return hydra.utils.instantiate(cfg)

@hydra.main(version_base=None, config_path="configs", config_name="train_lstm")
def train(args: DictConfig) -> None:
    
    if args.env.name ==  "Highway":
        from env.highway import Highway
        env = Highway(args.env)
    elif args.env.name ==  "Pong":
        from env.pong import Pong
        env = Pong(args.env)
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    if args.use_wandb:
        wandb.init(project="Project 1", name=f"{args.LSTM.Network.name}_{args.env.name}", config=OmegaConf.to_container(args, resolve=True))
    print("[INFO] Model name: ", args.LSTM.Network.name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    #Make the model
    model = make_model(env, args.LSTM)
    model = model.to(device)
    
    #Loading the datasets to the memory
    with open(f"{args.datapath}/{args.env.name}/{args.mode}/states_buffer.npy", "rb") as f:
        states = np.load(f)
    with open(f"{args.datapath}/{args.env.name}/{args.mode}/actions_buffer.npy", "rb") as f:
        actions = np.load(f)
    with open(f"{args.datapath}/{args.env.name}/{args.mode}/lens_buffer.npy", "rb") as f:
        lengths = np.load(f)
    with open(f"{args.datapath}/{args.env.name}/{args.mode}/labels_buffer.npy", "rb") as f:
        labels = np.load(f)
        
    #Creating the dataset and dataloader
    dataset = LSTMDataset([states, actions, lengths, labels])
    dataloader = DataLoader(dataset, batch_size=args.LSTM.Network.batch_size, shuffle=True)
    
    #Creating the model save dir
    model_dir = create_dump_directory(f"model_weights/{args.LSTM.Network.name}")
    print("[INFO] Dump dir: ", model_dir)
    
    #Dumping the training config files
    config_path = os.path.join(model_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)
    
    epoch_bar = tqdm(range(1, args.epochs+1), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        for data in dataloader:
            data = snip_trajectories(args.env.snip_trajectory, args.env.snip_trajectory_window, *data)
            metric = model.learn(data)
            
        if args.use_wandb:
            wandb.log(metric, step = epoch)
        if epoch % args.save_every == 0:
            model.save(f"{model_dir}/", save_name=f"{args.LSTM.Network.name}")
            
    model.save(f"{model_dir}/", save_name=f"{args.LSTM.Network.name}")
            
    
if __name__ == "__main__":
    train()