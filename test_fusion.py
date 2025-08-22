import torch
import wandb
import hydra
import warnings
import numpy as np

from PIL import Image
from collections import deque
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf
from Labeller.oracle import HumanOracleHighway
from agents.agent_utils.trajectory_buffer import TrajectoryReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_agent(env, cfg):
    cfg.agent.Network.action_dim = int(env.action_space.n)
    cfg.agent.Network.input_dim = int(env.observation_space.shape[0])
    return hydra.utils.instantiate(cfg)

def make_model(env, cfg):
    cfg.LSTM.Network.n_actions = int(env.action_space.n)
    cfg.LSTM.Network.feature_size = int(env.observation_space.shape[0])
    return hydra.utils.instantiate(cfg)

@hydra.main(version_base=None, config_path="configs", config_name="train_agent")
def train(args: DictConfig) -> None:
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    
    if args.env.name ==  "Highway":
        from env.highway import Highway
        env = Highway(args.env)
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    print("[INFO] Agent name: ", args.agent.Network.name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    #Make the agent and model
    agent = make_agent(env, args)
    agent = agent.agent.to(device)
    
    model = make_model(env, args)
    model = model.LSTM.to(device)
    
    