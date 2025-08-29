import torch
import wandb
import hydra
import numpy as np

from PIL import Image
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_agent(env, cfg):
    cfg.Network.action_dim = int(env.action_space.n)
    cfg.Network.input_dim = int(env.observation_space.shape[0])
    cfg.Network.initial_random_samples = 0
    return hydra.utils.instantiate(cfg)

def make_model(env, cfg):
    cfg.Network.n_actions = int(env.action_space.n)
    cfg.Network.feature_size = int(env.observation_space.shape[0])
    return hydra.utils.instantiate(cfg)

@hydra.main(version_base=None, config_path="configs", config_name="test_fusion")
def test(args: DictConfig) -> None:
    """
    Main evaluation function.
    """
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # Environment Setup
    if args.env.name ==  "Highway":
        from env.highway import Highway
        env = Highway(args.env)
    elif args.env.name ==  "Pong":
        from env.pong import Pong
        env = Pong(args.env)
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    print("[INFO] Strategy: ", args.action_selection_strategy)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Agent and Model Initialization
    agent = make_agent(env, args.agent).to(device)
    model = make_model(env, args.LSTM).to(device)
    
    #Loading the pretrained weights
    #Loading the agent
    if args.action_selection_strategy in ["fusion", "dqn"]: 
        agent_cfg = args.agent
        agent_dir = agent_cfg.Test.checkpoint
        if os.path.exists(agent_dir + "/config.yaml"):
            agent_args =  OmegaConf.load(agent_dir + "/config.yaml")
            agent_cfg = agent_args.agent
            args.agent = agent_cfg
            args.agent.Test.checkpoint = agent_dir
        else:
            raise FileNotFoundError(f"Config file not found in {agent_dir}/config.yaml")
        agent.load_params(agent_dir)
        agent.eval()
        for param in agent.parameters():
            param.requires_grad = False
        
    #Loading the LSTM
    if args.action_selection_strategy in ["fusion", "lstm"]: 
        lstm_cfg = args.LSTM
        lstm_dir = lstm_cfg.Test.checkpoint
        if os.path.exists(lstm_dir + "/config.yaml"):
            lstm_args =  OmegaConf.load(lstm_dir + "/config.yaml")
            lstm_cfg = lstm_args.LSTM
            args.LSTM = lstm_cfg
            args.LSTM.Test.checkpoint = lstm_dir
        else:
            raise FileNotFoundError(f"Config file not found in {lstm_dir}/config.yaml")
        model.load_params(lstm_dir)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    #Making sure the DQN select action greedly by setting the epsilon to 0
    agent.epsilon = 0
    
    def select_action(state, T_psi):
        """Selects an action based on the chosen strategy."""
        state = torch.tensor(state).float().to(device).unsqueeze(dim=0)
        if args.action_selection_strategy == "dqn":
            return agent.get_action(state)
        elif args.action_selection_strategy == "lstm":
            legal_action = env.get_legal_actions()
            return model.get_action(state, legal_action)
        else:
            legal_action = env.get_legal_actions()
            # 1. Get Q-values from the task-specific DQN agent
            dqn_q_values = agent.critic(state).squeeze()
            task_specific_policy = Boltzmann(dqn_q_values.detach().cpu().numpy(), args.env.Fusion.T_phi).squeeze()
            # 2. Get Q-values from the intent-specific LSTM model
            lstm_q_values = []
            for action in range(env.action_space.n):
                action = torch.tensor(action).float().to(device)
                q_value, _ = model(state.unsqueeze(dim=0), action.unsqueeze(dim=0).unsqueeze(dim=0))
                lstm_q_values.append(q_value.item())
            #Invoke Boltzmann distribution with T_psi
            intent_specific_policy = Boltzmann(np.array(lstm_q_values), T_psi)
            # 3. Fuse policies and select the best action
            #Original paper describe the argmax operation on square root of the product. However, it is equivalent to argmax on the product itself.
            fused_policy = task_specific_policy * intent_specific_policy
            for a in reversed(np.argsort(fused_policy)):
                if a in legal_action:
                    action = a
                    break
            return action, lstm_q_values
            
    dump_dir = f"{args.result_path}/{env.name}/{args.action_selection_strategy}"
    for episode in range(args.test_episodes):
        frame_array = []
        state, info = env.reset()
        frame_array.append(env.get_frame())
        g_t = 0
        done = False
        previous_q_value = 0
        while not done:
            if not args.action_selection_strategy == "fusion":
                action = select_action(state, 0)
            else:
                # Dynamically modulate T_psi based on accumulated reward g_t
                T_psi = max(args.env.Fusion.T_min, args.env.Fusion.T_max / (1 + np.exp(-args.env.Fusion.slop * (g_t - args.env.Fusion.crt))))
                action, lstm_q_values = select_action(state, T_psi)
                lstm_q_values = np.array(lstm_q_values)
                
                # Reward redistribution as per Equation (1)
                redistributed_reward = lstm_q_values - previous_q_value
                # Shifted reward as per Equation (5)
                shifted_redistributed_reward = redistributed_reward - np.mean(redistributed_reward)
                g_t += shifted_redistributed_reward[action]
                previous_q_value = lstm_q_values
            
            next_state, reward, truncated, terminated, _ = env.step(action)
            frame_array.append(env.get_frame())
            done = truncated + terminated
            state = next_state
        save_gif(frame_array, episode, dump_dir, fps=args.env.fps)

if __name__ == "__main__":
    test()