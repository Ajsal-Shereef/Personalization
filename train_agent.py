import torch
import wandb
import hydra
import warnings
import numpy as np

from PIL import Image
from collections import deque
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf
from agents.agent_utils.trajectory_buffer import TrajectoryReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_agent(env, cfg):
    cfg.agent.Network.action_dim = int(env.action_space.n)
    cfg.agent.Network.input_dim = int(env.observation_space.shape[0])
    return hydra.utils.instantiate(cfg)

@hydra.main(version_base=None, config_path="configs", config_name="train_agent")
def train(args: DictConfig) -> None:
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    
    if args.env.name ==  "Highway":
        from env.highway import Highway
        env = Highway(args.env)
        #Initialisng the labeller
        from Labeller.oracle import HumanOracleHighway
        labeller = HumanOracleHighway(env, args.mode)
    elif args.env.name ==  "Pong":
        from env.pong import Pong
        env = Pong(args.env)
        #Initialisng the labeller
        from Labeller.oracle import HumanOraclePong
        labeller = HumanOraclePong(env, args.mode)
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    if args.use_wandb:
        wandb.init(project="Project 1", name=f"{args.agent.Network.name}_{args.env.name}", config=OmegaConf.to_container(args, resolve=True))

    print("[INFO] Agent name: ", args.agent.Network.name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    #Make the agent
    agent = make_agent(env, args)
    agent = agent.agent.to(device)
    
    model_dir = create_dump_directory(f"model_weights/{args.agent.Network.name}")
    print("[INFO] Dump dir: ", model_dir)
    
    #Dumping the training config files
    config_path = os.path.join(model_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)
    
    #Initialising trajectory buffer
    trajectory_buffer = TrajectoryReplayBuffer(env.observation_space.shape[0], args.env.max_steps, int(np.floor(args.total_timestep/args.env.max_steps)), device)
    
    episode_states = []
    episode_next_states = []
    episode_actions = []
    episode_labels = []
    env_total_steps = 0
    env_episode_steps = 0
    env_episodes = 0
    agent.do_pre_task_proceessing()   
    state, info = env.reset()
    labeller.update_counts(info if args.env.name ==  "Highway" else state)
    cumulative_reward = 0
    average_episodic_return = deque(maxlen=10)
    for i in range(1, args.total_timestep+1):
        action = agent.get_action(state, env_total_steps)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated + truncated
        
        labeller.update_counts(info if args.env.name ==  "Highway" else state)
        episode_states.append(state)
        episode_next_states.append(next_state)
        episode_actions.append(action)
        episode_labels.append(labeller.get_human_feedback(state))
        
        agent.add_transition_to_buffer((state, action, reward, next_state, terminated, truncated))
        metric = agent.learn()
        state = next_state
        cumulative_reward += reward
        env_total_steps += 1
        env_episode_steps += 1
        metric["Returns"] = cumulative_reward
        metric["Average episodic returns"] = np.mean(average_episodic_return) if len(average_episodic_return) > 0 else 0
        metric["Episode steps"] = env_episode_steps
        metric["Env total steps"] = env_total_steps
        metric["Env episode"] = env_episodes
        metric["Buffer size"] = agent.buffer.__len__()
        if done:
            state, info = env.reset()
            env_episodes += 1
            average_episodic_return.append(cumulative_reward)
            env_episode_steps = 0
            cumulative_reward = 0
            agent.do_post_episode_processing(env_total_steps)
            trajectory_buffer.add((episode_states, episode_actions, episode_labels, episode_next_states))
            labeller.reset_episode_count()
            episode_states = []
            episode_next_states = []
            episode_actions = []
            episode_labels = []

        if args.use_wandb and env_total_steps%args.log_every==0:
            wandb.log(metric)
        if i % args.save_every == 0:
            agent.save(f"{model_dir}/", save_name=f"{args.agent.Network.name}")
    #Saving the trajectory data
    trajectory_buffer.save_buffer_data(f"{args.agent.Network.interaction_data_path}/{args.env.name}/{args.mode}", mode=args.mode)
    agent.eval()
    agent.test(env, args.env.fps)
    #Saving the model
    agent.save(f"{model_dir}/", save_name=f"{args.agent.Network.name}")
    
if __name__ == "__main__":
    train()