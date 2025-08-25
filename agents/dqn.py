import os
import copy
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from torch.nn.utils import clip_grad_norm_
from agents.agent_utils.networks import Critic
from agents.agent_utils.buffer import ReplayBuffer
from architectures.common_utils import save_gif, zip_strict


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self, Network: dict, Test: dict, **kwargs):
        """Initialize an Agent object.
        """
        super(DQN, self).__init__()
        
        self.device = device
        
        self.config = Network
        self.agent_name = Network["name"]
        self.input_dim = Network["input_dim"]
        self.action_size = Network["action_dim"]
        self.buffer_size = Network["buffer_size"]
        self.batch_size = Network["batch_size"]
        self.gamma = Network["gamma"]
        self.tau = Network["tau"]
        fc_hidden_size = Network["fc_hidden_size"]
        learning_rate = Network["lr"]
        self.clip_grad_param = Network["clip_grad_param"]
        self.epsilon_start = Network["epsilon_start"]
        self.epsilon_end = Network["epsilon_end"]
        self.epsilon_decay = Network["epsilon_decay"]
        self.initial_random_samples = Network["initial_random_samples"]
        self.test_episodes = Network["test_episodes"]
        self.video_dir =  Network["video_save_path"]
        self.hard_update = Network["hard_update"]
        
        self.critic = Critic(self.input_dim, self.action_size, fc_hidden_size).to(device)
        self.critic_target = Critic(self.input_dim, self.action_size, fc_hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        #Buffer for storing the experience
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, device=device)

        #Action_space
        self.action_space = torch.tensor(range(self.action_size)).to(device)
        
    def get_action(self, state, steps=0):
        if random.random() < self.epsilon or steps < self.initial_random_samples:
            return random.randrange(self.action_size)
        else:
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                self.critic.eval()
                q = self.critic(state)
                self.critic.train()
            return q.argmax().item()
    
    def learn(self, timstep):
        """
        DQN update rule
        """
        
        if len(self.buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, truncated, terminated = self.buffer.sample()
        done = truncated + terminated
        #Compute losses--------------------------------------------------
        # ---------------------------- Critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_next = self.critic(next_states)
            next_actions = Q_next.argmax(dim=1, keepdim=True)
            
            Q_target_next = self.critic_target(next_states)
            Q_target_next = Q_target_next.gather(1, next_actions)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - done) * Q_target_next) 

        # Compute critic loss
        q = self.critic(states)
        action_q_values = q.gather(1, actions.long())
                
        critic_loss = F.smooth_l1_loss(action_q_values, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
        self.critic_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        if not self.hard_update:
            self.soft_update(self.critic, self.critic_target)
        else:
            if timstep % self.config["target_update_freequency"] == 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
        
        metric = {"Critic loss": critic_loss.item(), 
                  "epsilon" : self.epsilon}
        
        return metric
    
    
    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        with torch.no_grad():
            for param, target_param in zip_strict(local_model.parameters(), target_model.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            
    def do_post_episode_processing(self, steps_done):
        # Update epsilon at the end of each episode
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * steps_done / self.epsilon_decay)
                       
    def add_transition_to_buffer(self, transition):
        self.buffer.add(transition)
                 
    def do_pre_task_proceessing(self):
        self.epsilon = self.epsilon_start
        
    def do_post_task_processing(self):
        self.buffer.clear()
        
    def test(self, env, fps):
        """Test the agent in the environment."""
        dump_dir = f"{self.video_dir}/{env.name}/{self.agent_name}"
        epsilon = self.epsilon
        self.epsilon = 0
        for episode in range(self.test_episodes):
            frame_array = []
            state, info = env.reset()
            frame_array.append(env.get_frame())
            cumulative_reward = 0
            done = False
            while not done:
                action = self.get_action(state, self.initial_random_samples+1)
                next_state, reward, truncated, terminated, _ = env.step(action)
                frame_array.append(env.get_frame())
                done = truncated + terminated
                cumulative_reward += reward
                state = next_state
            # write_video(frame_array, episode, dump_dir, frameSize=(env.unwrapped.get_frame().shape[1], env.unwrapped.get_frame().shape[0]))
            save_gif(frame_array, episode, dump_dir, fps=fps)
        self.epsilon = epsilon
        
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path + "DQN.tar", map_location=device, weights_only=True)
        self.critic.load_state_dict(params["critic"])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer.load_state_dict(params["critic_optim"])
        print("[INFO] loaded the DQN model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "critic": self.critic.state_dict(),
                "critic_optim" : self.critic_optimizer.state_dict(),
                }
        save_dir = dump_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] DQN model saved to: ", checkpoint_path)