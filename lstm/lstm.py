import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from architectures.mlp import MLP, FiLM
from widis_lstm_tools.nn import LSTMLayer
import torch.optim.lr_scheduler as lr_scheduler
from architectures.common_utils import custom_action_encoding, get_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, Network: dict, Test: dict, **kwargs):
        super(LSTM, self).__init__()
        feature_size = Network["feature_size"]
        self.n_action = Network["n_actions"]
        lstm_units = Network["n_units"]
        self.batch_size = Network["batch_size"]
        self.q_estimate_loss_window = Network["q_estimate_loss_window"]
        self.aux_loss_weight = Network["aux_loss_weight"]
        self.gradient_clipping = Network["gradient_clipping"]
        self.action_encoding_dim = round(feature_size / self.n_action) * self.n_action
        
        #Networks components
        self.film = FiLM(feature_size, self.action_encoding_dim)
        self.lstm_layer = LSTMLayer(in_features=feature_size, out_features=lstm_units,
                                    w_ci=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False),
                                    w_ig=(False, lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs)),
                                    w_og=False,
                                    b_ci=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                                    b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-5, *args, **kwargs),
                                    b_og=False,
                                    a_out=lambda x: x)
        self.post_lstm_linear_layer = MLP(Network["n_units"], 1, [Network["n_units"]//2], hidden_activation="lrelu")
        self.aux_lstm_linear_layer = MLP(Network["n_units"], 1, [Network["n_units"]//2], hidden_activation="lrelu")
        
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=Network["lr"],
                                    weight_decay=Network["l2_regularization"],
                                    eps=Network["adam_eps"],
                                    )
        
        #Scheduler
        sched_cfg = Network.get("scheduler", None).get("type", None)
        if sched_cfg:
            self.scheduler = get_scheduler(Network.get("scheduler", None), self.optimizer, Network["lr"])
        else:
            self.scheduler = None
    
    def forward(self, states, action):
        action_embedding = custom_action_encoding(action, self.n_action, self.action_encoding_dim)
        conditional_input = self.film(states, action_embedding)
        lstm_input = self.lstm_layer(conditional_input, return_all_seq_pos = True)[0]
        B, T, _ = lstm_input.shape
        q_values = self.post_lstm_linear_layer(lstm_input.view(B*T, -1)).view(B,T)
        q_estimate = self.aux_lstm_linear_layer(lstm_input.view(B*T, -1)).view(B,T)
        return q_values, q_estimate
    
    def get_action(self, state, legal_actions):
        state = state.unsqueeze(0).to(device)  # add batch dim
        best_action, best_q = None, -float("inf")

        with torch.no_grad():
            for a in legal_actions:
                a_tensor = torch.tensor([[a]], dtype=torch.float32, device=device)
                q_value, _ = self(state, a_tensor)
                q_value = q_value.item()
                if q_value > best_q:
                    best_q, best_action = q_value, a

        if best_action is None:
            raise ValueError("No legal actions available!")
        return int(best_action)
        
    def calculate_main_loss(self, q, label, length):
        all_timestep_loss = F.mse_loss(q, label.unsqueeze(-1), reduction = 'none')
        seq_len = length[:] - 1
        all_timestep_loss_indexed = all_timestep_loss[range(q.size(0)), seq_len]
        return all_timestep_loss_indexed
    
    def calculate_aux_loss(self, q, label, length):

        # B x L
        all_timestep_loss = F.mse_loss(q, label.unsqueeze(-1), reduction = 'none')
            
        # Create the mask
        self.mask = torch.zeros_like(all_timestep_loss)
        for l_num, l in enumerate(length):
            self.mask[l_num, :l] = 1
                
        #Multiplying with the self.mask to avoid the padded sequence
        all_timestep_loss = all_timestep_loss * self.mask

        # Average for each sequenace
        self.mean_all_timestep_loss_along_sequence = all_timestep_loss.sum(1) / self.mask.sum(1)
    
        return self.mean_all_timestep_loss_along_sequence
    
    def q_estimate_loss(self, q_values, q_estimate):
        q_values = q_values[:,self.q_estimate_loss_window:,...]
        q_values_estimate = q_estimate[:,:-self.q_estimate_loss_window,...]
        loss = F.mse_loss(q_values_estimate, q_values, reduction ='none')
        loss = loss.sum(1).squeeze(-1)
        return loss
    
    def learn(self, data):
        states = data[0]
        actions = data[1]
        lengths = data[2]
        labels = data[3]
        
        q_values, q_estimate = self(states, actions)
        main_loss = self.calculate_main_loss(q_values, labels, lengths).mean()
        aux_loss = self.calculate_aux_loss(q_values, labels, lengths).mean()
        q_estimate_loss = self.q_estimate_loss(q_values, q_estimate).mean()
        loss = main_loss + self.aux_loss_weight*(aux_loss + q_estimate_loss)
        # loss = main_loss + self.aux_loss_weight*aux_loss
        
        #Optimization
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping)
        self.optimizer.step()
        
        # Step scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss.item())  # needs validation metric
            else:
                self.scheduler.step()
        
        # Current LR
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        metrics = {"Main loss" : main_loss.item(),
                   "Aux loss" : aux_loss.item(),
                   "Q estimate loss" : q_estimate_loss.item(),
                   "Loss" : loss.item(),
                   "LR": current_lr}
        return metrics
    
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path + "LSTM.tar", map_location=device)
        self.film.load_state_dict(params["film"])
        self.lstm_layer.load_state_dict(params["lstm_layer"])
        self.post_lstm_linear_layer.load_state_dict(params["post_lstm_linear_layer"])
        self.aux_lstm_linear_layer.load_state_dict(params["aux_lstm_linear_layer"])
        print("[INFO] loaded the LSTM model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "film": self.film.state_dict(),
                "lstm_layer" : self.lstm_layer.state_dict(),
                "post_lstm_linear_layer" : self.post_lstm_linear_layer.state_dict(),
                "aux_lstm_linear_layer" : self.aux_lstm_linear_layer.state_dict()
                }
        save_dir = dump_dir
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] LSTM model saved to: ", checkpoint_path)
        
    
    