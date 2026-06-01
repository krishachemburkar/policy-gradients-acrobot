import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)   
        self.fc2 = nn.Linear(hidden_size, action_dim)  

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs
    

class Baseline(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)   
        self.fc2 = nn.Linear(hidden_size, 1)   

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        value = self.fc2(hidden)
        return value
    