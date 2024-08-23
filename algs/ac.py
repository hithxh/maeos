from torch import nn
from torch.nn import functional as F
import torch

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
        self.fc2.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)  
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        # 每个状态对应的动作的概率
        x = F.softmax(x, dim=1)  # [b,n_actions]-->[b,n_actions]
        return x


class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(n_hiddens, 1)
        self.fc2.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x
    
# 相当于rllib中的vf_share_layers = True
class ActorCritic(nn.Module):
    def __init__(self, obs_dim,hidden_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)
    
    def act(self, obs):
        logits, _ = self.forward(obs)
        return torch.relu(logits)
    
    def evaluate(self, obs):
        logits, value = self.forward(obs)
        return torch.relu(logits), value