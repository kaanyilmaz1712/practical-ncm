import torch.nn as nn
import torch as T
import torch.nn.functional as F

class Simple(nn.Module):
    """
    Continuous structural function:
    outputs mean and log_std of Gaussian
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # mean head
        self.fc_mu = nn.Linear(hidden_dim, 1)

        # log std head
        self.fc_logstd = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logstd.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logstd.bias)

    def forward(self, x):
        h = T.relu(self.fc1(x))
        h = T.relu(self.fc2(h))

        mu = self.fc_mu(h)
        log_std = self.fc_logstd(h)

        # optional: clamp for stability
        log_std = T.clamp(log_std, -5, 3)

        return mu, log_std
