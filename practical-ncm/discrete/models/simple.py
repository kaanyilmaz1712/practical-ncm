import torch.nn as nn
import torch as T

class Simple(nn.Module):
    """
    Basic neural network for each function of F of our variables.
    Used in the actual NCM

    Args:
        nn: nn structure
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Glorot Initialization -> like in paper, 
        # standard initialization is the reason why the difference in min-max starts often starts close to 0
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, inputs):
        """
        Pass through in network

        Args:
            inputs (concatted array or single value): parents of current variable in V

        Returns:
            data: prediction
        """
        data = T.relu(self.fc1(inputs))
        data = T.relu(self.fc2(data))
        data = self.fc3(data)
        return data