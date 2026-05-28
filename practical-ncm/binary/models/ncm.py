import torch.nn as nn
from models.simple import Simple
from utility.read_cg import read_cg


class NCM(nn.Module):
    """
    Actual NCM -> input graph with underlying causal structure

    nodes (array): nodes in graph
    parents (dictionary): each nodes gets their parents assigned
    shared_groups (array): each element in shared_groups contains nodes that share a noise (bidirectional arrow in graph)
    functions (nn-dictionary): for each node define Simple Network

    Args:
        nn: nn structure
    """
    def __init__(self, cg_path, hidden_dim=64):
        super().__init__()
        nodes, parents, shared_groups = read_cg(cg_path)
        self.nodes = nodes
        self.parents = parents
        self.shared_groups = shared_groups

        self.functions = nn.ModuleDict()

        # calculate input dimension based on nodes parents and shared nodes plus one for own noise
        for node in nodes:
            num_parents = len(parents[node])
            num_shared = sum([1 for (a,b) in shared_groups if node in (a,b)])
            input_dim = num_parents + num_shared + 1
            # for each node create a seperate Simple NN = aquivalent to term function eg. X = 2Y + 0.4Z but as a neural net
            self.functions[node] = Simple(input_dim, hidden_dim)

    def forward(self, inputs):
        # for each node run its network (inside functions) with corresponding input (parents, shared noise, noise)
        
        outputs = {}
        for node, net in self.functions.items():
            outputs[node] = net(inputs[node])