def read_cg(path):
    """
    We need a way to tell the NCM the underlying causal structure. 
    We do that by using a graph which contains information about all nodes, their parents aswell as groups of nodes that share a noise.

    Args:
        path (string): path to graph

    Returns:
        nodes (array): every single V
        parents (dictionary): each node -> parents
        shared_groups (array): if bidirectional -> nodes in each item of shared_groups share a noise
    """
    nodes, dir_edges, bi_edges, shared_groups= [], [], [], []
    mode = None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            if s == "<NODES>": mode = "nodes"; continue
            if s == "<EDGES>": mode = "edges"; continue
            if mode == "nodes":
                nodes.append(s)
            elif mode == "edges":
                if "<->" in s:
                    a,b = [t.strip() for t in s.split("<->")]
                    bi_edges.append((a,b))
                else:
                    a,b = [t.strip() for t in s.split("->")]
                    dir_edges.append((a,b))

    parents = {v: [] for v in nodes}
    for a,b in dir_edges:
        parents[b].append(a)

    for a,b in bi_edges:
        shared_groups.append((a, b))

    return nodes, parents, shared_groups
