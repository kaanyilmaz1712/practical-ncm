import torch as T
import math
import torch.nn.functional as F

def estimate(model, V, vk, X_set, x, m):
    """
    Estimate P(v) and P(y|do(x))

    Args:
        model (ncm): current neural causal model which is being trained
        V (array): all nodes
        vk (dictionary): current var in dataset
        X_set (array): intervention set
        x (dictionary): intervention value
        m (int): monte carlo sample size

    Returns:
        tensor: joint log probability of current datapoint
    """
    U_shared = {i: T.rand((m,1)) for i, _ in enumerate(model.shared_groups)}
    U_private = {v: T.rand((m,1)) for v in V}
    log_terms = []

    for v in V:
        f_v = model.functions[v]

        if v in X_set:
            if vk[v] != x[v]:
                return T.tensor(-float(math.inf))
            log_pv = T.zeros((m, 1)) # P(X=1 | do(X=1)) = 1
            continue

        inputs = []

        for p in model.parents[v]: 
            inputs.append(T.full((m,1), float(vk[p])))

        for gid, (a,b) in enumerate(model.shared_groups):
            if v in (a,b):
                inputs.append(U_shared[gid])

        inputs.append(U_private[v])

        val_v = float(vk[v])
        phi_v = f_v(T.cat(inputs, dim=1))
        log_pv = F.logsigmoid(phi_v) if val_v == 1 else F.logsigmoid(-phi_v)
        
        log_terms.append(log_pv)

    # joint log-prob via Monte-Carlo
    log_joint = sum(log_terms)
    return T.logsumexp(log_joint, dim=0) - math.log(m)
