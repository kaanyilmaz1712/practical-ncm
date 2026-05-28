import torch as T
import math


def estimate(model, V, vk, X_set, x, m):
    """
    Continuous version:
    Monte-Carlo estimate of joint log-density
    using Gaussian structural equations.
    """

    U_shared = {i: T.rand((m, 1)) for i, _ in enumerate(model.shared_groups)}
    U_private = {v: T.rand((m, 1)) for v in V}

    log_terms = []

    for v in V:
        if v in X_set:
            if abs(float(vk[v]) - float(x[v])) > 1e-8:
                return T.tensor(-float('inf'))

            log_terms.append(T.zeros((m, 1)))
            continue

        f_v = model.functions[v]

        inputs = []

        for p in model.parents[v]:
            inputs.append(T.full((m, 1), float(vk[p])))

        for gid, (a, b) in enumerate(model.shared_groups):
            if v in (a, b):
                inputs.append(U_shared[gid])

        inputs.append(U_private[v])

        x_in = T.cat(inputs, dim=1)

        mu, log_std = f_v(x_in)

        sigma = T.exp(log_std)

        v_val = float(vk[v])
        v_tensor = T.full((m, 1), v_val)

        log_pv = (
            -0.5 * ((v_tensor - mu) ** 2) / (sigma ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )

        log_terms.append(log_pv)

    log_joint = sum(log_terms)

    return T.logsumexp(log_joint, dim=0) - math.log(m)


import torch as T
import math

def estimate_batch(model, V, batch_vk, X_set, x, m):
    """
    batch_vk: dict of tensors (batch_size,)
    """

    batch_size = batch_vk[V[0]].shape[0]

    # expand to (batch, m, 1)
    def expand(v):
        return batch_vk[v].view(batch_size, 1, 1).expand(batch_size, m, 1)

    # noise
    U_private = {v: T.randn((batch_size, m, 1)) for v in V}
    U_shared = {i: T.randn((batch_size, m, 1)) for i, _ in enumerate(model.shared_groups)}

    log_terms = []

    for v in V:

        if v in X_set:
            log_terms.append(T.zeros((batch_size, m, 1)))
            continue

        inputs = []

        for p in model.parents[v]:
            inputs.append(expand(p))

        for gid, (a, b) in enumerate(model.shared_groups):
            if v in (a, b):
                inputs.append(U_shared[gid])

        inputs.append(U_private[v])

        x_in = T.cat(inputs, dim=2)  # (batch, m, input_dim)

        # flatten for NN
        x_in_flat = x_in.view(batch_size * m, -1)

        mu, log_std = model.functions[v](x_in_flat)

        mu = mu.view(batch_size, m, 1)
        log_std = log_std.view(batch_size, m, 1)

        sigma = T.exp(log_std)

        v_tensor = expand(v)

        log_pv = (
            -0.5 * ((v_tensor - mu) ** 2) / (sigma ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )

        log_terms.append(log_pv)

    log_joint = sum(log_terms)  # (batch, m, 1)

    return T.logsumexp(log_joint, dim=1) - math.log(m)  # (batch,1)