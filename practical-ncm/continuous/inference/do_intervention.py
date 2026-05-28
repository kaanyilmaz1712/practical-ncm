import itertools
from inference.estimate import estimate
import torch as T

def track_Q_continuous(model, data, treatment_var, outcome_var, x_val, m=300):
    model.eval()

    with T.no_grad():

        U = {v: T.randn((m,1)) for v in model.nodes}

        for (a,b) in model.shared_groups:
            U[(a,b)] = T.randn((m,1))

        values = {}

        for v in model.nodes:

            if v == treatment_var:
                values[v] = T.full((m,1), float(x_val))
                continue

            inputs = []

            for p in model.parents[v]:
                inputs.append(values[p])

            for (a,b) in model.shared_groups:
                if v in (a,b):
                    inputs.append(U[(a,b)])

            inputs.append(U[v])

            x_in = T.cat(inputs, dim=1)

            mu, log_std = model.functions[v](x_in)
            eps = T.randn_like(mu)

            values[v] = mu + T.exp(log_std) * eps

        return values[outcome_var].mean().item()
    

def ncm_do(model, val, m=50000):

    model.eval()

    with T.no_grad():
        U = {v: T.randn((m,1)) for v in model.nodes}

        for (a,b) in model.shared_groups:
            U[(a,b)] = T.randn((m,1))

        values = {}

        for v in model.nodes:

            if v == "X":
                values[v] = T.full((m,1), float(val))
                continue

            inputs = [values[p] for p in model.parents[v]]

            for (a,b) in model.shared_groups:
                if v in (a,b):
                    inputs.append(U[(a,b)])

            inputs.append(U[v])

            x_in = T.cat(inputs, dim=1)
            mu, log_std = model.functions[v](x_in)
            eps = T.randn_like(mu)

            values[v] = mu + T.exp(log_std) * eps

        return values["Y"].mean().item()