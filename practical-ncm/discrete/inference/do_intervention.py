import itertools
from inference.estimate import estimate
import torch as T

import itertools
from inference.estimate import estimate
import torch as T

import itertools
import torch as T


import itertools
import torch as T

def track_Q(model,
            treatment_var,
            x_value,
            outcome_var,
            outcome_target,
            m):

    V = model.nodes

    sum_vars = [
        v for v in V
        if v not in [treatment_var, outcome_var]
    ]

    log_terms = []

    for world_vals in itertools.product(
        *[model.domains[v] for v in sum_vars]
    ):
        v_dict = dict(zip(sum_vars, world_vals))

        v_dict[treatment_var] = x_value
        v_dict[outcome_var] = outcome_target

        log_terms.append(
            estimate(
                model,
                V,
                v_dict,
                X_set=[treatment_var],
                x={treatment_var: x_value},
                m=m
            )
        )

    logQ = T.logsumexp(T.stack(log_terms), dim=0)
    return T.exp(logQ).item()



