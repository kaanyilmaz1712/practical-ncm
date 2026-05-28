import itertools
from inference.estimate import estimate
import torch as T

def do_intervention(model, X_set, x_value, target='Y', m=1000):
    # estimate P(Y=1 | do(X=x)) by summing over all not-intervened vars besides Y (every single combination not just one dp)
    
    V = model.nodes  # alle Variablen im Graphen
    other_vars = [v for v in V if (v not in X_set) and (v != target)]

    # all combinations for non-intervened vars
    combos = list(itertools.product([0,1], repeat=len(other_vars)))

    log_terms = []

    for combo in combos:
        v_dict = {v: val for v, val in zip(other_vars, combo)}
        # target variable (y)
        v_dict[target] = 1
        v_dict.update({x: x_value for x in X_set})


        log_terms.append(estimate(model, V, v_dict, X_set, {x: x_value for x in X_set}, m))

    logQ = T.logsumexp(T.stack(log_terms), dim=0)
    return T.exp(logQ).detach().item()