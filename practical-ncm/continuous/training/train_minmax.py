
import torch as T
from inference.estimate import estimate_batch
from inference.do_intervention import track_Q_continuous
from utility.lambda_regu import lambda_regu
import numpy as np

def train_minmax_models(epoch_num, batch_size, m, data,
    min_model, max_model,
    optim_min, optim_max,
    treatment_var, outcome_var,
    x_values, seed=0):

    gap_hist = []

    for epoch in range(epoch_num):

        lamb = lambda_regu(epoch, epoch_num)

        batch = data.sample(
            n=min(batch_size, len(data)),
            random_state=seed + epoch
        )

        batch_dict = {
            v: T.tensor(batch[v].values, dtype=T.float32)
            for v in min_model.nodes
        }

        p_min = estimate_batch(min_model, min_model.nodes, batch_dict, [], {}, m)
        p_max = estimate_batch(max_model, max_model.nodes, batch_dict, [], {}, m)

        logQ_min_total = T.tensor(0.0)
        logQ_max_total = T.tensor(0.0)

        y_grid = T.linspace(-2, 2, 5)

        for x_val in x_values:

            batch_query = dict(batch_dict)

            batch_query[treatment_var] = T.full_like(
                batch_query[treatment_var],
                float(x_val)
            )

            Ey_min = T.tensor(0.0)
            Ey_max = T.tensor(0.0)
            Z_min = T.tensor(0.0)
            Z_max = T.tensor(0.0)

            for y_val in y_grid:

                batch_query[outcome_var] = T.full_like(
                    batch_query[outcome_var],
                    float(y_val)
                )

                logp_min = estimate_batch(
                    min_model,
                    min_model.nodes,
                    batch_query,
                    X_set=[treatment_var],
                    x={treatment_var: float(x_val)},
                    m=m
                )

                logp_max = estimate_batch(
                    max_model,
                    max_model.nodes,
                    batch_query,
                    X_set=[treatment_var],
                    x={treatment_var: float(x_val)},
                    m=m
                )

                logp_min_val = logp_min.mean()
                logp_max_val = logp_max.mean()

                p_min_val = T.exp(logp_min_val)
                p_max_val = T.exp(logp_max_val)

                Ey_min += y_val * p_min_val
                Ey_max += y_val * p_max_val

                Z_min += p_min_val
                Z_max += p_max_val

            Ey_min = Ey_min / (Z_min + 1e-8)
            Ey_max = Ey_max / (Z_max + 1e-8)

            logQ_min_total += Ey_min
            logQ_max_total += Ey_max

        logQ_min_total /= len(x_values)
        logQ_max_total /= len(x_values)

        p_min = p_min.mean()
        p_max = p_max.mean()

        L_min = -(p_min - lamb * logQ_min_total)
        L_max = -(p_max + lamb * logQ_max_total)

        L_min = L_min.mean()
        L_max = L_max.mean()

        optim_min.zero_grad()
        L_min.backward()
        T.nn.utils.clip_grad_norm_(min_model.parameters(), 5.0)
        optim_min.step()

        optim_max.zero_grad()
        L_max.backward()
        T.nn.utils.clip_grad_norm_(max_model.parameters(), 5.0)
        optim_max.step()

        if epoch % 10 == 0:

            x_track = [-2, -1, 0, 1, 2]

            gaps = []

            for x_val in x_track:

                Q_min = track_Q_continuous(
                    min_model, data, treatment_var, outcome_var, x_val
                )

                Q_max = track_Q_continuous(
                    max_model, data, treatment_var, outcome_var, x_val
                )

                gaps.append(abs(Q_max - Q_min))

            gap_mean = np.mean(gaps)
            gap_max  = np.max(gaps)

            print(f"Epoch {epoch:03d} | Gap_mean={gap_mean:.4f} | Gap_max={gap_max:.4f}")

            gap_hist.append(gap_mean)

        print(epoch)
    return gap_hist

