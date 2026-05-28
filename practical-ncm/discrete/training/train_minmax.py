from inference.estimate import estimate
import itertools
from utility.lambda_regu import lambda_regu
from inference.do_intervention import track_Q
import torch as T



def train_minmax_models(epoch_num, batch_size, m, data,
                        min_model, max_model,
                        optim_min, optim_max,
                        treatment_var, outcome_var, 
                        treatment_value, outcome_target):
    gap_hist = []

    


    for epoch in range(epoch_num):
        lamb = lambda_regu(epoch, epoch_num)
        batch_Lmin = 0.0
        batch_Lmax = 0.0

        batch = data.sample(n=min(batch_size, len(data)))

        for _, row in batch.iterrows():
            vk_dict = row.to_dict()

            p_min = estimate(min_model, min_model.nodes, vk_dict, [], {}, m)
            p_max = estimate(max_model, max_model.nodes, vk_dict, [], {}, m)


            other_vars = [
                v for v in min_model.nodes
                if v not in [treatment_var, outcome_var]
            ]

            fixed_vars = [
                v for v in min_model.nodes
                if v not in other_vars + [treatment_var, outcome_var]
            ]

            log_terms_min, log_terms_max = [], []

            for world_vals in itertools.product(
                *[min_model.domains[v] for v in other_vars]
            ):
                v_dict = {}

                v_dict.update(dict(zip(other_vars, world_vals)))

                for v in fixed_vars:
                    v_dict[v] = vk_dict[v]


                v_dict[treatment_var] = treatment_value
                v_dict[outcome_var] = outcome_target

                log_terms_min.append(
                    estimate(min_model, min_model.nodes, v_dict, [treatment_var], {treatment_var: treatment_value}, m)
                )
                log_terms_max.append(
                    estimate(max_model, max_model.nodes, v_dict, [treatment_var], {treatment_var: treatment_value}, m)
                )

            logQ_min = T.logsumexp(T.stack(log_terms_min), dim=0)
            logQ_max = T.logsumexp(T.stack(log_terms_max), dim=0)

            L_min_i = -(p_min + lamb * T.log(1.0 - T.exp(logQ_min) + 1e-8))
            L_max_i = -(p_max + lamb * logQ_max)

            batch_Lmin += L_min_i
            batch_Lmax += L_max_i

        batch_Lmin /= len(batch)
        batch_Lmax /= len(batch)

        optim_min.zero_grad()
        batch_Lmin.backward()
        optim_min.step()

        optim_max.zero_grad()
        batch_Lmax.backward()
        optim_max.step()


        x_star = treatment_value
        y_star = outcome_target

        Q_min = track_Q(
            min_model,
            treatment_var,
            x_star,
            outcome_var,
            y_star,
            m=3000
        )

        Q_max = track_Q(
            max_model,
            treatment_var,
            x_star,
            outcome_var,
            y_star,
            m=3000
        )

        gap = abs(Q_max - Q_min)

        print(
            f"Epoch {epoch:03d} | "
            f"Lmin={batch_Lmin.item():.4f} | "
            f"Lmax={batch_Lmax.item():.4f} | "
            f"Gap={gap:.4f} | "
            f"Qmin={Q_min:.4f} | Qmax={Q_max:.4f}"
        )

        gap_hist.append(gap)

    return gap_hist