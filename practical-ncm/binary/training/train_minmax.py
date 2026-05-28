from inference.estimate import estimate
import itertools
from utility.lambda_regu import lambda_regu
from inference.do_intervention import do_intervention
import torch as T


def train_minmax_models(epoch_num, m, data, min_model, max_model,optim_min, optim_max):
    """
    actual training process of our ncm. pseudocode from paper translated

    Args:
        epoch_num (int): epoch amount
        m (int): monte carlo sample size
        data (DataFrame): dataset to train on
        min_model (nn): ncm which is minimized
        max_model (nn): ncm which is maximized
        optim_min (optimizer): optimizer for min model
        optim_max (optimizer): optimizer for max model

    Returns:
        array: gap history between min and max model ates for visualization
    """
    gap_hist = []
    ate_min_hist, ate_max_hist = [], []

    for epoch in range(epoch_num):
        lamb = lambda_regu(epoch, epoch_num)
        batch_Lmin = T.zeros(1)
        batch_Lmax = T.zeros(1)

        for i in range(len(data)):
            vk_dict = data.iloc[i].to_dict()

            p_min = estimate(min_model, min_model.nodes, vk_dict, [], {}, m)
            p_max = estimate(max_model, max_model.nodes, vk_dict, [], {}, m)

            other_vars = [v for v in min_model.nodes if (v != 'X') and (v != 'Y')]

            log_terms_min, log_terms_max = [], []
            for world_vals in itertools.product([0, 1], repeat=len(other_vars)):
                v_dict = {var: val for var, val in zip(other_vars, world_vals)}
                v_dict['X'] = 1
                v_dict['Y'] = 1

                log_terms_min.append(estimate(min_model, min_model.nodes, v_dict, ['X'], {'X':1}, m))
                log_terms_max.append(estimate(max_model, max_model.nodes, v_dict, ['X'], {'X':1}, m))

            logQ_min = T.logsumexp(T.stack(log_terms_min), dim=0)
            logQ_max = T.logsumexp(T.stack(log_terms_max), dim=0)

            L_min_i = -(p_min + lamb * T.log(1.0 - T.exp(logQ_min) + 1e-8))
            L_max_i = -(p_max + lamb * logQ_max)

            batch_Lmin += L_min_i.squeeze()
            batch_Lmax += L_max_i.squeeze()

        batch_Lmin /= len(data)
        batch_Lmax /= len(data)

        optim_min.zero_grad(); batch_Lmin.backward(); optim_min.step()
        optim_max.zero_grad(); batch_Lmax.backward(); optim_max.step()  

        # ---- ATE tracking ----
        p_y1_do_x1_min = do_intervention(min_model, ['X'], 1)
        p_y1_do_x0_min = do_intervention(min_model, ['X'], 0)
        ATE_min = p_y1_do_x1_min - p_y1_do_x0_min

        p_y1_do_x1_max = do_intervention(max_model, ['X'], 1)
        p_y1_do_x0_max = do_intervention(max_model, ['X'], 0)
        ATE_max = p_y1_do_x1_max - p_y1_do_x0_max

        gap = abs(ATE_max - ATE_min)
        gap_hist.append(gap)
        ate_min_hist.append(ATE_min)
        ate_max_hist.append(ATE_max)

        print(f"Epoch {epoch:03d} | λ={lamb:.4f} | Lmin={batch_Lmin.item():.4f} | Lmax={batch_Lmax.item():.4f} | "
                f"ATE_min={ATE_min:.4f} | ATE_max={ATE_max:.4f} | gap={gap:.4f}")

    return gap_hist