import numpy as np
import pandas as pd
import torch as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import os

from models.ncm import NCM
from training.train_minmax import train_minmax_models


SEEDS = [0, 1, 2]

epoch_num = 500
batch_size = 1024
m = 200
lr = 4e-3

def set_seed(seed):
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_discrete_experiment(
    name,
    graph_path,
    data_fn,
    treatment_var,
    outcome_var,
    treatment_value,
    outcome_target
):

    print(f"\n===== {name.upper()} (DISCRETE) =====")

    os.makedirs(f"checkpoints/{name}_discrete", exist_ok=True)
    os.makedirs(f"results/{name}_discrete", exist_ok=True)

    for seed in SEEDS:
        print(f"[{name}] seed={seed}")

        set_seed(seed)

        data = data_fn(seed=seed)

        min_model = NCM(graph_path)
        max_model = NCM(graph_path)

        opt_min = T.optim.Adam(min_model.parameters(), lr=lr)
        opt_max = T.optim.Adam(max_model.parameters(), lr=lr)

        gap_hist = train_minmax_models(
            epoch_num=epoch_num,
            batch_size=batch_size,
            m=m,
            data=data,
            min_model=min_model,
            max_model=max_model,
            optim_min=opt_min,
            optim_max=opt_max,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            treatment_value=treatment_value,
            outcome_target=outcome_target
        )

        T.save(
            min_model.state_dict(),
            f"checkpoints/{name}_discrete/min_model_x{treatment_value}_y{outcome_target}_{seed}.pt"
        )

        T.save(
            max_model.state_dict(),
            f"checkpoints/{name}_discrete/max_model_x{treatment_value}_y{outcome_target}_{seed}.pt"
        )

        np.save(
            f"results/{name}_discrete/gap_x{treatment_value}_y{outcome_target}_{seed}.npy",
            np.array(gap_hist, dtype=np.float32)
        )

        plt.figure(figsize=(8,4))
        plt.plot(
            gap_hist,
            label="GapQ = |P_max(Y=y* | do(X=x*)) - P_min(Y=y* | do(X=x*))|"
        )

        plt.xlabel("Epoch")
        plt.ylabel("ΔGap")
        plt.title(f"{name} Discrete (seed={seed})")

        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"results/{name}_discrete/gap_history_{seed}.svg",
            format="svg"
        )
        plt.close()

def backdoor_data(seed=0):
    rng = np.random.RandomState(seed)

    A = rng.randint(0, 5, size=100000)

    F = np.clip(
        np.round(0.3 * A + rng.normal(0, 0.6, size=100000) + 0.5),
        0, 2
    ).astype(int)

    H = np.clip(
        np.round(2.2 - 0.5 * A + 0.8 * F + rng.normal(0, 0.8, size=100000)),
        0, 3
    ).astype(int)

    M = np.clip(
        np.round(0.6 * H + rng.normal(0, 0.5, size=100000)),
        0, 2
    ).astype(int)

    return pd.DataFrame({'A': A, 'F': F, 'H': H, 'M': M})


def bow_data(seed=0):
    rng = np.random.RandomState(seed)

    U = rng.randint(0, 3, size=100000)

    X = np.clip(
        np.round(0.8 * U + rng.normal(0, 0.8, size=100000)),
        0, 2
    ).astype(int)

    Y = np.clip(
        np.round(1.2 * X + 0.9 * U + rng.normal(0, 1.0, size=100000)),
        0, 3
    ).astype(int)

    return pd.DataFrame({'X': X, 'Y': Y})


if __name__ == "__main__":

    # Backdoor discrete
    run_discrete_experiment(
        name="backdoor",
        graph_path="graphs/backdoor.cg",
        data_fn=backdoor_data,
        treatment_var="F",
        outcome_var="H",
        treatment_value=2,
        outcome_target=3
    )

    # Bow discrete
    run_discrete_experiment(
        name="bow",
        graph_path="graphs/bow.cg",
        data_fn=bow_data,
        treatment_var="X",
        outcome_var="Y",
        treatment_value=2,
        outcome_target=3
    )