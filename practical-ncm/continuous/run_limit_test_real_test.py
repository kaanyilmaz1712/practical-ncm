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

lr = 5e-4
batch_size = 1024
m = 300


def set_seed(seed):
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_basic_scm(n, seed=0):
    rng = np.random.RandomState(seed)

    eps_X = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)

    Z = rng.uniform(0, 50, size=n)
    X = 0.5 * Z + 10 * eps_X
    Y = 0.01 * (50 - Z * Z) + 0.5 * X + 30 * eps_Y

    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})


def generate_real_scm(n, seed=0):
    rng = np.random.RandomState(seed)

    A = rng.uniform(0, 100, size=n)

    eps_F = rng.normal(10, 10, size=n)
    eps_H = rng.normal(40, 30, size=n)
    eps_M = rng.normal(20, 10, size=n)

    F = 0.5 * A + eps_F
    H = (1/100) * (100 - A**2) + 0.5 * F + eps_H
    M = 0.5 * H + eps_M

    return pd.DataFrame({"A": A, "F": F, "H": H, "M": M})


def train_and_save(
    base_path,
    seed,
    tag,
    data,
    graph_path,
    epoch_num,
    treatment_var,
    outcome_var,
    x_values
):

    os.makedirs(f"{base_path}/checkpoints/{seed}", exist_ok=True)
    os.makedirs(f"{base_path}/results/{seed}", exist_ok=True)

    # normalize
    data = (data - data.mean()) / data.std()

    min_model = NCM(graph_path, hidden_dim=64)
    max_model = NCM(graph_path, hidden_dim=64)

    opt_min = T.optim.Adam(min_model.parameters(), lr=lr)
    opt_max = T.optim.Adam(max_model.parameters(), lr=lr)

    gap_hist = train_minmax_models(
        min_model=min_model,
        max_model=max_model,
        optim_min=opt_min,
        optim_max=opt_max,
        data=data,
        epoch_num=epoch_num,
        batch_size=batch_size,
        m=m,
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        x_values=x_values
    )

    # save
    T.save(
        min_model.state_dict(),
        f"{base_path}/checkpoints/{seed}/min_{tag}.pt"
    )
    T.save(
        max_model.state_dict(),
        f"{base_path}/checkpoints/{seed}/max_{tag}.pt"
    )

    np.save(
        f"{base_path}/results/{seed}/gap_{tag}.npy",
        np.array(gap_hist, dtype=np.float32)
    )

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(gap_hist)
    plt.xlabel("Epoch")
    plt.ylabel("|Q_max - Q_min|")
    plt.title(f"{tag} (seed={seed})")
    plt.grid(True)

    plt.savefig(
        f"{base_path}/results/{seed}/gap_{tag}.svg",
        format="svg"
    )
    plt.close()


def run_sample_size_test():

    base_path = "1-samplesize-test"
    data_sizes = [100000, 50000, 20000, 10000, 5000, 2000, 1000, 500]

    for seed in SEEDS:
        for n in data_sizes:
            print(f"[SampleSize] seed={seed}, n={n}")

            set_seed(seed)

            data = generate_basic_scm(n, seed)

            train_and_save(
                base_path=base_path,
                seed=seed,
                tag=f"n{n}",
                data=data,
                graph_path="graphs/backdoor.cg",
                epoch_num=500,
                treatment_var="X",
                outcome_var="Y",
                x_values=[-1, 0, 1]
            )


def run_epoch_test():

    base_path = "2-epoch-test"
    epochs_list = [500, 300, 200, 100, 50, 25]

    for seed in SEEDS:
        for epochs in epochs_list:
            print(f"[EpochTest] seed={seed}, epochs={epochs}")

            set_seed(seed)

            data = generate_basic_scm(100000, seed)

            train_and_save(
                base_path=base_path,
                seed=seed,
                tag=f"ep{epochs}",
                data=data,
                graph_path="graphs/backdoor.cg",
                epoch_num=epochs,
                treatment_var="X",
                outcome_var="Y",
                x_values=[-1, 0, 1]
            )


def run_real_scm_test():

    base_path = "3-realscm-test"
    x_values_list = [[0], [-1,0,1], [-2,-1,0,1,2]]

    for seed in SEEDS:
        for i, x_set in enumerate(x_values_list):
            print(f"[RealSCM] seed={seed}, x_set={x_set}")

            set_seed(seed)

            data = generate_real_scm(100000, seed)

            train_and_save(
                base_path=base_path,
                seed=seed,
                tag=f"xset{i}",
                data=data,
                graph_path="graphs/realscm.cg",
                epoch_num=500,
                treatment_var="F",
                outcome_var="H",
                x_values=x_set
            )


if __name__ == "__main__":

    run_sample_size_test()
    run_epoch_test()
    run_real_scm_test()