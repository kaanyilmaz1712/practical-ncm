import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch as T
from models.ncm import NCM
from training.train_minmax import train_minmax_models

import random
import os

seeds = [0]

base_path = "0-basic-graphs"

for graph in ["backdoor", "frontdoor", "bow", "extended_bow"]:
    for mode in ["linear", "nonlinear"]:
        os.makedirs(f"{base_path}/checkpoints/{graph}/{mode}", exist_ok=True)
        os.makedirs(f"{base_path}/results/{graph}/{mode}", exist_ok=True)

def scm_backdoor_linear(n=10000, seed=0):
    rng = np.random.RandomState(seed)
    Z = rng.uniform(0, 50, size=n)
    eps_X = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)
    X = 0.5 * Z + 5 * eps_X
    Y = 0.6 * X + 0.4 * Z + 5 * eps_Y
    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

def scm_backdoor_nonlinear(n=10000, seed=0):
    rng = np.random.RandomState(seed)

    Z = rng.uniform(0, 50, size=n)

    eps_X = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)

    X = 0.01 * (Z**2) + 0.5 * Z + 5 * eps_X

    Y = 0.02 * (X**2) + 0.5 * X + 0.4 * Z + 5 * eps_Y
    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

def scm_frontdoor_linear(n=10000, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 50, size=n)
    eps_X = rng.normal(0, 1, size=n)
    eps_Z = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)
    X = 0.5 * U + 5 * eps_X
    Z = 1.0 * X + 5 * eps_Z
    Y = 1.0 * Z + 0.5 * U + 5 * eps_Y
    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

def scm_frontdoor_nonlinear(n=10000, seed=0):
    rng = np.random.RandomState(seed)

    U = rng.uniform(0, 50, size=n)

    eps_X = rng.normal(0, 1, size=n)
    eps_Z = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)

    X = 0.01 * (U**2) + 0.5 * U + 5 * eps_X

    Z = 0.02 * (X**2) + 1.0 * X + 5 * eps_Z

    Y = 0.02 * (Z**2) + 1.0 * Z + 0.5 * U + 5 * eps_Y

    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

def scm_bow_linear(n=10000, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 50, size=n)
    eps_X = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)
    X = 0.5 * U + 5 * eps_X
    Y = 0.6 * X + 0.6 * U + 5 * eps_Y
    return pd.DataFrame({'X': X, 'Y': Y})

def scm_bow_nonlinear(n=10000, seed=0):
    rng = np.random.RandomState(seed)

    U = rng.uniform(0, 50, size=n)

    eps_X = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)

    X = 0.01 * (U**2) + 0.5 * U + 5 * eps_X

    Y = 0.02 * (X**2) + 0.6 * X + 0.6 * U + 5 * eps_Y

    return pd.DataFrame({'X': X, 'Y': Y})

def scm_extended_bow_linear(n=10000, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 50, size=n)
    eps_X = rng.normal(0, 1, size=n)
    eps_Z = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)
    X = 0.5 * U + 2 * eps_X
    Z = 1.5 * X + 1.0 * U + 2 * eps_Z
    Y = 1.8 * Z + 1.0 * U + 2 * eps_Y
    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

def scm_extended_bow_nonlinear(n=10000, seed=0):
    rng = np.random.RandomState(seed)

    U = rng.uniform(0, 50, size=n)

    eps_X = rng.normal(0, 1, size=n)
    eps_Z = rng.normal(0, 1, size=n)
    eps_Y = rng.normal(0, 1, size=n)

    X = 0.01 * (U**2) + 0.5 * U + 2 * eps_X

    Z = 0.02 * (X**2) + 1.5 * X + 1.0 * U + 2 * eps_Z

    Y = 0.02 * (Z**2) + 1.8 * Z + 1.0 * U + 2 * eps_Y

    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

experiments = [
    ("backdoor", "linear", scm_backdoor_linear, "graphs/backdoor.cg"),
    ("backdoor", "nonlinear", scm_backdoor_nonlinear, "graphs/backdoor.cg"),

    ("frontdoor", "linear", scm_frontdoor_linear, "graphs/frontdoor.cg"),
    ("frontdoor", "nonlinear", scm_frontdoor_nonlinear, "graphs/frontdoor.cg"),

    ("bow", "linear", scm_bow_linear, "graphs/bow.cg"),
    ("bow", "nonlinear", scm_bow_nonlinear, "graphs/bow.cg"),

    ("extended_bow", "linear", scm_extended_bow_linear, "graphs/extended_bow.cg"),
    ("extended_bow", "nonlinear", scm_extended_bow_nonlinear, "graphs/extended_bow.cg"),
]

# Intervention values
x_values_list = [[0], [-1, 0, 1], [-2, -1, 0, 1, 2]]

for graph_name, mode, gen_fn, graph_path in experiments:

    for i, x_values in enumerate(x_values_list):
        if graph_name == "backdoor" and i==2: continue
        if graph_name == "frontdoor" and i==2: continue

        for seed in seeds:

            print(f"\n[{graph_name.upper()} | {mode} | seed={seed} | x={x_values}]")

            T.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


            data = gen_fn(100000, seed)

            data = (data - data.mean()) / data.std()

            min_model = NCM(graph_path, hidden_dim=64)
            max_model = NCM(graph_path, hidden_dim=64)

            opt_min = T.optim.Adam(min_model.parameters(), lr=5e-4)
            opt_max = T.optim.Adam(max_model.parameters(), lr=5e-4)

            gap_hist = train_minmax_models(
                min_model=min_model,
                max_model=max_model,
                optim_min=opt_min,
                optim_max=opt_max,
                data=data,
                epoch_num=500,
                batch_size=1024,
                m=300,
                treatment_var='X',
                outcome_var='Y',
                x_values=x_values
            )

            T.save(
                min_model.state_dict(),
                f"{base_path}/checkpoints/{graph_name}/{mode}/min_{seed}_{i}.pt"
            )

            T.save(
                max_model.state_dict(),
                f"{base_path}/checkpoints/{graph_name}/{mode}/max_{seed}_{i}.pt"
            )

            np.save(
                f"{base_path}/results/{graph_name}/{mode}/gap_{seed}_{i}.npy",
                np.array(gap_hist, dtype=np.float32)
            )

            plt.figure(figsize=(8,4))
            plt.plot(gap_hist)

            plt.xlabel("Epoch")
            plt.ylabel("|Q_max - Q_min|")
            plt.title(f"{graph_name} ({mode}) | seed={seed} | x={x_values}")
            plt.grid(True)

            plt.savefig(
                f"{base_path}/results/{graph_name}/{mode}/gap_history_{seed}_{i}.svg",
                format="svg"
            )
            plt.close()