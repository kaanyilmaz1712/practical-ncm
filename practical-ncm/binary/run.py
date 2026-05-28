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

m = 200
lr = 4e-3
epoch_num = 500
n = 3000

DATA_AMOUNTS = [12800, 6400, 3200, 1600, 800, 400, 200, 100, 50]

def set_seed(seed):
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_experiment(name, graph_path, data_fn):
    print(f"\n===== {name.upper()} =====")

    os.makedirs(f"checkpoints/{name}", exist_ok=True)
    os.makedirs(f"results/{name}", exist_ok=True)

    for seed in SEEDS:
        print(f"[{name}] seed={seed}")

        set_seed(seed)

        data = data_fn(n_samples=n, seed=seed)

        # Models
        min_model = NCM(graph_path)
        max_model = NCM(graph_path)

        optim_min = T.optim.Adam(min_model.parameters(), lr=lr)
        optim_max = T.optim.Adam(max_model.parameters(), lr=lr)

        # Training
        gap_hist = train_minmax_models(
            epoch_num, m,
            data,
            min_model,
            max_model,
            optim_min,
            optim_max
        )

        # Save
        T.save(min_model.state_dict(), f"checkpoints/{name}/min_model_{seed}.pt")
        T.save(max_model.state_dict(), f"checkpoints/{name}/max_model_{seed}.pt")

        np.save(f"results/{name}/gap_{seed}.npy", np.array(gap_hist))

        # Plot
        plt.figure(figsize=(8,4))
        plt.plot(gap_hist, label="GapATE = |ATE_max - ATE_min|")
        plt.xlabel("Epoch")
        plt.ylabel("ΔATE")
        plt.title(f"{name} (seed={seed})")
        plt.grid(True)
        plt.legend()

        plt.savefig(f"results/{name}/gap_history_{seed}.svg", format="svg")
        plt.close()

def run_sample_size_experiment(name, graph_path, data_fn, data_amounts):

    print(f"\n===== {name.upper()} (SAMPLE SIZE TEST) =====")

    base_path = f"{name}-sample-size"

    os.makedirs(f"{base_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_path}/results", exist_ok=True)

    for seed in SEEDS:
        for n in data_amounts:

            print(f"[{name}] n={n}, seed={seed}")

            set_seed(seed)

            data = data_fn(n_samples=n, seed=seed)

            # Models
            min_model = NCM(graph_path)
            max_model = NCM(graph_path)

            optim_min = T.optim.Adam(min_model.parameters(), lr=lr)
            optim_max = T.optim.Adam(max_model.parameters(), lr=lr)

            # Train
            gap_hist = train_minmax_models(
                epoch_num, m,
                data,
                min_model,
                max_model,
                optim_min,
                optim_max
            )

            # Save models
            T.save(
                min_model.state_dict(),
                f"{base_path}/checkpoints/min_model_{n}_{seed}.pt"
            )

            T.save(
                max_model.state_dict(),
                f"{base_path}/checkpoints/max_model_{n}_{seed}.pt"
            )

            # Save gap
            np.save(
                f"{base_path}/results/gap_{n}_{seed}.npy",
                np.array(gap_hist, dtype=np.float32)
            )

            # Plot
            plt.figure(figsize=(8,4))
            plt.plot(gap_hist, label="GapATE = |ATE_max - ATE_min|")

            plt.xlabel("Epoch")
            plt.ylabel("ΔATE")
            plt.title(f"{name} Sample Size (n={n}, seed={seed})")

            plt.grid(True)
            plt.legend()

            plt.savefig(
                f"{base_path}/results/gap_history_{n}_{seed}.svg",
                format="svg"
            )
            plt.close()


def backdoor_data(n_samples, seed):
    rng = np.random.RandomState(seed)
    
    Z = rng.binomial(1, 0.5, size=n_samples)
    
    X = np.array([rng.rand() < (0.7 if z == 1 else 0.3) for z in Z]).astype(int)

    Y = np.array([rng.rand() < (0.9 if (x == 1 and z == 1)
                                   else 0.6 if (x == 1 and z == 0)
                                   else 0.3 if (x == 0 and z == 1) 
                                   else 0.1
                                   )
                    for x,z in zip(X,Z)], dtype=int)

    return pd.DataFrame(dict(Z=Z, X=X, Y=Y))


def frontdoor_data(n_samples, seed):
    rng = np.random.RandomState(seed)

    U = rng.binomial(1, 0.5, size=n_samples)

    X = np.array([rng.rand() < (0.7 if u else 0.3) for u in U]).astype(int)

    Z = np.array([rng.rand() < (0.8 if x else 0.2) for x in X]).astype(int)

    Y = np.array([rng.rand() < (0.9 if (z == 1 and u == 1)
                                   else 0.6 if (z == 1 and u == 0)
                                   else 0.3 if (z == 0 and u == 1)
                                   else 0.1)
                  for z, u in zip(Z, U)]).astype(int)

    return pd.DataFrame(dict(X=X, Z=Z, Y=Y))


def m_data(n_samples, seed):
    rng = np.random.RandomState(seed)

    U_XZ = rng.binomial(1, 0.5, size=n_samples)
    U_ZY = rng.binomial(1, 0.5, size=n_samples)

    X = np.array([rng.rand() < (0.7 if u == 1 else 0.3) for u in U_XZ]).astype(int)

    Z = np.array([
        rng.rand() < (0.8 if (u1 == 1 or u2 == 1) else 0.2)
        for u1, u2 in zip(U_XZ, U_ZY)
    ]).astype(int)

    Y = np.array([
        rng.rand() < (
            0.9 if (x == 1 and u == 1)
            else 0.6 if (x == 1 and u == 0)
            else 0.3 if (x == 0 and u == 1)
            else 0.1
        )
        for x, u in zip(X, U_ZY)
    ]).astype(int)

    return pd.DataFrame(dict(X=X, Z=Z, Y=Y))


def napkin_data(n_samples, seed=0):
    rng = np.random.RandomState(seed)

    U_WX = rng.binomial(1, 0.5, size=n_samples)
    U_WY = rng.binomial(1, 0.5, size=n_samples)

    W = np.array([
        rng.rand() < (0.8 if (u1 == 1 or u2 == 1) else 0.2)
        for u1, u2 in zip(U_WX, U_WY)
    ]).astype(int)

    Z = np.array([
        rng.rand() < (0.75 if w == 1 else 0.25)
        for w in W
    ]).astype(int)

    X = np.array([
        rng.rand() < (0.85 if (z == 1 or u == 1) else 0.15)
        for z, u in zip(Z, U_WX)
    ]).astype(int)

    Y = np.array([
        rng.rand() < (
            0.9 if (x == 1 and u == 1)
            else 0.6 if (x == 1 and u == 0)
            else 0.4 if (x == 0 and u == 1)
            else 0.1
        )
        for x, u in zip(X, U_WY)
    ]).astype(int)

    return pd.DataFrame(dict(W=W, Z=Z, X=X, Y=Y))

def bow_data(n_samples, seed):
    rng = np.random.RandomState(seed)

    U = rng.binomial(1, 0.5, size=n_samples)

    X = np.array([
        rng.rand() < (0.7 if u else 0.3)
        for u in U
    ]).astype(int)

    Y = np.array([
        rng.rand() < (
            0.9 if (x == 1 and u == 1)
            else 0.7 if (x == 1 and u == 0)
            else 0.4 if (x == 0 and u == 1)
            else 0.1
        )
        for x, u in zip(X, U)
    ]).astype(int)

    return pd.DataFrame(dict(X=X, Y=Y))

def extendedbow_data(n_samples, seed=0):
    rng = np.random.RandomState(seed)

    U = rng.binomial(1, 0.5, size=n_samples)

    X = np.array([
        rng.rand() < (0.7 if u else 0.3)
        for u in U
    ]).astype(int)

    Z = np.array([
        rng.rand() < (0.8 if (x == 1 or u == 1) else 0.2)
        for x, u in zip(X, U)
    ]).astype(int)

    Y = np.array([
        rng.rand() < (0.85 if z == 1 else 0.25)
        for z in Z
    ]).astype(int)

    return pd.DataFrame(dict(X=X, Z=Z, Y=Y))

def iv_data(n_samples, seed):
    rng = np.random.RandomState(seed)

    U = rng.binomial(1, 0.5, size=n_samples)
    Z = rng.binomial(1, 0.5, size=n_samples)

    X = np.array([
        rng.rand() < (0.8 if z else 0.3) if u == 0
        else rng.rand() < (0.6 if z else 0.2)
        for z, u in zip(Z, U)
    ]).astype(int)

    Y = np.array([
        rng.rand() < (
            0.9 if (x == 1 and u == 1)
            else 0.6 if (x == 1 and u == 0)
            else 0.4 if (x == 0 and u == 1)
            else 0.1
        )
        for x, u in zip(X, U)
    ]).astype(int)

    return pd.DataFrame(dict(Z=Z, X=X, Y=Y))


def badm_data(n_samples, seed=0):
    rng = np.random.RandomState(seed)

    U_ZX = rng.binomial(1, 0.5, size=n_samples)
    U_ZY = rng.binomial(1, 0.5, size=n_samples)

    Z = np.array([
        rng.rand() < (0.8 if (u1 == 1 and u2 == 1) else 0.2)
        for u1, u2 in zip(U_ZX, U_ZY)
    ]).astype(int)

    X = np.array([
        rng.rand() < (0.9 if (z == 1 and u == 1) else 0.1)
        for z, u in zip(Z, U_ZX)
    ]).astype(int)

    Y = np.array([
        rng.rand() < (
            0.9 if (x == 1 and z == 0 and u == 1) else
            0.7 if (x == 1 and z == 1 and u == 0) else
            0.2
        )
        for x, z, u in zip(X, Z, U_ZY)
    ]).astype(int)

    return pd.DataFrame(dict(Z=Z, X=X, Y=Y))

if __name__ == "__main__":

    # Normal experiments
    run_experiment("backdoor", "./graphs/backdoor.cg", backdoor_data)
    run_experiment("frontdoor", "./graphs/frontdoor.cg", frontdoor_data)
    run_experiment("m", "./graphs/m.cg", m_data)
    run_experiment("napkin", "./graphs/napkin.cg", napkin_data)
    run_experiment("bow", "./graphs/bow.cg", bow_data)
    run_experiment("extendedbow", "./graphs/extended_bow.cg", extendedbow_data)
    run_experiment("iv", "./graphs/iv.cg", iv_data)
    run_experiment("badm", "./graphs/bad_m.cg", badm_data)


    # Sample size test
    run_sample_size_experiment(
        "frontdoor",
        "./graphs/frontdoor.cg",
        frontdoor_data,
        DATA_AMOUNTS
    )