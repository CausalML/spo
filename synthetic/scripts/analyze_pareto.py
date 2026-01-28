import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spo_sim.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="outputs/results.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/plots")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--s", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    ensure_dir(args.out_dir)

    pareto_n = args.n or df["n"].max()
    pareto_s = args.s if args.s is not None else df["s"].unique()[0]
    pareto_tau = args.tau if args.tau is not None else df["tau"].unique()[0]

    df_pareto = df[(df["n"] == pareto_n) & (df["s"] == pareto_s) & (df["tau"] == pareto_tau)]
    pareto = (
        df_pareto.groupby(["method", "beta"], as_index=False)
        .agg(avg_reward=("avg_reward", "mean"), avg_kl=("avg_kl", "mean"))
        .sort_values(["method", "avg_kl"])
    )
    pareto.to_csv(os.path.join(args.out_dir, "pareto_table.csv"), index=False)

    plt.figure(figsize=(4, 4))
    for method, sub in pareto.groupby("method"):
        plt.plot(sub["avg_kl"], sub["avg_reward"], marker="o", label=method)
    plt.xlabel("Average KL")
    plt.ylabel("Average reward")
    # plt.title(f"Pareto curve (n={pareto_n}, s={pareto_s}, tau={pareto_tau})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"pareto_curve_{pareto_s}.pdf"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
