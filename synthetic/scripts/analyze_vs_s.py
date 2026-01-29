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
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/plots")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--kappa", type=float, required=True)
    return parser.parse_args()


def interp_reward(group: pd.DataFrame, kappa: float) -> float:
    group = group.sort_values("avg_kl")
    kl = group["avg_kl"].values
    rw = group["avg_reward"].values
    if kappa <= kl.min():
        return rw[0]
    if kappa >= kl.max():
        return rw[-1]
    hi = (kl >= kappa).argmax()
    lo = max(hi - 1, 0)
    if kl[hi] == kl[lo]:
        return rw[hi]
    w = (kappa - kl[lo]) / (kl[hi] - kl[lo])
    return rw[lo] + w * (rw[hi] - rw[lo])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    args.n = args.n if args.n is not None else df["n"].unique()[0]
    args.tau = args.tau if args.tau is not None else df["tau"].unique()[0]
    df = df[(df["n"] == args.n) & (df["tau"] == args.tau)]

    rows = []
    for (seed, s, method), group in df.groupby(["seed", "s", "method"]):
        reward = interp_reward(group, args.kappa)
        rows.append({"seed": seed, "s": s, "method": method, "reward": reward})
    res = pd.DataFrame(rows)

    summary = (
        res.groupby(["s", "method"], as_index=False)
        .agg(mean_reward=("reward", "mean"),
             # low=("reward", lambda x: np.quantile(x, 0.05)),
             # high=("reward", lambda x: np.quantile(x, 0.95)))
             low=("reward", lambda x: np.mean(x)-1.64*np.std(x)/np.sqrt(len(x))),
             high=("reward", lambda x: np.mean(x)+1.64*np.std(x)/np.sqrt(len(x))))
        .sort_values(["s", "method"])
    )

    ensure_dir(args.out_dir)
    summary.to_csv(os.path.join(args.out_dir, "reward_vs_s_table.csv"), index=False)

    plt.figure(figsize=(4, 2.5))
    for method, sub in summary.groupby("method"):
        plt.plot(sub["s"], sub["mean_reward"], marker="o", label=method)
        plt.fill_between(sub["s"], sub["low"], sub["high"], alpha=0.2)
    plt.xlabel("Logistic link shift s")
    plt.ylabel(f"Reward at KL≈{args.kappa:.2f}")
    # plt.title(f"Reward vs misspecification (n={args.n}, tau={args.tau})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "reward_vs_s.pdf"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
