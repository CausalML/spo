from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_glob", default="outputs/eval/*.csv")
    parser.add_argument("--out_dir", default="outputs/plots")
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = glob.glob(args.eval_glob)
    if not files:
        raise FileNotFoundError("No evaluation CSVs found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Extract metadata from CSV if present, else parse from model_id path
    if "method" not in df.columns or df["method"].isna().all():
        df["run_id"] = df["model_id"].apply(lambda x: os.path.basename(os.path.dirname(x)))
        df["method"] = df["run_id"].str.split("_").str[0]
        df["seed"] = df["run_id"].str.extract(r"_s(\d+)").astype(float)
        df["shift"] = df["run_id"].str.extract(r"_shift([0-9.\-]+)").astype(float)
        df["n"] = df["run_id"].str.extract(r"_n(\d+)").astype(float)
    else:
        df = df.rename(columns={"link_shift": "shift", "num_preferences": "n"})

    n = args.n or int(df["n"].max())
    df = df[df["n"] == n]
    
    kappa = args.kappa
    def pick_beta(group):
        group = group.sort_values("kl_div")
        below = group[group["kl_div"] <= kappa]
        above = group[group["kl_div"] >= kappa]

        if not below.empty and not above.empty:
            lo = below.iloc[-1]
            hi = above.iloc[0]
            if hi["kl_div"] == lo["kl_div"]:
                reward = float(lo["avg_reward"])
            else:
                t = (kappa - float(lo["kl_div"])) / (float(hi["kl_div"]) - float(lo["kl_div"]))
                reward = float(lo["avg_reward"]) + t * (float(hi["avg_reward"]) - float(lo["avg_reward"]))
            row = lo.to_dict()
            row["avg_reward"] = reward
            row["kl_div"] = float(kappa)
            row["beta"] = np.nan
            return pd.Series(row)

        idx = (group["kl_div"] - kappa).abs().idxmin()
        row = group.loc[idx].to_dict()
        row["kl_div"] = float(kappa)
        row["beta"] = np.nan
        return pd.Series(row)

    rows = []
    for (shift, method, seed, n), group in df.groupby(["shift", "method", "seed", "n"]):
        rows.append(pick_beta(group))
    conv_df = pd.DataFrame(rows)

    summary = conv_df.groupby(["shift", "method", "n"]).agg(
        mean_reward=("avg_reward", "mean"),
        std_reward=("avg_reward", "std"),
        count=("avg_reward", "count"),
    ).reset_index()
    summary["ci95"] = 1.96 * summary["std_reward"] / np.sqrt(summary["count"].clip(lower=1))

    plt.figure(figsize=(4, 2.5))
    for method, group in summary.groupby("method"):
        plt.errorbar(group["shift"], group["mean_reward"], marker="o", label=method)
    plt.xlabel("Logistic link shift s")
    plt.ylabel("Reward at KL≈{:.2f}".format(kappa))
    # plt.title(f"Reward vs shift (n={int(n)})")
    plt.legend()
    plt.tight_layout()
    shift_path = os.path.join(args.out_dir, f"reward_vs_shift_n{int(n)}.pdf")
    plt.savefig(shift_path, dpi=200)
    plt.close()

    summary_path = os.path.join(args.out_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
