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
    # summary["ci90"] = 1.64 * summary["std_reward"] / np.sqrt(summary["count"].clip(lower=1))

    dpo = conv_df[conv_df["method"] == "dpo"][["shift", "n", "seed", "avg_reward"]].rename(
        columns={"avg_reward": "dpo_reward"}
    )
    merged = conv_df.merge(dpo, on=["shift", "n", "seed"], how="left")
    merged["diff_to_dpo"] = merged["avg_reward"] - merged["dpo_reward"]
    diff_summary = merged.groupby(["shift", "method", "n"]).agg(
        diff_mean=("diff_to_dpo", "mean"),
        diff_std=("diff_to_dpo", "std"),
        diff_count=("diff_to_dpo", "count"),
    ).reset_index()
    diff_summary["diff_ci90"] = 1.64 * diff_summary["diff_std"] / np.sqrt(
        diff_summary["diff_count"].clip(lower=1)
    )

    summary = summary.merge(diff_summary[["shift", "method", "n", "diff_ci90"]], on=["shift", "method", "n"], how="left")

    methods = sorted(summary["method"].unique())
    offsets = np.linspace(-0.03, 0.03, len(methods))
    jitter = dict(zip(methods, offsets))

    plt.figure(figsize=(5, 3))
    for method, group in summary.groupby("method"):
        plt.errorbar(group["shift"].to_numpy(dtype=float) + jitter.get(method, 0.0), group["mean_reward"], yerr=group["diff_ci90"], marker="o", label=method)
    plt.xlabel("Logistic link shift s")
    plt.ylabel("Reward at KL≈{:.2f}".format(kappa))
    # plt.title(f"Reward vs shift (n={int(n)})")
    plt.legend(loc="lower left")
    plt.tight_layout()
    shift_path = os.path.join(args.out_dir, f"reward_vs_shift_n{int(n)}.pdf")
    plt.savefig(shift_path, dpi=200)
    plt.close()

    summary_path = os.path.join(args.out_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
