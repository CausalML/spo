import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spo_sim.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n_grid", type=str, default="500,1000,2000")
    parser.add_argument("--s_grid", type=str, default="0.0,1.0,1.5")
    parser.add_argument("--tau_grid", type=str, default="0.25")
    parser.add_argument("--methods", type=str, default="dpo,pspo,ospo,rspo")
    parser.add_argument("--out_csv", type=str, default="outputs/results.csv")
    parser.add_argument("--m", type=int, default=2000)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nu", type=float, default=10.0)
    parser.add_argument("--hidden", type=str, default="32,32")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--beta_grid", type=str, default="0.2,0.3,0.5,0.8,1,1.5,2,3,5,8,10,15,20,30,50,80,100,150,200,300,500,800,1000")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ospo_mem", type=int, default=0)
    parser.add_argument("--ospo_bandwidth", type=float, default=1.0)
    parser.add_argument("--ospo_no_detach", action="store_true")
    parser.add_argument("--ospo_warm", type=int, default=2)
    parser.add_argument("--ospo_kernel", type=str, default="gaussian")
    parser.add_argument("--ospo_l2", type=float, default=0.0)
    parser.add_argument("--ospo_g_hat", type=str, default="kernel")
    parser.add_argument("--ospo_g_hat_lr", type=float, default=1e-2)
    parser.add_argument("--ospo_g_hat_steps", type=int, default=5)
    parser.add_argument("--pspo_outer", type=int, default=4)
    parser.add_argument("--pspo_inner", type=int, default=2)
    parser.add_argument("--pspo_warm", type=int, default=3)
    parser.add_argument("--max_time", type=float, default=600.0)
    parser.add_argument("--n_jobs", type=int, default=4)
    return parser.parse_args()


def parse_list(arg: str, cast=float) -> List:
    return [cast(x) for x in arg.split(",") if x]


def run_cmd(cmd: List[str], timeout: float) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def main() -> None:
    args = parse_args()

    seeds = parse_list(args.seeds, int)
    n_grid = parse_list(args.n_grid, int)
    s_grid = parse_list(args.s_grid, float)
    tau_grid = parse_list(args.tau_grid, float)
    methods = parse_list(args.methods, str)

    ensure_dir(os.path.dirname(args.out_csv))

    jobs = []
    temp_dir = os.path.join(os.path.dirname(args.out_csv), f"tmp_runs_{int(time.time())}")
    ensure_dir(temp_dir)
    temp_paths = []
    for seed, n, s, tau, method in itertools.product(seeds, n_grid, s_grid, tau_grid, methods):
        temp_path = os.path.join(temp_dir, f"run_{len(temp_paths)}.csv")
        temp_paths.append(temp_path)
        cmd = [
            "python",
            "scripts/run_single.py",
            "--seed",
            str(seed),
            "--n",
            str(n),
            "--m",
            str(args.m),
            "--d",
            str(args.d),
            "--k",
            str(args.k),
            "--nu",
            str(args.nu),
            "--s",
            str(s),
            "--tau",
            str(tau),
            "--method",
            str(method),
            "--hidden",
            args.hidden,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--beta_grid",
            args.beta_grid,
            "--device",
            str(args.device) if args.device is not None else "cpu",
            "--ospo_mem",
            str(args.ospo_mem),
            "--ospo_bandwidth",
            str(args.ospo_bandwidth),
            "--ospo_warm",
            str(args.ospo_warm),
            "--ospo_kernel",
            str(args.ospo_kernel),
            "--ospo_l2",
            str(args.ospo_l2),
            "--ospo_g_hat",
            str(args.ospo_g_hat),
            "--ospo_g_hat_lr",
            str(args.ospo_g_hat_lr),
            "--ospo_g_hat_steps",
            str(args.ospo_g_hat_steps),
            "--pspo_outer",
            str(args.pspo_outer),
            "--pspo_inner",
            str(args.pspo_inner),
            "--pspo_warm",
            str(args.pspo_warm),
            "--out_csv",
            temp_path,
            "--max_time",
            str(args.max_time),
        ]
        if args.ospo_no_detach:
            cmd.append("--ospo_no_detach")
        jobs.append(cmd)

    with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = [ex.submit(run_cmd, cmd, args.max_time) for cmd in jobs]
        for fut in as_completed(futures):
            fut.result()

    frames = []
    for path in temp_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            frames.append(pd.read_csv(path))
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
