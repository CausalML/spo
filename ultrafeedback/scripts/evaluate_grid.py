from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


RUN_RE = re.compile(r"^(?P<method>[^_]+)_s(?P<seed>\d+)_shift(?P<shift>[0-9.\-]+)_n(?P<n>\d+)$")


def parse_list(value: str) -> List[str]:
    return [v for v in value.split(",") if v]


def parse_run_id(path: str) -> Tuple[str, int, float, int] | None:
    name = os.path.basename(path.rstrip("/"))
    match = RUN_RE.match(name)
    if not match:
        return None
    return (
        match.group("method"),
        int(match.group("seed")),
        float(match.group("shift")),
        int(match.group("n")),
    )


def run_cmd(cmd: List[str], env: dict, timeout: int):
    return subprocess.run(cmd, env=env, timeout=timeout, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", default="outputs/*_s*_shift*_n*")
    parser.add_argument("--eval_dir", default="outputs/eval_v4")
    parser.add_argument("--beta_grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--base_model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--reward_model_id", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    parser.add_argument("--dataset_id", default="openbmb/UltraFeedback")
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--eval_max_new_tokens", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--link_shifts", default=None)
    parser.add_argument("--n_list", default=None)
    parser.add_argument("--methods", default=None)
    args = parser.parse_args()

    os.makedirs(args.eval_dir, exist_ok=True)
    gpus = parse_list(args.gpus)

    seed_filter = None
    if args.seeds:
        seed_filter = {int(s) for s in parse_list(args.seeds)}
    shift_filter = None
    if args.link_shifts:
        shift_filter = {float(s) for s in parse_list(args.link_shifts)}
    n_filter = None
    if args.n_list:
        n_filter = {int(s) for s in parse_list(args.n_list)}
    method_filter = None
    if args.methods:
        method_filter = set(parse_list(args.methods))

    runs = []
    for path in sorted(glob.glob(args.runs_glob)):
        parsed = parse_run_id(path)
        if not parsed:
            continue
        method, seed, shift, n = parsed
        if seed_filter is not None and seed not in seed_filter:
            continue
        if shift_filter is not None and shift not in shift_filter:
            continue
        if n_filter is not None and n not in n_filter:
            continue
        if method_filter is not None and method not in method_filter:
            continue
        model_id = os.path.join(path, f"{method}_final")
        if not os.path.exists(model_id):
            continue
        eval_path = os.path.join(args.eval_dir, f"eval_{method}_s{seed}_shift{shift}_n{n}.csv")
        if os.path.exists(eval_path):
            continue
        runs.append((method, seed, shift, n, model_id, eval_path))

    if not runs:
        print("No runs to evaluate.")
        return

    def worker(run, gpu_id: str):
        method, seed, shift, n, model_id, eval_path = run
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            sys.executable,
            "scripts/evaluate_policy.py",
            "--model_id",
            model_id,
            "--ref_model_id",
            args.base_model_id,
            "--reward_model_id",
            args.reward_model_id,
            "--dataset_id",
            args.dataset_id,
            "--beta_grid",
            args.beta_grid,
            "--output",
            eval_path,
            "--seed",
            str(seed),
            "--eval_samples",
            str(args.eval_samples),
            "--max_new_tokens",
            str(args.eval_max_new_tokens),
            "--batch_size",
            str(args.eval_batch_size),
            "--method",
            method,
            "--seed_id",
            str(seed),
            "--link_shift",
            str(shift),
            "--link_temperature",
            "1.0",
            "--num_preferences",
            str(n),
        ]
        run_cmd(cmd, env=env, timeout=args.timeout)
        return eval_path

    futures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for i, run in enumerate(runs):
            futures.append(executor.submit(worker, run, gpus[i % len(gpus)]))
        for fut in as_completed(futures):
            try:
                path = fut.result()
                print(f"completed: {path}")
            except Exception as exc:
                print(f"task failed: {exc}")


if __name__ == "__main__":
    main()
