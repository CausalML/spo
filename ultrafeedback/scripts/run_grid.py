from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


def parse_list(value: str) -> List[str]:
    return [v for v in value.split(",") if v]


def run_cmd(cmd: List[str], env: dict, timeout: int):
    return subprocess.run(cmd, env=env, timeout=timeout, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--link_shifts", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--link_temperature", type=float, default=1.0)
    parser.add_argument("--n_list", default="500,2000,5000")
    parser.add_argument("--methods", default="dpo,pspo,ospo,rspo")
    parser.add_argument("--beta_grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--base_model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--ref_model_id", default=None)
    parser.add_argument("--reward_model_id", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    parser.add_argument("--dataset_id", default="openbmb/UltraFeedback")
    parser.add_argument("--pref_dir", default="data/preferences")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=36000)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--eval_max_new_tokens", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    args = parser.parse_args()

    seeds = [int(s) for s in parse_list(args.seeds)]
    shifts = [float(s) for s in parse_list(args.link_shifts)]
    n_list = [int(n) for n in parse_list(args.n_list)]
    methods = parse_list(args.methods)
    gpus = parse_list(args.gpus)

    os.makedirs(args.pref_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    tasks = []
    for seed in seeds:
        for shift in shifts:
            for n in n_list:
                pref_path = os.path.join(args.pref_dir, f"prefs_s{seed}_shift{shift}_n{n}.jsonl")
                for method in methods:
                    tasks.append((seed, shift, n, method, pref_path))

    def worker(task, gpu_id: str):
        seed, shift, n, method, pref_path = task
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        if not os.path.exists(pref_path):
            lock_path = pref_path + ".lock"
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                # Another worker is generating it; wait for completion.
                while not os.path.exists(pref_path):
                    time.sleep(1)
            else:
                model_id = args.ref_model_id or args.base_model_id
                cmd = [
                    sys.executable,
                    "scripts/generate_preferences.py",
                    "--dataset_id",
                    args.dataset_id,
                    "--num_samples",
                    str(n),
                    "--seed",
                    str(seed),
                    "--model_id",
                    model_id,
                    "--reward_model_id",
                    args.reward_model_id,
                    "--output",
                    pref_path,
                    "--link_shift",
                    str(shift),
                    "--link_temperature",
                    str(args.link_temperature),
                    "--max_new_tokens",
                    str(args.gen_max_new_tokens),
                    "--batch_size",
                    str(args.gen_batch_size),
                ]
                run_cmd(cmd, env=env, timeout=args.timeout)
                os.remove(lock_path)

        run_id = f"{method}_s{seed}_shift{shift}_n{n}"
        output_dir = os.path.join(args.output_dir, run_id)
        cmd = [
            sys.executable,
            "scripts/train_policy.py",
            "--pref_path",
            pref_path,
            "--method",
            method,
            "--base_model_id",
            args.base_model_id,
            "--output_dir",
            output_dir,
            "--max_steps",
            str(args.max_steps),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--max_prompt_length",
            str(args.max_prompt_length),
            "--max_response_length",
            str(args.max_response_length),
        ]
        if args.ref_model_id:
            cmd.extend(["--ref_model_id", args.ref_model_id])
        run_cmd(cmd, env=env, timeout=args.timeout)

        model_id = os.path.join(output_dir, f"{method}_final")
        eval_path = os.path.join(args.eval_dir, f"eval_{run_id}.csv")
        cmd = [
            sys.executable,
            "scripts/evaluate_policy.py",
            "--model_id",
            model_id,
            "--ref_model_id",
            args.ref_model_id or args.base_model_id,
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
            str(args.link_temperature),
            "--num_preferences",
            str(n),
        ]
        run_cmd(cmd, env=env, timeout=args.timeout)
        return eval_path

    futures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for i, task in enumerate(tasks):
            gpu = gpus[i % len(gpus)]
            futures.append(executor.submit(worker, task, gpu))

        for fut in as_completed(futures):
            try:
                path = fut.result()
                print(f"completed: {path}")
            except Exception as exc:
                print(f"task failed: {exc}")


if __name__ == "__main__":
    main()
