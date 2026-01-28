import argparse
import csv
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spo_sim.data import DGPConfig, generate_teacher, sample_dataset
from spo_sim.eval import evaluate_policy
from spo_sim.methods import OSPOConfig, PSPOConfig, RSPOConfig, train_dpo, train_ospo, train_pspo, train_rspo
from spo_sim.models import MLPPolicy
from spo_sim.utils import TrainConfig, ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, default=2000)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nu", type=float, default=10.0)
    parser.add_argument("--s", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--hidden", type=str, default="32,32")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--beta_grid", type=str, default="0.2,0.3,0.5,0.8,1,1.5,2,3,5,8,10,15,20,30,50,80,100,150,200,300,500,800,1000")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_time", type=float, default=600.0)
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
    parser.add_argument("--rspo_batch_square", action="store_true")
    return parser.parse_args()


def parse_list(arg: str) -> List[float]:
    return [float(x) for x in arg.split(",") if x]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or get_device(prefer_cuda=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    hidden_sizes = tuple(int(x) for x in args.hidden.split(",") if x)
    dgp = DGPConfig(d=args.d, k=args.k, nu=args.nu, s=args.s, tau=args.tau)
    teacher = generate_teacher(args.seed, args.d, args.k, hidden_sizes).to(device)

    data = sample_dataset(args.seed, args.n, args.m, dgp, teacher, device=device)

    model = MLPPolicy(args.d, args.k, hidden_sizes).to(device)

    train_cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, device=device)

    start_time = time.time()
    zeta = 1.0
    if args.method.lower() == "dpo":
        model = train_dpo(model, data, train_cfg)
    elif args.method.lower() == "pspo":
        pspo_cfg = PSPOConfig(outer_iters=args.pspo_outer, inner_epochs=args.pspo_inner, warm_start_epochs=args.pspo_warm)
        model = train_pspo(model, data, train_cfg, pspo_cfg)
    elif args.method.lower() == "ospo":
        ospo_cfg = OSPOConfig(
            mem_size=args.ospo_mem,
            bandwidth=args.ospo_bandwidth,
            detach_memory=not args.ospo_no_detach,
            warm_start_epochs=args.ospo_warm,
            kernel=args.ospo_kernel,
            l2_reg=args.ospo_l2,
            g_hat=args.ospo_g_hat,
            g_hat_lr=args.ospo_g_hat_lr,
            g_hat_steps=args.ospo_g_hat_steps,
        )
        model = train_ospo(model, data, train_cfg, ospo_cfg)
        with torch.no_grad():
            x_t = torch.from_numpy(data["x"]).to(device)
            y0 = torch.from_numpy(data["y0"]).to(device)
            y1 = torch.from_numpy(data["y1"]).to(device)
            z = torch.from_numpy(data["z"]).to(device).float()
            log_pi = model.log_probs(x_t)
            log_pi_ref = torch.full((args.k,), -torch.log(torch.tensor(float(args.k), device=device)), device=device)
            t = (log_pi.gather(1, y1.unsqueeze(1)).squeeze(1) - log_pi_ref[y1]) - (
                log_pi.gather(1, y0.unsqueeze(1)).squeeze(1) - log_pi_ref[y0]
            )
            cov = torch.mean((t - t.mean()) * (z - z.mean())).item()
            zeta = 1.0 if cov >= 0 else -1.0
    elif args.method.lower() == "rspo":
        rspo_cfg = RSPOConfig(batch_square=args.rspo_batch_square)
        model = train_rspo(model, data, train_cfg, rspo_cfg)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    if time.time() - start_time > args.max_time:
        raise TimeoutError("Training exceeded max_time")

    beta_grid = parse_list(args.beta_grid)
    rewards, kls = evaluate_policy(model, teacher, data["x_eval"], dgp, beta_grid, zeta=zeta, device=device)

    ensure_dir(os.path.dirname(args.out_csv))
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["seed", "n", "s", "tau", "method", "beta", "avg_reward", "avg_kl"])
        for beta, reward, kl in zip(beta_grid, rewards, kls):
            writer.writerow([args.seed, args.n, args.s, args.tau, args.method, beta, reward, kl])


if __name__ == "__main__":
    main()
