import os, argparse, yaml
from pathlib import Path
import torch
from accelerate import Accelerator
from itertools import product

from src.utils.io_helpers import ensure_dir, append_csv
from src.utils.calibration import calibrate_beta_for_kappa
from src.utils.metrics import expected_true_reward, kl_to_uniform
from src.models.mlp_policy import MLPPolicy
from src.trainers.sft_trainer import SFTTrainer
from src.trainers.dpo_trainer import DPOTrainer
from src.trainers.pspo_trainer import PSPOTrainer
from src.trainers.espo_trainer import ESPOTrainer
from src.trainers.smsspo_trainer import SMSSPOTrainer
from src.data.simulate_pairs import generate_dataset
from src.data.utils import set_seed

ALGOS = ["sft", "dpo", "pspo", "espo", "smsspo"]

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/sim_default.yaml")
    ap.add_argument("--n_runs", type=int, default=3)
    ap.add_argument("--kappa", type=float, default=None, help="override κ")
    ap.add_argument("--train_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    return ap.parse_args()

def main():
    args = get_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.kappa is not None: cfg['kappa'] = args.kappa
    if args.train_steps is not None: cfg['train_steps'] = args.train_steps
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.save_dir is not None: cfg['save_dir'] = args.save_dir

    accelerator = Accelerator()
    device = accelerator.device

    out_dir = Path(cfg['save_dir'])
    ensure_dir(str(out_dir))
    agg_csv_final = out_dir / "aggregate.csv"
    agg_csv_rank = out_dir / f"aggregate_rank{accelerator.process_index}.csv"

    # --- Build all tasks and shard across ranks (task-parallel) ---
    all_tasks = list(product(range(args.n_runs), ALGOS))  # [(run_idx, algo), ...]
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    my_tasks = all_tasks[rank::world_size]

    accelerator.print(f"[rank {rank}] Assigned tasks: {my_tasks}")

    for run_idx, algo in my_tasks:
        seed = cfg['seed'] + run_idx
        set_seed(int(seed))

        accelerator.print(f"\n=== Task (run={run_idx+1}/{args.n_runs}, seed={seed}, algo={algo}) ===")

        data = generate_dataset(
            n_pairs=cfg['n_pairs'],
            n_val_calib=cfg['n_val_calib'],
            n_test=cfg['n_test'],
            n_actions=cfg['n_actions'],
            feat_dim=cfg['feat_dim'],
            link=cfg['link'],
            beta_a=cfg['beta_sigmoid_a'],
            beta_b=cfg['beta_sigmoid_b'],
            seed=seed,
            device=device
        )
        x, y0, y1, z, W = data['train']
        x_calib = data['calib_x']
        x_test = data['test_x']

        policy = MLPPolicy(input_dim=cfg['feat_dim'], n_actions=cfg['n_actions'],
                           hidden_sizes=tuple(cfg['hidden_sizes']),
                           activation=cfg['activation']).to(device)

        if algo == "sft":
            trainer = SFTTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                 wd=cfg['weight_decay'])
        elif algo == "dpo":
            trainer = DPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                 wd=cfg['weight_decay'], beta_train=1.0)
        elif algo == "pspo":
            trainer = PSPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                  wd=cfg['weight_decay'])
        elif algo == "espo":
            trainer = ESPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                  wd=cfg['weight_decay'], h=cfg['h_espo'])
        elif algo == "smsspo":
            trainer = SMSSPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                    wd=cfg['weight_decay'], sigma=cfg['sigma_smsspo'])
        else:
            raise ValueError(algo)

        # If a trainer DDP-wrapped the model, disable cross-rank sync so each GPU trains independently
        try:
            ddp_model = getattr(trainer, "policy", None)
            if hasattr(ddp_model, "require_backward_grad_sync"):
                ddp_model.require_backward_grad_sync = False
        except Exception:
            pass

        accelerator.print(f"-- [{algo.upper()}] training --")
        policy = trainer.train(x, y0, y1, z, steps=cfg['train_steps'], batch_size=cfg['batch_size'])

        # h(x) := log πθ(y|x)  (forward returns log-probs)
        def logits_fn(X):
            return policy(X)

        beta = calibrate_beta_for_kappa(logits_fn, x_calib, kappa=cfg['kappa'])

        with torch.no_grad():
            h_test = policy(x_test)                      # log-probs
            p_test = torch.softmax(h_test / beta, dim=-1)
            gt_reward = expected_true_reward(p_test, x_test, W)
            kl_u = kl_to_uniform(p_test)

        row = {
            'run': run_idx, 'algo': algo, 'seed': seed,
            'beta': float(beta), 'kappa_target': cfg['kappa'],
            'test_reward': float(gt_reward), 'test_kl_uniform': float(kl_u),
            'rank': rank
        }
        append_csv(str(agg_csv_rank), row)
        accelerator.print(f"[{algo.upper()}] reward={gt_reward:.4f}, KL_U={kl_u:.4f}, beta={beta:.4f}")

    # Merge per-rank CSVs -> aggregate.csv
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        headers_written = False
        tmp_path = agg_csv_final.with_suffix(".csv.tmp")
        with open(tmp_path, "w") as fout:
            for r in range(world_size):
                path = out_dir / f"aggregate_rank{r}.csv"
                if not path.exists():
                    continue
                with open(path, "r") as fin:
                    lines = fin.readlines()
                if not lines:
                    continue
                if not headers_written:
                    fout.write(lines[0])   # header
                    headers_written = True
                    start = 1
                else:
                    start = 1 if len(lines) > 0 else 0
                for line in lines[start:]:
                    fout.write(line)
        os.replace(tmp_path, agg_csv_final)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean shutdown for distributed runs
        from contextlib import suppress
        with suppress(Exception):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # On NCCL it's cleaner to pass device_ids to avoid the "guessing device" warning
                try:
                    backend = torch.distributed.get_backend()
                except Exception:
                    backend = None
                if backend == "nccl" and torch.cuda.is_available():
                    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    torch.distributed.barrier()
                torch.distributed.destroy_process_group()
