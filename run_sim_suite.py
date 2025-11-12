import os, argparse, yaml, math
import torch
from accelerate import Accelerator
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

    out_dir = cfg['save_dir']
    ensure_dir(out_dir)
    agg_csv = os.path.join(out_dir, "aggregate.csv")

    for run in range(args.n_runs):
        seed = cfg['seed'] + run
        set_seed(seed)
        accelerator.print(f"=== Simulation run {run+1}/{args.n_runs} (seed={seed}) ===")

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

        for algo in ALGOS:
            accelerator.print(f"\n-- [{algo.upper()}] training --")
            policy = MLPPolicy(input_dim=cfg['feat_dim'], n_actions=cfg['n_actions'],
                               hidden_sizes=tuple(cfg['hidden_sizes']),
                               activation=cfg['activation']).to(device)

            if algo == "sft":
                trainer = SFTTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                     wd=cfg['weight_decay'], save_dir=out_dir)
            elif algo == "dpo":
                trainer = DPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                     wd=cfg['weight_decay'], beta_train=1.0, save_dir=out_dir)
            elif algo == "pspo":
                trainer = PSPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                      wd=cfg['weight_decay'], save_dir=out_dir)
            elif algo == "espo":
                trainer = ESPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                      wd=cfg['weight_decay'], h=cfg['h_espo'], save_dir=out_dir)
            elif algo == "smsspo":
                trainer = SMSSPOTrainer(accelerator, policy, lr=cfg['learning_rate'],
                                        wd=cfg['weight_decay'], sigma=cfg['sigma_smsspo'], save_dir=out_dir)
            else:
                raise ValueError(algo)

            # Train
            policy = trainer.train(x, y0, y1, z,
                                   steps=cfg['train_steps'], batch_size=cfg['batch_size'])

            # Calibration (β for KL budget)
            def logits_fn(X):
                # For KL-uniform recalibration, we can use h = log πθ
                return policy(X)  # [B,A]

            beta = calibrate_beta_for_kappa(logits_fn, x_calib, kappa=cfg['kappa'])

            # Evaluate on test
            with torch.no_grad():
                h_test = policy(x_test)  # logits h
                p_test = torch.softmax(h_test / beta, dim=-1)  # calibrated policy
                gt_reward = expected_true_reward(p_test, x_test, W)
                kl_u = kl_to_uniform(p_test)

            row = {
                'run': run, 'algo': algo, 'seed': seed,
                'beta': beta, 'kappa_target': cfg['kappa'],
                'test_reward': gt_reward, 'test_kl_uniform': kl_u
            }
            append_csv(agg_csv, row)
            accelerator.print(f"[{algo.upper()}] reward={gt_reward:.4f}, KL_U={kl_u:.4f}, beta={beta:.4f}")

if __name__ == "__main__":
    main()

