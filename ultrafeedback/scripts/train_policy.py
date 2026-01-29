from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse

from spo.config import RunConfig
from spo.training.trainer import train
from spo.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref_path", required=True)
    parser.add_argument("--method", required=True, choices=["dpo", "pspo", "ospo", "rspo"])
    parser.add_argument("--base_model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--ref_model_id", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)
    args = parser.parse_args()

    cfg = RunConfig()
    cfg.model.base_model_id = args.base_model_id
    cfg.model.ref_sft_model_id = args.ref_model_id
    cfg.training.method = args.method
    cfg.training.output_dir = args.output_dir
    cfg.training.max_steps = args.max_steps
    cfg.training.batch_size = args.batch_size
    cfg.training.grad_accum_steps = args.grad_accum_steps
    cfg.training.learning_rate = args.lr
    cfg.preference.seed = args.seed
    cfg.data.max_prompt_length = args.max_prompt_length
    cfg.data.max_response_length = args.max_response_length

    seed_everything(args.seed)
    train(cfg, pref_path=args.pref_path, device=args.device)


if __name__ == "__main__":
    main()
