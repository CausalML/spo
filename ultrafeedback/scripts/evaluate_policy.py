from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except Exception:
    PeftConfig = None
    PeftModel = None

from spo.eval.generate import generate_mixed
from spo.eval.metrics import compute_kl, summarize_rewards
from spo.models.reward import load_reward_model, score_rewards
from spo.utils.logging import setup_logging
from spo.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--ref_model_id", required=True)
    parser.add_argument("--reward_model_id", required=True)
    parser.add_argument("--dataset_id", default="openbmb/UltraFeedback")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--beta_grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed_id", type=int, default=None)
    parser.add_argument("--link_shift", type=float, default=None)
    parser.add_argument("--link_temperature", type=float, default=None)
    parser.add_argument("--num_preferences", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging("spo.evaluate")
    seed_everything(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    betas = [float(b) for b in args.beta_grid.split(",")]

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)
    prompts = []
    for ex in dataset.select(range(min(args.eval_samples, len(dataset)))):
        prompt = ex.get("prompt") or ex.get("instruction") or ex.get("query") or ex.get("question")
        if prompt is None:
            raise KeyError("Prompt field not found in dataset example")
        prompts.append(prompt)

    meta_path = os.path.join(args.model_id, "training_meta.json")
    sign = 1.0
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sign = float(meta.get("sign", 1.0))

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = None
    if PeftConfig is not None and os.path.exists(os.path.join(args.model_id, "adapter_config.json")):
        peft_cfg = PeftConfig.from_pretrained(args.model_id)
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        policy = PeftModel.from_pretrained(base, args.model_id)
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
    ref = AutoModelForCausalLM.from_pretrained(args.ref_model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    reward = load_reward_model(args.reward_model_id, bf16=True, trust_remote_code=True, device="auto")

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "model_id",
                "ref_model_id",
                "reward_model_id",
                "method",
                "seed",
                "link_shift",
                "link_temperature",
                "num_preferences",
                "beta",
                "avg_reward",
                "kl_div",
            ],
        )
        writer.writeheader()
        for beta in betas:
            all_rewards = []
            all_logprob_beta = []
            all_logprob_ref = []
            all_prompts = []
            all_responses = []
            for i in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[i : i + args.batch_size]
                gen = generate_mixed(
                    policy,
                    ref,
                    tokenizer,
                    batch_prompts,
                    beta=beta,
                    sign=sign,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=policy.device,
                )
                all_logprob_beta.append(gen.logprob_beta.detach().cpu())
                all_logprob_ref.append(gen.logprob_ref.detach().cpu())
                all_prompts.extend(batch_prompts)
                all_responses.extend(gen.responses)

            rewards = score_rewards(
                reward.model,
                reward.tokenizer,
                all_prompts,
                all_responses,
                args.max_new_tokens,
                reward.model.device,
                batch_size=args.batch_size,
            )
            avg_reward = summarize_rewards(rewards)
            logprob_beta = torch.cat(all_logprob_beta, dim=0)
            logprob_ref = torch.cat(all_logprob_ref, dim=0)
            kl = compute_kl(logprob_beta, logprob_ref)
            writer.writerow(
                {
                    "model_id": args.model_id,
                    "ref_model_id": args.ref_model_id,
                    "reward_model_id": args.reward_model_id,
                    "method": args.method,
                    "seed": args.seed_id,
                    "link_shift": args.link_shift,
                    "link_temperature": args.link_temperature,
                    "num_preferences": args.num_preferences,
                    "beta": beta,
                    "avg_reward": avg_reward,
                    "kl_div": kl,
                }
            )
            logger.info("beta %s reward %.4f kl %.4f", beta, avg_reward, kl)


if __name__ == "__main__":
    main()
