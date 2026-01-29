from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from spo.data.preference import normalize_reward_diffs, sample_preferences
from spo.models.reward import load_reward_model, score_rewards
from spo.utils.logging import setup_logging
from spo.utils.seed import seed_everything


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_responses = tokenizer.batch_decode(outputs[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        responses.extend(batch_responses)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="openbmb/UltraFeedback")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--reward_model_id", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--link_type", default="shifted_logistic_mixture")
    parser.add_argument("--link_shift", type=float, default=0.0)
    parser.add_argument("--link_temperature", type=float, default=1.0)
    parser.add_argument("--reward_diff_std", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    logger = setup_logging("spo.generate_preferences")
    seed_everything(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        logger.info("Preference file already exists: %s", args.output)
        return

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)
    prompts = []
    for ex in dataset.select(range(min(args.num_samples, len(dataset)))):
        prompt = ex.get("prompt") or ex.get("instruction") or ex.get("query") or ex.get("question")
        if prompt is None:
            raise KeyError("Prompt field not found in dataset example")
        prompts.append(prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    responses_0 = generate_responses(
        model, tokenizer, prompts, args.max_new_tokens, args.temperature, args.top_p, args.batch_size
    )
    responses_1 = generate_responses(
        model, tokenizer, prompts, args.max_new_tokens, args.temperature, args.top_p, args.batch_size
    )

    reward = load_reward_model(args.reward_model_id, bf16=True, trust_remote_code=True, device="auto")
    rewards_0 = score_rewards(
        reward.model, reward.tokenizer, prompts, responses_0, args.max_new_tokens, reward.model.device
    )
    rewards_1 = score_rewards(
        reward.model, reward.tokenizer, prompts, responses_1, args.max_new_tokens, reward.model.device
    )

    diffs = (rewards_1 - rewards_0).cpu().numpy()
    diffs, scale = normalize_reward_diffs(diffs, target_std=args.reward_diff_std)
    rng = np.random.default_rng(args.seed)
    z = sample_preferences(diffs, args.link_type, args.link_shift, args.link_temperature, rng)

    with open(args.output, "w", encoding="utf-8") as f:
        for prompt, r0, r1, z_i, d in zip(prompts, responses_0, responses_1, z, diffs):
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "response_0": r0,
                        "response_1": r1,
                        "z": int(z_i),
                        "reward_diff": float(d),
                        "reward_scale": float(scale),
                    }
                )
                + "\n"
            )

    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
