from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class LoadedRewardModel:
    model: torch.nn.Module
    tokenizer: AutoTokenizer


def load_reward_model(model_id: str, bf16: bool = True, trust_remote_code: bool = True, device: Optional[str] = None) -> LoadedRewardModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device or "auto",
    )
    model.eval()
    return LoadedRewardModel(model=model, tokenizer=tokenizer)


def score_rewards(
    reward_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch_texts = [p + r for p, r in zip(prompts[i : i + batch_size], responses[i : i + batch_size])]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = reward_model(**enc)
            batch_scores = outputs.logits.squeeze(-1)
        scores.append(batch_scores.detach().float().cpu())
    return torch.cat(scores, dim=0)
