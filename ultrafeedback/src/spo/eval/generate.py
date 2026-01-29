from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GenerationResult:
    responses: List[str]
    logprob_beta: torch.Tensor
    logprob_ref: torch.Tensor


def sample_from_logits(logits: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    denom = sorted_probs.sum(dim=-1, keepdim=True)
    safe = denom > 0
    sorted_probs = torch.where(safe, sorted_probs / denom, sorted_probs)
    if not safe.all():
        # Fallback to argmax on rows with invalid distributions
        argmax = torch.argmax(logits, dim=-1)
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_token).squeeze(-1)
        next_token = torch.where(safe.squeeze(-1), next_token, argmax)
        return next_token
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_idx.gather(-1, next_token)
    return next_token.squeeze(-1)


def generate_mixed(
    policy_model,
    ref_model,
    tokenizer,
    prompts: List[str],
    beta: float,
    sign: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> GenerationResult:
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch_size = input_ids.size(0)
    policy_past = None
    ref_past = None
    logprob_beta = torch.zeros(batch_size, device=device)
    logprob_ref = torch.zeros(batch_size, device=device)
    generated = []

    # Prime cache with full prompt
    with torch.no_grad():
        policy_out = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    policy_past = policy_out.past_key_values
    ref_past = ref_out.past_key_values

    for _ in range(max_new_tokens):
        with torch.no_grad():
            policy_out = policy_model(input_ids=input_ids[:, -1:], attention_mask=attention_mask, past_key_values=policy_past, use_cache=True)
            ref_out = ref_model(input_ids=input_ids[:, -1:], attention_mask=attention_mask, past_key_values=ref_past, use_cache=True)
        policy_past = policy_out.past_key_values
        ref_past = ref_out.past_key_values

        logits_pol = policy_out.logits[:, -1, :]
        logits_ref = ref_out.logits[:, -1, :]

        mix_weight = sign / beta
        mix_logits = (1 - mix_weight) * logits_ref + mix_weight * logits_pol
        if torch.isnan(mix_logits).any() or torch.isinf(mix_logits).any():
            mix_logits = torch.nan_to_num(mix_logits, nan=0.0, posinf=0.0, neginf=0.0)

        next_token = sample_from_logits(mix_logits, top_p=top_p, temperature=temperature)

        logprob_beta += F.log_softmax(mix_logits, dim=-1).gather(1, next_token.unsqueeze(1)).squeeze(1)
        logprob_ref += F.log_softmax(logits_ref, dim=-1).gather(1, next_token.unsqueeze(1)).squeeze(1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(1))], dim=1)
        generated.append(next_token)

    gen_tokens = torch.stack(generated, dim=1) if generated else torch.empty((batch_size, 0), dtype=input_ids.dtype)
    responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return GenerationResult(responses=responses, logprob_beta=logprob_beta, logprob_ref=logprob_ref)
