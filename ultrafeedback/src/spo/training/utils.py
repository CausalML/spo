from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def build_inputs(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_prompt_length: int,
    max_response_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_list = []
    attention_list = []
    response_mask_list = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-max_prompt_length:]
        response_ids = response_ids[:max_response_length]
        input_ids = prompt_ids + response_ids
        attention = [1] * len(input_ids)
        response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        input_ids_list.append(input_ids)
        attention_list.append(attention)
        response_mask_list.append(response_mask)

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id

    def pad(seq, value=0):
        return seq + [value] * (max_len - len(seq))

    input_ids = torch.tensor([pad(ids, pad_id) for ids in input_ids_list])
    attention_mask = torch.tensor([pad(attn, 0) for attn in attention_list])
    response_mask = torch.tensor([pad(mask, 0) for mask in response_mask_list])
    return input_ids, attention_mask, response_mask


def batch_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_prompt_length: int,
    max_response_length: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention_mask, response_mask = build_inputs(
        tokenizer, prompts, responses, max_prompt_length, max_response_length
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    response_mask = response_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Shift to align tokens
    shifted_ids = input_ids[:, 1:]
    shifted_log_probs = log_probs[:, :-1, :]
    shifted_mask = response_mask[:, 1:]

    token_logprobs = shifted_log_probs.gather(2, shifted_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * shifted_mask
    return token_logprobs.sum(dim=1)
