from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - optional
    LoraConfig = None
    get_peft_model = None


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: AutoTokenizer


def load_causal_lm(model_id: str, bf16: bool = True, trust_remote_code: bool = True, device: Optional[str] = None) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device or "auto",
    )
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer)


def apply_lora(model: torch.nn.Module, r: int, alpha: int, dropout: float) -> torch.nn.Module:
    if get_peft_model is None:
        raise RuntimeError("peft is required for LoRA")
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def compute_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_prompt_length: int,
    max_response_length: int,
    device: torch.device,
) -> torch.Tensor:
    texts = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length + max_response_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute logprob of each response token only
    logprob_sums = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length)["input_ids"][0]
        resp_ids = tokenizer(response, return_tensors="pt", truncation=True, max_length=max_response_length)["input_ids"][0]
        start = len(prompt_ids) - 1
        end = start + len(resp_ids)
        token_logprobs = log_probs[i, start:end, :].gather(1, resp_ids.unsqueeze(1).to(device)).squeeze(1)
        logprob_sums.append(token_logprobs.sum())
    return torch.stack(logprob_sums, dim=0)
