from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .dataset import PreferenceDataset, load_preferences
from .dpo import dpo_loss
from .pspo import compute_t as compute_t_pspo
from .pspo import pspo_loss, update_isotonic_state
from .ospo import compute_t as compute_t_ospo
from .ospo import init_memory, kernel_regression, ospo_loss, update_memory
from .rspo import rspo_loss
from .utils import batch_logprobs
from ..config import RunConfig
from ..models.policy import apply_lora, load_causal_lm
from ..utils.logging import setup_logging
from ..utils.seed import seed_everything


def split_responses(batch: Dict[str, list]):
    z = batch["z"]
    y_pref = []
    y_other = []
    for r0, r1, zi in zip(batch["response_0"], batch["response_1"], z):
        if int(zi) == 1:
            y_pref.append(r1)
            y_other.append(r0)
        else:
            y_pref.append(r0)
            y_other.append(r1)
    return y_pref, y_other


def save_checkpoint(model: torch.nn.Module, output_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)


def train(cfg: RunConfig, pref_path: str, device: str) -> str:
    logger = setup_logging("spo.train")
    seed_everything(cfg.preference.seed)
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    base = load_causal_lm(
        cfg.model.base_model_id,
        bf16=cfg.model.bf16,
        trust_remote_code=cfg.model.trust_remote_code,
        device=device,
    )
    ref_model_id = cfg.model.ref_sft_model_id or cfg.model.base_model_id
    ref = load_causal_lm(ref_model_id, bf16=cfg.model.bf16, trust_remote_code=cfg.model.trust_remote_code, device=device)

    policy_model = apply_lora(base.model, cfg.training.lora_r, cfg.training.lora_alpha, cfg.training.lora_dropout)
    policy_model.train()

    dataset = PreferenceDataset(load_preferences(pref_path))
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    pspo_iso_loader = None
    if cfg.training.method.lower() == "pspo":
        iso_bs = max(1, cfg.pspo.iso_batch_size)
        pspo_iso_loader = DataLoader(dataset, batch_size=iso_bs, shuffle=False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=cfg.training.max_steps)

    method = cfg.training.method.lower()
    step = 0
    memory = init_memory()
    psi_state = None

    while step < cfg.training.max_steps:
        for batch in dataloader:
            if step >= cfg.training.max_steps:
                break
            prompts = batch["prompt"]
            responses_0 = batch["response_0"]
            responses_1 = batch["response_1"]
            z = torch.tensor(batch["z"], device=policy_model.device)

            if method == "pspo" and (psi_state is None or step % cfg.pspo.theta_steps_per_round == 0):
                psi_state = update_isotonic_state(
                    policy_model,
                    ref.model,
                    base.tokenizer,
                    pspo_iso_loader or dataloader,
                    cfg.data.max_prompt_length,
                    cfg.data.max_response_length,
                    policy_model.device,
                )

            if method in {"dpo", "rspo"}:
                y_pref, y_other = split_responses(batch)
                logp_pref = batch_logprobs(
                    policy_model,
                    base.tokenizer,
                    prompts,
                    y_pref,
                    cfg.data.max_prompt_length,
                    cfg.data.max_response_length,
                    policy_model.device,
                )
                logp_other = batch_logprobs(
                    policy_model,
                    base.tokenizer,
                    prompts,
                    y_other,
                    cfg.data.max_prompt_length,
                    cfg.data.max_response_length,
                    policy_model.device,
                )
                with torch.no_grad():
                    logp_ref_pref = batch_logprobs(
                        ref.model,
                        base.tokenizer,
                        prompts,
                        y_pref,
                        cfg.data.max_prompt_length,
                        cfg.data.max_response_length,
                        policy_model.device,
                    )
                    logp_ref_other = batch_logprobs(
                        ref.model,
                        base.tokenizer,
                        prompts,
                        y_other,
                        cfg.data.max_prompt_length,
                        cfg.data.max_response_length,
                        policy_model.device,
                    )

                if method == "dpo":
                    loss = dpo_loss(logp_pref, logp_other, logp_ref_pref, logp_ref_other)
                else:
                    s_values = (logp_pref - logp_ref_pref) - (logp_other - logp_ref_other)
                    loss = rspo_loss(s_values)

            elif method == "pspo":
                t = compute_t_pspo(
                    policy_model,
                    ref.model,
                    base.tokenizer,
                    prompts,
                    responses_0,
                    responses_1,
                    cfg.data.max_prompt_length,
                    cfg.data.max_response_length,
                    policy_model.device,
                )
                loss = pspo_loss(t, z, psi_state)

            elif method == "ospo":
                t = compute_t_ospo(
                    policy_model,
                    ref.model,
                    base.tokenizer,
                    prompts,
                    responses_0,
                    responses_1,
                    cfg.data.max_prompt_length,
                    cfg.data.max_response_length,
                    policy_model.device,
                )
                g_hat = kernel_regression(t, memory, cfg.ospo.kernel, cfg.ospo.bandwidth)
                mix = 0.05
                g_hat = (1 - mix) * g_hat + mix * torch.sigmoid(t)
                loss = ospo_loss(t, z, g_hat)
                memory = update_memory(memory, t, z.cpu(), cfg.ospo.memory_size)
            else:
                raise ValueError(f"Unknown method: {method}")

            if torch.isfinite(loss):
                loss.backward()
            else:
                logger.warning("non-finite loss at step %s; skipping backward", step)
            if (step + 1) % cfg.training.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.training.log_every == 0:
                logger.info("step %s loss %.4f", step, loss.item())
            if step % cfg.training.save_every == 0 and step > 0:
                save_checkpoint(policy_model, cfg.training.output_dir, step)

            step += 1

    output_dir = os.path.join(cfg.training.output_dir, f"{method}_final")
    policy_model.save_pretrained(output_dir)

    # OSPO sign alignment
    sign = 1.0
    if method == "ospo":
        all_t = []
        all_z = []
        align_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in align_loader:
                t = compute_t_ospo(
                    policy_model,
                    ref.model,
                    base.tokenizer,
                    batch["prompt"],
                    batch["response_0"],
                    batch["response_1"],
                    cfg.data.max_prompt_length,
                    cfg.data.max_response_length,
                    policy_model.device,
                )
                all_t.append(t.detach().cpu())
                all_z.append(torch.tensor(batch["z"]))
        t_concat = torch.cat(all_t)
        z_concat = torch.cat(all_z).float()
        cov = torch.mean((t_concat - t_concat.mean()) * (z_concat - z_concat.mean()))
        sign = 1.0 if cov >= 0 else -1.0

    meta = {"method": method, "sign": sign, "config": asdict(cfg)}
    with open(os.path.join(output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return output_dir
