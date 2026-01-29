from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    base_model_id: str = "Qwen/Qwen3-1.7B"
    ref_sft_model_id: Optional[str] = None
    reward_model_id: str = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
    trust_remote_code: bool = True
    use_flash_attn: bool = True
    bf16: bool = True


@dataclass
class DataConfig:
    dataset_id: str = "openbmb/UltraFeedback"
    dataset_split: str = "train"
    cache_dir: str = "data/cache"
    prompt_field: str = "prompt"
    response_field: str = "response"
    max_prompt_length: int = 512
    max_response_length: int = 512


@dataclass
class PreferenceConfig:
    num_samples: int = 2000
    link_shift: float = 0.0
    link_temperature: float = 1.0
    link_type: str = "shifted_logistic_mixture"
    seed: int = 123
    reward_diff_std_target: float = 1.0


@dataclass
class TrainingConfig:
    method: str = "dpo"
    num_epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_steps: int = 500
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "outputs"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class PSPOConfig:
    theta_steps_per_round: int = 50
    iso_rounds: int = 10
    pava_eps: float = 1e-4
    iso_batch_size: int = 1


@dataclass
class OSPOConfig:
    kernel: str = "gaussian"
    bandwidth: float = 0.5
    memory_size: int = 4096
    freeze_kernel_grads: bool = True


@dataclass
class RSPOConfig:
    pair_batch_size: int = 16


@dataclass
class EvalConfig:
    eval_samples: int = 256
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    beta_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    seed: int = 123


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pspo: PSPOConfig = field(default_factory=PSPOConfig)
    ospo: OSPOConfig = field(default_factory=OSPOConfig)
    rspo: RSPOConfig = field(default_factory=RSPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    extra: Dict[str, Any] = field(default_factory=dict)


def asdict(cfg: RunConfig) -> Dict[str, Any]:
    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(cfg)
