from __future__ import annotations

from typing import Dict, Iterable, List

from datasets import load_dataset


DEFAULT_PROMPT_FIELDS = ["prompt", "instruction", "query", "question"]
DEFAULT_RESPONSE_FIELDS = ["response", "output", "answer", "chosen"]


def load_ultrafeedback(dataset_id: str, split: str, cache_dir: str):
    return load_dataset(dataset_id, split=split, cache_dir=cache_dir)


def pick_field(example: Dict[str, str], fields: List[str]) -> str:
    for field in fields:
        if field in example and example[field] is not None:
            return field
    raise KeyError(f"None of fields found in example: {fields}")


def get_prompt(example: Dict[str, str], prompt_field: str | None = None) -> str:
    if prompt_field and prompt_field in example:
        return example[prompt_field]
    field = pick_field(example, DEFAULT_PROMPT_FIELDS)
    return example[field]


def get_response(example: Dict[str, str], response_field: str | None = None) -> str:
    if response_field and response_field in example:
        return example[response_field]
    field = pick_field(example, DEFAULT_RESPONSE_FIELDS)
    return example[field]


def iter_prompts(dataset, prompt_field: str | None = None) -> Iterable[str]:
    for ex in dataset:
        yield get_prompt(ex, prompt_field)


def sample_prompts(dataset, num_samples: int, seed: int, prompt_field: str | None = None) -> List[str]:
    dataset = dataset.shuffle(seed=seed)
    prompts = []
    for ex in dataset.select(range(min(num_samples, len(dataset)))):
        prompts.append(get_prompt(ex, prompt_field))
    return prompts
