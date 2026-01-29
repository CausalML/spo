from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class PreferenceExample:
    prompt: str
    response_0: str
    response_1: str
    z: int


def load_preferences(path: str) -> List[PreferenceExample]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                PreferenceExample(
                    prompt=item["prompt"],
                    response_0=item["response_0"],
                    response_1=item["response_1"],
                    z=int(item["z"]),
                )
            )
    return examples


class PreferenceDataset(Dataset):
    def __init__(self, examples: List[PreferenceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str | int]:
        ex = self.examples[idx]
        return {
            "prompt": ex.prompt,
            "response_0": ex.response_0,
            "response_1": ex.response_1,
            "z": ex.z,
        }
