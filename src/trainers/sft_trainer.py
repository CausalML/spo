import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer, make_loader
from src.algos.sft import sft_step

class SFTTrainer(BaseTrainer):
    def __init__(self, accelerator, policy, lr, wd, save_dir):
        super().__init__(accelerator, save_dir)
        self.policy = policy
        self.optimizer = AdamW(self.policy.parameters(), lr=lr, weight_decay=wd)
        self.policy, self.optimizer = accelerator.prepare(self.policy, self.optimizer)

    def train(self, x, y0, y1, z, steps=10000, batch_size=2048, eval_every=1000):
        # winners only
        y_w = torch.where(z==1, y1, y0)
        loader = make_loader(x, y0, y1, z, batch_size)
        for step, (xb, y0b, y1b, zb) in enumerate(loader):
            y_wb = torch.where(zb==1, y1b, y0b)
            metrics = sft_step(self.policy, xb, y_wb, self.optimizer)
            if step+1 >= steps:
                break
        return self.policy

