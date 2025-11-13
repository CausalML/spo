import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer, make_loader
from src.algos.pspo import MonoLink, pspo_step

class PSPOTrainer(BaseTrainer):
    def __init__(self, accelerator, policy, lr, wd, use_ddp=False):
        super().__init__(accelerator)
        self.policy = policy
        self.link = MonoLink(n_knots=16, t_min=-6, t_max=6, gamma=0.5)
        self.opt_theta = AdamW(self.policy.parameters(), lr=float(lr), weight_decay=float(wd))
        self.opt_link = AdamW(self.link.parameters(), lr=1e-2, weight_decay=0.0)
        # self.policy, self.link, self.opt_theta, self.opt_link = accelerator.prepare(
            # self.policy, self.link, self.opt_theta, self.opt_link
        # )
        from ._utils import maybe_prepare
        self.policy, self.link, self.opt_theta, self.opt_link = maybe_prepare(accelerator, use_ddp, self.policy, self.link, self.opt_theta, self.opt_link)

    def train(self, x, y0, y1, z, steps=10000, batch_size=2048):
        loader = make_loader(x, y0, y1, z, batch_size)
        for step, (xb, y0b, y1b, zb) in enumerate(loader):
            pspo_step(self.policy, self.link, xb, y0b, y1b, zb,
                      optimizer_theta=self.opt_theta, optimizer_link=self.opt_link, alt_steps=2)
            if step+1 >= steps:
                break
        return self.policy

