from torch.optim import AdamW
from .base_trainer import BaseTrainer, make_loader
from src.algos.smsspo import smsspo_step

class SMSSPOTrainer(BaseTrainer):
    def __init__(self, accelerator, policy, lr, wd, sigma, save_dir):
        super().__init__(accelerator, save_dir)
        self.policy = policy
        self.optimizer = AdamW(self.policy.parameters(), lr=float(lr), weight_decay=float(wd))
        self.sigma = sigma
        self.policy, self.optimizer = accelerator.prepare(self.policy, self.optimizer)

    def train(self, x, y0, y1, z, steps=10000, batch_size=2048):
        loader = make_loader(x, y0, y1, z, batch_size)
        for step, (xb, y0b, y1b, zb) in enumerate(loader):
            # Optional: anneal sigma
            cur_sigma = max(self.sigma * (0.98 ** step), 0.05)
            smsspo_step(self.policy, xb, y0b, y1b, zb, optimizer=self.optimizer, sigma=cur_sigma)
            if step+1 >= steps:
                break
        return self.policy

