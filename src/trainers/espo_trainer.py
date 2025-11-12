from torch.optim import AdamW
from .base_trainer import BaseTrainer, make_loader
from src.algos.espo import espo_step

class ESPOTrainer(BaseTrainer):
    def __init__(self, accelerator, policy, lr, wd, h, save_dir, memory_cap=100000):
        super().__init__(accelerator, save_dir)
        self.policy = policy
        self.optimizer = AdamW(self.policy.parameters(), lr=float(lr), weight_decay=float(wd))
        self.h = h
        self.memory = {'t': None, 'z': None, 'cap': memory_cap}
        self.policy, self.optimizer = accelerator.prepare(self.policy, self.optimizer)

    def train(self, x, y0, y1, z, steps=10000, batch_size=2048):
        loader = make_loader(x, y0, y1, z, batch_size)
        for step, (xb, y0b, y1b, zb) in enumerate(loader):
            espo_step(self.policy, xb, y0b, y1b, zb, optimizer=self.optimizer, h=self.h, memory=self.memory)
            if step+1 >= steps:
                break
        return self.policy

