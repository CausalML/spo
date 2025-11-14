# src/trainers/espo_trainer.py
import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer, make_loader
from src.algos.espo import espo_step

class ESPOTrainer(BaseTrainer):
    def __init__(self, accelerator, policy, lr, wd, h, use_ddp=False, memory_cap=100_000,
                 kl_coef=0.01, prior_beta=2.0, prior_warmup_steps=500):
        super().__init__(accelerator)
        self.policy = policy
        self.optimizer = AdamW(self.policy.parameters(), lr=float(lr), weight_decay=float(wd))
        from ._utils import maybe_prepare
        self.policy, self.optimizer = maybe_prepare(accelerator, use_ddp, self.policy, self.optimizer)

        self.h = None if h=='None' else h  # can be None for auto
        self.kl_coef = kl_coef
        self.prior_beta = prior_beta
        self.prior_warmup = prior_warmup_steps

        # memory holds raw samples; we recompute t under current theta every step
        self.memory = {'x': None, 'y0': None, 'y1': None, 'z': None, 'cap': memory_cap}

        self._step = 0
        self._prev_logp = None  # for tiny TR-KL

    def train(self, x, y0, y1, z, steps=10000, batch_size=2048):
        loader = make_loader(x, y0, y1, z, batch_size, shuffle=True)
        for xb, y0b, y1b, zb in loader:
            # prior mix (ramps up to 1.0)
            prior_mix = min(1.0, max(0.0, (self._step / max(self.prior_warmup, 1))))
            metrics = espo_step(
                self.policy, xb, y0b, y1b, zb, optimizer=self.optimizer,
                memory=self.memory, h=self.h, step=self._step,
                kl_coef=self.kl_coef, prev_logp=self._prev_logp,
                prior_mix=prior_mix, prior_beta=self.prior_beta
            )

            # update memory with raw samples (on device)
            if self.memory['x'] is None:
                self.memory['x']  = xb.detach()
                self.memory['y0'] = y0b.detach()
                self.memory['y1'] = y1b.detach()
                self.memory['z']  = zb.detach()
            else:
                self.memory['x']  = torch.cat([self.memory['x'], xb.detach()],  dim=0)[-self.memory['cap']:]
                self.memory['y0'] = torch.cat([self.memory['y0'], y0b.detach()], dim=0)[-self.memory['cap']:]
                self.memory['y1'] = torch.cat([self.memory['y1'], y1b.detach()], dim=0)[-self.memory['cap']:]
                self.memory['z']  = torch.cat([self.memory['z'], zb.detach()],  dim=0)[-self.memory['cap']:]

            # keep a copy of current log-probs for TR-KL next step
            with torch.no_grad():
                self._prev_logp = self.policy(xb)

            self._step += 1
            if self._step >= steps:
                break
        return self.policy
