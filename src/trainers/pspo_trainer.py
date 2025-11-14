import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer, make_loader
from src.algos.pspo import MonoLink, pspo_step
from src.algos.robust_tnorm import RobustTNormalizer

class PSPOTrainer(BaseTrainer):
    def __init__(self, accelerator, policy, lr, wd, use_ddp=False,
                 n_knots=16, t_min=-3, t_max=3, gamma=0.5,
                 link_lr=5e-3, link_mem_cap=50_000,
                 kl_coef=1e-2, clip_grad=1.0, alt_steps=2, train_knots=False,
                 save_dir=None, **_ignore):
        super().__init__(accelerator)
        self.policy = policy
        self.link = MonoLink(n_knots=n_knots, t_min=t_min, t_max=t_max, gamma=gamma, train_knots=train_knots)
        self.opt_theta = AdamW(self.policy.parameters(), lr=float(lr),  weight_decay=float(wd))
        self.opt_link  = AdamW(self.link.parameters(),   lr=float(link_lr), weight_decay=0.0)

        # Place each object on the right device; avoid DDP in task-parallel mode
        from ._utils import maybe_prepare
        self.policy, self.link, self.opt_theta, self.opt_link = maybe_prepare(
            accelerator, use_ddp, self.policy, self.link, self.opt_theta, self.opt_link
        )

        # Replay buffer for (normalized t, z) used by link fitting
        self.link_memory = {"t": None, "z": None, "cap": int(link_mem_cap)}

        # Robust normalizer mapping IQR to [-1,1] (EMA-smoothed)
        self.tnorm = RobustTNormalizer(s_min=1e-3, ema=0.05)

        self.kl_coef = float(kl_coef)
        self.clip_grad = float(clip_grad) if clip_grad is not None else None
        self.alt_steps = int(alt_steps)
        self._prev_logp = None
        self.save_dir = save_dir  # accepted for compatibility; not used here

    def train(self, x, y0, y1, z, steps=10000, batch_size=2048):
        loader = make_loader(x, y0, y1, z, batch_size, shuffle=True)
        for step, (xb, y0b, y1b, zb) in enumerate(loader):
            metrics = pspo_step(
                self.policy, self.link, self.tnorm,
                xb, y0b, y1b, zb,
                optimizer_theta=self.opt_theta, optimizer_link=self.opt_link,
                alt_steps=self.alt_steps,
                link_memory=self.link_memory, link_mem_cap=self.link_memory["cap"],
                kl_coef=self.kl_coef, prev_logp=self._prev_logp,
                clip_grad_norm=self.clip_grad,
            )

            # Keep previous log-probs for KL(prev||cur) next step
            with torch.no_grad():
                self._prev_logp = self.policy(xb)

            if step + 1 >= steps:
                break
        return self.policy
