import torch

class RobustTNormalizer:
    """
    Maps IQR to [-1, 1] via an EMA-smoothed affine transform:
      m = (q25+q75)/2, s = (q75-q25)/2,  t_norm = (t - m)/max(s, s_min).
    Compute/update stats with no grad; treat them as constants during backprop.
    """
    def __init__(self, s_min: float = 1e-3, ema: float = 0.05):
        self.m = None
        self.s = None
        self.s_min = float(s_min)
        self.ema = float(ema)

    @torch.no_grad()
    def update_from(self, t_pool: torch.Tensor, max_points: int = 8192):
        if t_pool is None or t_pool.numel() == 0:
            return
        if t_pool.numel() > max_points:
            idx = torch.randint(0, t_pool.numel(), (max_points,), device=t_pool.device)
            t_sample = t_pool[idx]
        else:
            t_sample = t_pool

        q25, q75 = torch.quantile(
            t_sample, torch.tensor([0.25, 0.75], device=t_sample.device)
        )
        m_new = 0.5 * (q25 + q75)
        s_new = 0.5 * (q75 - q25)
        s_new = torch.clamp(s_new, min=self.s_min)

        if self.m is None:
            self.m = m_new
            self.s = s_new
        else:
            alpha = self.ema
            self.m = (1 - alpha) * self.m + alpha * m_new
            self.s = (1 - alpha) * self.s + alpha * s_new

    def normalize(self, t: torch.Tensor) -> torch.Tensor:
        # if not yet initialized, fall back to detached batch stats
        if self.m is None:
            m = t.detach().median()
            s = torch.clamp(t.detach().std(unbiased=False), min=self.s_min)
        else:
            m = self.m
            s = self.s
        return (t - m) / s
