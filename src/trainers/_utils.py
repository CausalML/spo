import torch, torch.nn as nn

def maybe_prepare(accelerator, use_ddp, *objs):
    if use_ddp and accelerator.num_processes > 1:
        return accelerator.prepare(*objs)
    prepped = []
    for o in objs:
        if isinstance(o, nn.Module):
            o = o.to(accelerator.device)
        prepped.append(o)
    return tuple(prepped)
