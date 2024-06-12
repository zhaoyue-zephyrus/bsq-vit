import torch
import torch.nn as nn


def _inmap(x, eps=0.1):
    # map [0,1] range to [eps, 1-eps]
    return (1 - 2 * eps) * x + eps


def _unmap(x, eps=0.1):
    # inverse map, from [eps, 1-eps] to [0,1], with clamping
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


class LogitLaplaceLoss(nn.Module):
    def __init__(self, logit_laplace_eps = 0.1):
        super().__init__()
        self._logit_laplace_eps = logit_laplace_eps

    def inmap(self, x):
        return _inmap(x, eps=self._logit_laplace_eps)

    def unmap(self, x):
        return _unmap(x, eps=self._logit_laplace_eps)

    def forward(self, mu, lnb, target):
        logit = torch.logit(target)
        b = torch.exp(lnb)
        loss = torch.mean(torch.abs(logit - mu) / b + lnb + torch.log(target * (1 - target)))
        return loss
