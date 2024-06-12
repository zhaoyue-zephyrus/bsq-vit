import functools
from typing import Dict, List, Optional, Tuple, Union, Iterable, cast

import torch
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """
    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


def clip_grad_norm_per_layer_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        foreach: Optional[bool] = None) -> None:
    r"""Clip the gradient norm of an iterable of parameters layer by layer.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]

    for ((device, _), ([grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        clips_coef_clamped = []
        for grad in grads:
            norm = torch.linalg.vector_norm(grad, norm_type)
            clip_coef = max_norm / (norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            clips_coef_clamped.append(clip_coef_clamped)
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):  # type: ignore[arg-type]
            torch._foreach_mul_(grads, clips_coef_clamped)  # type: ignore[call-overload]
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            for g, clip_coef_clamped in zip(grads, clips_coef_clamped):
                g.detach().mul_(clip_coef_clamped)
