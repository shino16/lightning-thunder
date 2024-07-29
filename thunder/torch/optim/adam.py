import torch
from torch import Tensor
from torch.optim.optimizer import _use_grad_for_differentiable
from torch.optim.optimizer import _get_scalar_dtype

import thunder
from thunder.core.proxies import TensorProxy


def _dispatch_sqrt(x):
    if isinstance(x, (Tensor, TensorProxy)):
        return x.sqrt()
    return math.sqrt(x)


def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

        bias_correction1 = 1 - beta1**step_t
        bias_correction2 = 1 - beta2**step_t

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

        param.addcdiv_(exp_avg, denom, value=-step_size)
