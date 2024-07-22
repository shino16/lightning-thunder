import torch
from torch import Tensor

import thunder


def _dispatch_sqrt(x):
    if isinstance(x, Tensor):
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
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        assert not torch.is_complex(param)

        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

        assert not capturable and not differentiable

        step = step_t.to(param.dtype)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        assert not amsgrad
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

        param.addcdiv_(exp_avg, denom, value=-step_size)
