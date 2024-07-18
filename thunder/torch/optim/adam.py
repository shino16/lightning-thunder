import torch
from torch import Tensor
import thunder
import math
from typing import List, Optional, Union
from torch.optim.optimizer import _use_grad_for_differentiable
from torch.optim.optimizer import _get_scalar_dtype


class Adam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError("`requires_grad` is not supported for `step` in differentiable mode")

                # Foreach without capturable does not support a tensor lr
                if group["foreach"] and torch.is_tensor(group["lr"]) and not group["capturable"]:
                    raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # if foreach and torch.jit.is_scripting():
    #     raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    # if fused and torch.jit.is_scripting():
    #     raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # if fused and not torch.jit.is_scripting():
    #     func = _fused_adam
    # elif foreach and not torch.jit.is_scripting():
    #     func = _multi_tensor_adam
    # else:
    func = _single_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _dispatch_sqrt(x):
    if isinstance(x, Tensor):
        return x.sqrt()
    return math.sqrt(x)


def _single_tensor_adam_impl(
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
        step_t = thunder.prims.copy_(step_t + 1, step_t)

        if weight_decay != 0:
            #     grad = grad.add(param, alpha=weight_decay)
            grad = grad.add(param * weight_decay)

        assert not torch.is_complex(param)

        # Decay the first and second moment running average coefficient
        exp_avg = thunder.prims.copy_(exp_avg * beta1 + grad * (1 - beta1), exp_avg)
        exp_avg_sq = thunder.prims.copy_((exp_avg_sq * beta2) + (1 - beta2) * grad * grad, exp_avg_sq)

        assert not capturable and not differentiable

        step = step_t.to(param.dtype)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        assert not amsgrad
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

        # param.addcdiv_(exp_avg, denom, value=-step_size)
        param = thunder.prims.copy_(param + (-step_size) * exp_avg / denom, param)


_single_tensor_adam = thunder.jit(_single_tensor_adam_impl, disable_inplace_copy_check=True)
