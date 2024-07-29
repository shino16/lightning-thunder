import torch

from thunder.extend import OperatorExecutor, register_executor
import thunder.torch.optim.adam
from thunder.core.proxies import TensorProxy


def _adam_step_meta(optim):
    return None


adam_ex = thunder.extend.OperatorExecutor("adam_ex")
register_executor(adam_ex)


def _adam_checker(
    params: list[TensorProxy],
    grads: list[TensorProxy],
    exp_avgs: list[TensorProxy],
    exp_avg_sqs: list[TensorProxy],
    max_exp_avg_sqs: list[TensorProxy],
    state_steps: list[TensorProxy],
    foreach: None | bool = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: None | bool = None,
    grad_scale: None | TensorProxy = None,
    found_inf: None | TensorProxy = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float | TensorProxy,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    if grad_scale is not None or found_inf is not None:
        return False
    if any(param.is_complex() for param in params):
        return False
    if capturable or differentiable or amsgrad:
        return False

    return True


def _adam_transform(
    params: list[TensorProxy],
    grads: list[TensorProxy],
    exp_avgs: list[TensorProxy],
    exp_avg_sqs: list[TensorProxy],
    max_exp_avg_sqs: list[TensorProxy],
    state_steps: list[TensorProxy],
    foreach: None | bool = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: None | bool = None,
    grad_scale: None | TensorProxy = None,
    found_inf: None | TensorProxy = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float | TensorProxy,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    thunder.torch.optim.adam._single_tensor_adam(
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


adam_ex.register_implementation(
    thunder.torch._adam,
    checker=_adam_checker,
    execution_transform=_adam_transform,
)
