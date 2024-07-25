import torch

from thunder.extend import OperatorExecutor, register_executor
import thunder.torch.optim.adam


def _adam_step_meta(optim):
    return None


adam_ex = thunder.extend.OperatorExecutor("adam_ex")
register_executor(adam_ex)

adam_step = adam_ex.register_operator(
    "adam_step",
    meta=_adam_step_meta,
    fn=thunder.torch.optim.adam.step,
    tags=(thunder.prims.OpTags.DONT_DCE,),
)


def _adam_step_checker(optim):
    if getattr(optim, "grad_scale", None) is not None or getattr(optim, "found_inf", None) is not None:
        return False

    for group in optim.param_groups:
        if group["foreach"] or group["fused"]:
            return False
        if group["capturable"] or group["differentiable"] or group["amsgrad"]:
            return False
    return True


def _adam_step_transform(optim):
    adam_step(optim)


adam_ex.register_implementation(
    thunder.torch._adam_step,
    checker=_adam_step_checker,
    execution_transform=_adam_step_transform,
)
