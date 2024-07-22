import torch.testing

from thunder.tests.make_tensor import make_tensor
from thunder.tests.framework import instantiate, NOTHING
from thunder.torch.optim import adam


@instantiate(
    dtypes=NOTHING,
)
def test_single_tensor_adam_like(executor, device, _):
    shape = (1024 * 1024,)
    params = [make_tensor(shape, device=device, dtype=torch.float32, high=2, low=1) for _ in range(2)]
    tensors = [
        [make_tensor(shape, device=device, dtype=torch.float32, high=2, low=1) for _ in range(2)] for _ in range(4)
    ]
    tensors = [params] + tensors
    state_steps = [torch.tensor(1, device=device) for _ in range(2)]

    kwargs = {
        "grad_scale": None,
        "found_inf": None,
        "amsgrad": False,
        "has_complex": False,
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 0.001,
        "weight_decay": 0,
        "eps": 1e-8,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }

    ref_tensors = [[t.clone().detach() for t in tensorlist] for tensorlist in tensors]
    ref_state_steps = [torch.tensor(1, device=device) for _ in range(2)]
    adam._single_tensor_adam(*ref_tensors, ref_state_steps, **kwargs)

    jitted = executor.make_callable(adam._single_tensor_adam)

    jitted(*tensors, state_steps, **kwargs)
    torch.testing.assert_close(actual=tensors + [state_steps], expected=ref_tensors + [ref_state_steps])
