import torch.testing

from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.framework import instantiate, NOTHING
import thunder.torch.optim.adam


@instantiate(
    dtypes=NOTHING,
)
def test_single_tensor_adam_like(executor, device, _):
    shape = (8, 8)

    params = [make_tensor(shape, device=device, dtype=torch.float32, requires_grad=True) for _ in range(2)]
    ref_params = [param.clone().detach().requires_grad_(True) for param in params]

    for param, ref_param in zip(params, ref_params):
        param.grad = make_tensor_like(param)
        ref_param.grad = param.grad.clone().detach()

    adam_ex = thunder.extend.get_executor("adam_ex")
    executors = [adam_ex] + executor.executors_list()

    adam = torch.optim.Adam(params, foreach=False)
    jitted_step = thunder.jit(adam.step, executors=executors)
    jitted_step()
    jitted_step()

    ref_adam = torch.optim.Adam(ref_params, foreach=False)
    ref_adam.step()
    ref_adam.step()

    torch.testing.assert_close(actual=params, expected=ref_params)
