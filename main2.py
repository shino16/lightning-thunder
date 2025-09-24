import thunder
import thunder.dynamo
import torch

def fn(x):
    return x.mT

jfn = thunder.jit(fn)

x = torch.randn(10, 10).requires_grad_(True)
y = jfn(x)
print(y)
