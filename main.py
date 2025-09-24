import torch
from torch.utils.checkpoint import checkpoint
from thunder.dynamo import thunderfx

def fn(x):
    return x.sinc().sinc().sinc()

def checkpoint_fn(x):
    return checkpoint(fn, x, use_reentrant=False)

for compiler in [thunderfx, torch.compile]:
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.memory_allocated() == 0

    x = torch.randn((1024, 1024, 1024), device="cuda", requires_grad=True)  # 4 GB
    y = compiler(checkpoint_fn)(x)
    del x, y

    peak_mem_usage = torch.cuda.max_memory_allocated()
    print(f"{compiler.__name__}: {peak_mem_usage / 1024 / 1024 / 1024} GB")
