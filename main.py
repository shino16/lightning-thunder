from functools import partial
import torch
import thunder
from thunder.dynamo import thunderfx
from pprint import pprint
from thunder.dev_utils.debug_memory_transform import DebugMemoryTransform, DebugMemoryFXTransform
from thunder.dynamo.utils import CompilerType
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.utils.checkpoint import checkpoint
use_thunderfx = False
enable_grad = True


def fn(x):
    w = x.sin()
    return (w @ w).sinc()

def checkpoint_fn(x):
    return checkpoint(fn, x, use_reentrant=False)

debug_memory_transform = DebugMemoryTransform()
debug_memory_fx_transform = DebugMemoryFXTransform()


if use_thunderfx:
    jfn = thunderfx(checkpoint_fn, executors=[], transforms=[debug_memory_transform], pre_inductor_transforms=[debug_memory_fx_transform])
else:
    jfn = torch.compile(checkpoint_fn)

x = torch.randn((1024, 1024, 1024), device="cuda", requires_grad=enable_grad)

initial_mem = torch.cuda.memory_allocated()
assert initial_mem == torch.cuda.max_memory_allocated()

y = jfn(x)

peak_mem_usage = torch.cuda.max_memory_allocated() - initial_mem

# y.sum().backward()

if use_thunderfx:
    for sinfo in jfn._backend.subgraph_infos:
        for thunder_fn in sinfo.thunder_compiled_fns:
            print(thunder.last_traces(thunder_fn)[-1])
    for gm in debug_memory_fx_transform.augumented_graph_modules:
        gm.print_readable()

# pprint(debug_memory_transform.memory_events)
# pprint(debug_memory_fx_transform.memory_events)

print(f"{peak_mem_usage / 1024 / 1024 / 1024} GB")
