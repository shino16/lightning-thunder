from functools import partial
import torch
import thunder
from thunder.dynamo import thunderfx
from pprint import pprint
from thunder.dev_utils.debug_memory_transform import DebugMemoryTransform, DebugMemoryFXTransform
from thunder.dynamo.utils import CompilerType

use_thunderfx = True
enable_grad = True


def fn(x, y):
    z = x.sinc() * 2
    w = z @ y
    del z
    return w

checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, fn, use_reentrant=False)

debug_memory_transform = DebugMemoryTransform()
debug_memory_fx_transform = DebugMemoryFXTransform()


if use_thunderfx:
    jfn = thunderfx(checkpoint_fn, executors=[], transforms=[debug_memory_transform], pre_inductor_transforms=[debug_memory_fx_transform])
else:
    jfn = thunder.jit(checkpoint_fn, executors=[], transforms=[debug_memory_transform])

x = torch.randn((128, 128), device="cuda", requires_grad=enable_grad)
y = torch.randn((128, 128), device="cuda", requires_grad=enable_grad)
y = jfn(x, y)
y.sum().backward()

if use_thunderfx:
    for sinfo in jfn._backend.subgraph_infos:
        for thunder_fn in sinfo.thunder_compiled_fns:
            print(thunder.last_traces(thunder_fn)[-1])

# pprint(debug_memory_transform.memory_events)
# pprint(debug_memory_fx_transform.memory_events)
for gm in debug_memory_fx_transform.augumented_graph_modules:
    gm.print_readable()
