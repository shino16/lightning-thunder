import torch
from torch.fx import Interpreter
import thunder
from thunder.dynamo import thunderfx
from pprint import pprint, pformat
from thunder.dev_utils.debug_memory_transform import DebugMemoryTransform, DebugMemoryFXTransform
from thunder.core.prims import PrimIDs
from thunder.dynamo.utils import CompilerType
from functools import partial

use_thunderfx = True
enable_grad = False


def fn(x):
    return x.sin().cos().sinc().sinc().exp()


debug_memory_transform = DebugMemoryTransform()
debug_memory_fx_transform = DebugMemoryFXTransform()


if use_thunderfx:
    jfn = thunderfx(fn, executors=[], transforms=[debug_memory_transform], pre_inductor_transforms=[debug_memory_fx_transform])
else:
    jfn = thunder.jit(fn, executors=[], transforms=[debug_memory_transform])

x = torch.randn(512 // 4, device="cuda", requires_grad=enable_grad)
y = jfn(x)
y.sum().backward()

if use_thunderfx:
    for sinfo in jfn._backend.subgraph_infos:
        for jfn in sinfo.submodule_to_compiled_functions.values():
            if jfn.compiler == CompilerType.THUNDER:
                print(thunder.last_traces(jfn.compiled_fn)[-1])

pprint(debug_memory_transform.memory_events)
pprint(debug_memory_fx_transform.memory_events)
for gm in debug_memory_fx_transform.augumented_graph_modules:
    gm.print_readable()

