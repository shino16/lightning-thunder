# ThunderFX Dynamo Scalar Boundaries – Findings (2025-12-15)

## What we instrumented
- Added scalar/prologue logging switches:
  - `THUNDER_DEBUG_SCALARS` logs tensor/scalar proxy decisions (symbolic vs constrained) inside `thunder/core/jit_ext.py`.
  - `THUNDER_DEBUG_CACHE` logs Thunder prologue cache hits/misses with input summaries.
  - `THUNDER_DEBUG_DYNAMO` / `THUNDER_DEBUG_DYNAMO_FILE` record Dynamo compile invocations and guard sources in `thunder/dynamo/compiler.py`.
- Added `--dynamic` and `--scalar-log-dir` flags to `benchmark_inference.py` and `benchmark_peft.py`; both now emit recompile events via `ScalarRecompileTracker` (saved under the provided log dir).

## Experiments run
- **Inference (single GPU)**: `sshleifer/tiny-gpt2`, ThunderFX, cache=`symbolic values`
  - `dynamic=None` (default) `batch=1, seq_in=32, seq_out=8` → stable cache hits, no Thunder recompiles.
  - `dynamic=False` `batch=2, seq_in=48, seq_out=8` → stable cache hits, no Thunder recompiles.
  - `dynamic=True` crashed in nvFuser during dynamic transform concretization (nvfuser `DynamicTransformConcretizationInfo` stack).
- **PEFT training (single GPU)**:
  - ThunderFX (`compile=thunder`) with `--var-seq-length` hit `TensorifyScalarRestartAnalysis` during AOTAutograd tensorify pass.
  - Thunder JIT (`compile=thunder+jit`) failed in SDPA mask vmap path: `_add_batch_dim` received `TensorProxy` (functorch vmap on proxy).
  - Eager baseline succeeded (static and variable seq lengths); no recompiles by design.
- **Minimal symbolic-value probes (single GPU)**:
  - Shape changes alone with `cache="symbolic values"` did **not** trigger recompiles (Thunder treats tensor metadata as symbolic enough to pass prologue checks).
  - **dtype changes _do_ trigger Thunder recompiles** under symbolic cache: float32 → bfloat16 → float32 produced `cache_misses=2`.
  - Data-dependent Python int conversions (`int(proxy)`) are disallowed → TypeError; indicates boundary where symbolic scalars are not concretized and cannot enter Python control flow.
  - Data-dependent shape construction (`torch.zeros((int(x[0,0]), …))`) fails for the same reason (proxy-to-int not allowed).

## Where recompiles happened
- `results/scalar_logs/minimal_dtype` contains `minimal_dtype_graph0_sub0_miss2.json` (dtype change forced Thunder JIT recompile despite symbolic cache). GraphModule saving is limited for plain `thunder.jit` (TraceCtx lacks an attached FX module), but the meta file records the trigger (`dtype=torch.bfloat16`).
- No Thunder recompiles were observed in the HF inference runs for constant shapes/precision; Dynamo guards specialized Conv1D integer attributes (`nf`) even when `dynamic=True` was requested, but nvFuser crashed before usable data could be collected.
- ThunderFX with dtype variation did not record additional recompiles—Dynamo likely re-generated FX graphs per dtype, so Thunder saw a fresh graph instead of a cache miss.

## Behavioral boundaries observed
- **Dynamo**: Guard logs (see `f034be67-...txt`) show specialization on module integer attributes (e.g., `Conv1D.nf`) and parameter metadata. Changing these requires graph rebuilds even with `dynamic=True/None`.
- **Thunder symbolic cache**:
  - Symbolic values still guard dtype and layout; dtype changes cause Thunder recompile (cache miss >1).
  - Tensor shape/value derived Python ints are not allowed; attempting to convert symbolic proxies to `int` raises, showing where symbolic coverage ends.
  - Shape variation alone did not trigger Thunder recompile in our small probes, implying `check_tensor_shape_and_metadata` accepts symbolic extents when cache=`symbolic values`.

## Open issues / next steps
- Multi-GPU coverage not exercised due to time; would need torchrun + simple workload to see if collectives introduce new symbolic scalars.
- ThunderFX dynamic=True path currently crashes in nvFuser dynamic concretization for tiny GPT-2; needs upstream nvFuser fix or disabling dynamic transforms for that path.
- Thunder JIT SDPA mask path breaks with TensorProxy under vmap; likely needs mask transform guarding or data path bypass.
- Capturing FX GraphModules for Thunder JIT recompiles would require attaching FX exports to `TraceCtx` or saving from interpreter cache; current traces lack `graph_module`.

## Recommendations
- Treat dtype changes as a known recompilation trigger even under symbolic cache; avoid toggling dtypes across calls or ensure Thunder cache entries are keyed by dtype explicitly.
- Avoid Python control flow that consumes SymInt proxies (e.g., `int(x.shape[0])`, data-dependent shape creation) unless Dynamo is allowed to rebuild; these are outside symbolic-scalar coverage.
- For HF workloads, consider keeping `dynamic=None/False` until nvFuser dynamic-shape stability improves; for `dynamic=True`, disable offending transforms or fallback to eager/inductor on guard failures.
- Add optional export of FX GraphModules from Thunder JIT traces to better catalog recompiled graphs; currently only meta is saved for JIT cache misses.

