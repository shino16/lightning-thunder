# ThunderFX/Thunder JIT scalar boundary rerun (2025-12-15)

## Runs performed
- **Inference, ThunderFX, cache="symbolic values", sshleifer/tiny-gpt2**
  - `dynamic=None`, bs=1, in=32, out=8 → OK (no Thunder recompiles observed).
  - `dynamic=False`, bs=2, in=48, out=8 → OK (no Thunder recompiles observed).
  - `dynamic=True`, bs=1, in=32, out=8 → FAIL during AOTAutograd tensorify step:
    - `torch._dynamo.exc.TensorifyScalarRestartAnalysis` raised from `torch/fx/passes/_tensorify_python_scalars.py`.
    - Stack in log: `inductor.compile_fx -> aot_autograd -> tensorify_python_scalars`.

- **PEFT training (LoRA) sshleifer/tiny-gpt2**
  - `compile=thunder`, `dynamic=True`, `--var-seq-length` → same `TensorifyScalarRestartAnalysis` from AOTAutograd during torch.compile backend.
  - `compile=thunder+jit`, `--var-seq-length` → SDPA mask path still fails:
    - `TypeError: _add_batch_dim()` invoked with `TensorProxy` (int64 shape (1,)) inside functorch vmap for `sdpa_mask_recent_torch`.

## Notes
- Updated nvFuser did not surface in these failures; both torch.compile paths fail earlier in tensorify, and thunder.jit path still hits SDPA vmap proxy issue.
- No GraphModules were saved in `results/scalar_logs_v2/*` because no Thunder recompiles occurred before the failures; logs are in `agent-tools/*.txt`.

## Relevant logs
- Inference dynamic=True failure: `agent-tools/2186e273-2a77-4afe-8d62-419b744cffc1.txt`
- PEFT thunder dynamic=True failure: `agent-tools/5cdf67f9-3d89-456a-aa49-8b3bf642fd3b.txt`
- PEFT thunder+jit SDPA failure: `agent-tools/00d97113-8d28-4413-a7c2-44ebd7aeaecc.txt`

