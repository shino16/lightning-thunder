import os
from contextlib import contextmanager
import unittest.mock
import functools

import torch
import thunder
from thunder.dynamo import thunderfx

import sglang
from sglang.srt.model_executor.cuda_graph_runner import GroupCoordinator, _to_torch


def nvtx_annotate(name):
    """Decorator to add NVTX range markers around a function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                torch.cuda.nvtx.range_pop()
        return wrapper
    return decorator


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            yield thunderfx(
                torch.no_grad()(model.forward),
                dynamic=False,
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def main():
    from sglang.bench_one_batch import ServerArgs, BenchArgs, argparse, logging, kill_process_tree
    import sys

    model_name = sys.argv[1]
    os.environ["THUNDER_SAVE_DIR"] = os.path.join("gm", model_name)
    os.environ["SGLANG_USE_DUMMY_WEIGHTS"] = "1"
    args = [
        "--model-path", model_name,
        "--tp-size", str(torch.cuda.device_count()),
        "--trust-remote-code",
        "--enable-torch-compile",
    ]
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args(args)
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    # Patch sglang functions with NVTX annotations for profiling
    import sglang.bench_one_batch as bench_module
    original_extend = bench_module.extend
    original_decode = bench_module.decode
    bench_module.extend = nvtx_annotate("extend_prefill")(original_extend)
    bench_module.decode = nvtx_annotate("decode")(original_decode)

    try:
        # Start CUDA profiler for nsys
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("sglang_benchmark")

        sglang.bench_one_batch.main(server_args, bench_args)

        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    finally:
        # Restore original functions
        bench_module.extend = original_extend
        bench_module.decode = original_decode

        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)

    # Note: _thunderfx_compiled_forward is set in subprocesses, so we can't access it here.
    # The subprocesses will save debug info to files via atexit handlers.
    # Check the output directory (configured via THUNDER_SAVE_DIR env var, defaults to thunderfx_debug/)
    debug_dir = os.path.join(os.getcwd(), os.environ.get("THUNDER_SAVE_DIR", "thunderfx_debug"))
    if os.path.exists(debug_dir):
        print(f"\n{'='*80}")
        print(f"ThunderFX debug info saved to: {debug_dir}")
        print(f"Files:")
        for fname in sorted(os.listdir(debug_dir)):
            fpath = os.path.join(debug_dir, fname)
            size = os.path.getsize(fpath)
            print(f"  - {fname} ({size:,} bytes)")
        print(f"{'='*80}\n")
    else:
        print("\nWarning: No thunderfx debug info found. This may be expected if compilation was skipped.")


if __name__ == "__main__":
    main()
