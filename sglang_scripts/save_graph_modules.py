import os
import functools

from safetensors.torch import save_file
from huggingface_hub import get_safetensors_metadata, get_hf_file_metadata, hf_hub_url
from huggingface_hub.file_download import repo_folder_name, _get_pointer_path
import huggingface_hub.constants

import torch
import sglang


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


def populate_dummy_model_cache(model_name):
    dtype_map = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F8_E4M3": torch.float8_e4m3fn,
    }

    storage_folder = os.path.join(
        huggingface_hub.constants.HF_HUB_CACHE, repo_folder_name(repo_id=model_name, repo_type="model")
    )
    metadata = get_safetensors_metadata(model_name)

    for filename, file_metadata in metadata.files_metadata.items():
        print(f"Preparing dummy cache for {filename}")
        dummy_tensors = {}
        for tensor_name, tensor_info in file_metadata.tensors.items():
            dtype = dtype_map[tensor_info.dtype]
            dummy_tensors[tensor_name] = torch.empty(tensor_info.shape, dtype=dtype)
        commit_hash = get_hf_file_metadata(hf_hub_url(model_name, filename)).commit_hash
        pointer_path = _get_pointer_path(storage_folder, commit_hash, filename)
        print(f"Saving dummy cache to {pointer_path}")
        os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
        save_file(dummy_tensors, pointer_path)


def main():
    from sglang.bench_one_batch import ServerArgs, BenchArgs, argparse, logging, kill_process_tree
    import sys

    model_name = sys.argv[1]
    os.environ["THUNDER_SAVE_DIR"] = os.path.join("gm", model_name)

    # populate_dummy_model_cache(model_name)

    args = [
        "--model-path",
        model_name,
        "--tp-size",
        str(torch.cuda.device_count()),
        "--trust-remote-code",
        "--enable-torch-compile",
        "--load-format",
        "dummy",
    ] + sys.argv[2:]
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
        print(f"\n{'=' * 80}")
        print(f"ThunderFX debug info saved to: {debug_dir}")
        print("Files:")
        for fname in sorted(os.listdir(debug_dir)):
            fpath = os.path.join(debug_dir, fname)
            size = os.path.getsize(fpath)
            print(f"  - {fname} ({size:,} bytes)")
        print(f"{'=' * 80}\n")
    else:
        print("\nWarning: No thunderfx debug info found. This may be expected if compilation was skipped.")


if __name__ == "__main__":
    main()
