[`gm/`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/gm) contains the GraphModule compiled by ThunderFX.

ThunderFX compiled the following modules with SGLang.

- Qwen/Qwen3-0.6B
- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
- Qwen/Qwen3-Next-80B-A3B-Thinking
- deepseek-ai/DeepSeek-V3.1
- huihui-ai/Qwen2.5-32B-Instruct-abliterated
- meta-llama/Llama-4-Maverick-17B-128E-Instruct
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- microsoft/Phi-4-mini-reasoning
- mistralai/Magistral-Small-2509
- mistralai/Ministral-8B-Instruct-2410
- mistralai/Mistral-Large-Instruct-2411
- mistralai/Mixtral-8x7B-Instruct-v0.1
- openai/gpt-oss-20b
- unsloth/Qwen3-30B-A3B-GGUF
- unsloth/gpt-oss-20b-BF16

You can run the models in the same configuration with the following command:

```sh
python3 sglang_scripts/save_graph_modules.py <model_name>
```

whose output should be the same as

```sh
python3 -m sglang.bench_one_batch \
    --model-path <model_name> \
    --tp-size 4 \
    --trust-remote-code \
    --enable-torch-compile
```

except that the former sets some NVTX annotations for profiling and creates dummy model cache to skip downloading. See [`sglang_scripts/save_graph_modules.py`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/sglang_scripts/save_graph_modules.py) for the details.

See [`gm/all_ops.py`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/gm/all_ops.py) for the list of ops in those GraphModules that Thunder did not support. This file was produced by [`sglang_scripts/extract_ops.py`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/sglang_scripts/extract_ops.py).

Note. meta-llama/Llama-4-Maverick-17B-128E-Instruct and Llama-4-Scout-17B-16E-Instruct require specifying `--attention-backend [fa3|aiter|triton]`. However, any combination gives the error `AttributeError: module 'triton.language' has no attribute 'constexpr_function'` as seen in [`gm/meta-llama-Llama-4-Maverick-17B-128E-Instruct/log.txt`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/gm/meta-llama-Llama-4-Maverick-17B-128E-Instruct/log.txt). The former is also run with `--load-format dummy` option without creating model cache, because the weight tensors were too big for a typical disk space of a node.

Note. On openai/gpt-oss-20b and unsloth/gpt-oss-20b-BF16, a `_to_torch` call switches the CustomOp's forward implementation from forward_cuda to PyTorch-native implementation, causing a TypeError for an unrecognized argument. I edited patch_model to remove the problematic `_to_torch` calls.
