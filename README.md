[`gm/`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/gm) contains the GraphModule compiled by ThunderFX.

ThunderFX compiled the following modules with SGLang.

- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3-0.6B
- openai/gpt-oss-20b
- mistralai/Magistral-Small-2509
- mistralai/Ministral-8B-Instruct-2410
- mistralai/Mistral-Large-Instruct-2411
- mistralai/Mixtral-8x7B-Instruct-v0.1
- microsoft/Phi-4-mini-reasoning
- huihui-ai/Qwen2.5-32B-Instruct-abliterated
- unsloth/Qwen3-30B-A3B-GGUF
- unsloth/gpt-oss-20b-BF16
- deepseek-ai/DeepSeek-V3.1

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
