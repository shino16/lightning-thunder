[`gm/`](https://github.com/shino16/lightning-thunder/tree/sglang-graph-modules/gm) contains the GraphModule compiled by ThunderFX.

All GraphModules collected here come from `sglang/srt/model_executor/cuda_graph_runner.py::patch_model`.

Reproduce with:
```sh
git checkout 1bb32071b54f7b4fc766a41b236d967aea6e6dd8
python3 -m sglang.bench_one_batch \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tp-size 4 \
  --trust-remote-code \
  --enable-torch-compile
```
after editing `sglang/srt/model_executor/cuda_graph_runner.py::patch_model` appropriately.
