#!/bin/bash

python thunder/benchmarks/benchmark_litgpt.py --model_name Gemma-2-27b-like --compile dynamo_thunder --checkpoint_activations True --max_iters 1 --warmup_iters 0
