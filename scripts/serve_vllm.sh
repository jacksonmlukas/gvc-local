#!/usr/bin/env bash
# Launch vLLM with an OpenAI-compatible API for the GVC-Local extension.
#
# Requirements: CUDA-capable GPU (A10 / A100 / 4090 / H100). For Llama 3.1 8B in fp16,
# ~16 GB VRAM is enough. For LLaMa 3.3 70B you'll need 2x 80GB A100 or quantized weights.
#
# Usage:
#   ./scripts/serve_vllm.sh                       # default: Llama 3.1 8B
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/serve_vllm.sh
#   PORT=8001 ./scripts/serve_vllm.sh

set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"         # tensor-parallel size
MAX_LEN="${MAX_LEN:-4096}"

echo "==> Serving ${MODEL} on port ${PORT} (tp=${TP}, max_model_len=${MAX_LEN})"

exec vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_LEN}" \
    --dtype auto \
    --gpu-memory-utilization 0.90
