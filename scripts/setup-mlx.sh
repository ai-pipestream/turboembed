#!/usr/bin/env bash
# Fetch native MLX weights + HF tokenizers for inferstream-apple.
# No Python. Runtime is the all-Swift server (swift/, mlx-swift in-process).
set -euo pipefail
cd "$(dirname "$0")/.."

cargo xtask fetch --mlx minilm bge-small qwen-0.5b
# Tokenize RPCs for default-llm / qwen-0.5b use the GGUF-side tokenizer.json
# (swift-transformers). Same file the LLM manifest already pins.
if [ ! -f models/gguf/qwen-0.5b/tokenizer.json ]; then
    cargo xtask fetch --llms qwen-0.5b
fi
if [ ! -f models/gguf/qwen-7b/tokenizer.json ]; then
    # tokenizer only — skip the multi-GB GGUF if already present as hash miss
    cargo xtask fetch --llms qwen-7b || true
fi

echo "--- native MLX weights ---"
ls -la models/mlx 2>/dev/null || echo "(run make fetch-mlx -- more aliases)"
echo "setup-mlx OK (no Python)"
