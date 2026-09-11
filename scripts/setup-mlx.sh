#!/usr/bin/env bash
# Create the MLX venv the apple backend expects (.venv, gitignored) and
# fetch the MiniLM tokenizer.json the server's Tokenize/Detokenize RPCs use.
#
# Requires uv (https://docs.astral.sh/uv/). Python 3.12 is pinned because
# MLX wheel coverage for newer Pythons is still patchy. macOS only — MLX
# needs Metal.
set -euo pipefail
cd "$(dirname "$0")/.."

uv python install 3.12
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python mlx mlx-lm

# Embedding models (MiniLM & friends) need mlx-embeddings; best-effort since
# its dependency pins move faster than mlx itself.
if ! uv pip install --python .venv/bin/python mlx-embeddings; then
    echo "WARN: mlx-embeddings failed to install; the 'embed' op will be unavailable" >&2
fi

# HF tokenizer.json for the default MiniLM embedding model — served locally
# by the Rust server (tokenizers crate) for Tokenize/Detokenize, no Python
# round-trip. The MLX 4-bit community port shares the upstream tokenizer.
mkdir -p models/minilm
if [ ! -f models/minilm/tokenizer.json ]; then
    curl -fsSL -o models/minilm/tokenizer.json \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    echo "fetched models/minilm/tokenizer.json"
fi

# HF tokenizer.json for default-llm / qwen-0.5b (mlx-lm Qwen2.5-0.5B).
# Same dest the catalog's apple tokenizer_dir points at. This is the
# tokenizer only (~7 MB) — do not pull the nvidia GGUF here. Re-hash with
# `make fetch-llms ALIASES=qwen-0.5b` if you also want the Q8_0 weights.
mkdir -p models/gguf/qwen-0.5b
if [ ! -f models/gguf/qwen-0.5b/tokenizer.json ]; then
    curl -fsSL -o models/gguf/qwen-0.5b/tokenizer.json \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json"
    echo "fetched models/gguf/qwen-0.5b/tokenizer.json"
fi

echo "--- Metal sanity check (mlx.core matmul via the persistent bridge) ---"
echo '{"op":"ping"}' | .venv/bin/python python/mlx_bridge.py
