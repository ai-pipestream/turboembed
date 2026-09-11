#!/usr/bin/env bash
# Fetch the HF tokenizer.json files the apple arch uses for Tokenize /
# Detokenize (served locally by the Rust tokenizers crate — no interpreter
# hop). The MLX runtime itself is native; this script does not create a
# venv or install packages.
#
# macOS only for a live Metal backend. Tokenizer files are just JSON and
# can be fetched on any host.
set -euo pipefail
cd "$(dirname "$0")/.."

# HF tokenizer.json for the default MiniLM embedding model.
mkdir -p models/minilm
if [ ! -f models/minilm/tokenizer.json ]; then
    curl -fsSL -o models/minilm/tokenizer.json \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    echo "fetched models/minilm/tokenizer.json"
fi

# HF tokenizer.json for default-llm / qwen-0.5b (mlx-community Qwen2.5-0.5B).
# Same dest the catalog's apple tokenizer_dir points at. Tokenizer only
# (~7 MB) — do not pull the nvidia GGUF here. Re-hash with
# `make fetch-llms ALIASES=qwen-0.5b` if you also want the Q8_0 weights.
mkdir -p models/gguf/qwen-0.5b
if [ ! -f models/gguf/qwen-0.5b/tokenizer.json ]; then
    curl -fsSL -o models/gguf/qwen-0.5b/tokenizer.json \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json"
    echo "fetched models/gguf/qwen-0.5b/tokenizer.json"
fi

# tokenizer.json for qwen-7b (mlx-community Qwen2.5-7B 4-bit).
mkdir -p models/gguf/qwen-7b
if [ ! -f models/gguf/qwen-7b/tokenizer.json ]; then
    curl -fsSL -o models/gguf/qwen-7b/tokenizer.json \
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer.json"
    echo "fetched models/gguf/qwen-7b/tokenizer.json"
fi

echo "setup-mlx: tokenizer files ready (models/minilm, models/gguf/qwen-0.5b, models/gguf/qwen-7b)"
echo "native MLX runtime: cargo build -p inferstream-arch-apple --release"
