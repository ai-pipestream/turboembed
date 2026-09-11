#!/usr/bin/env bash
# Download prebuilt ONNX embedding artifacts for the catalog's nvidia
# resolutions (backend = "ort") into models/onnx/<alias>/.
#
# The built-in catalog (config/catalog.toml) resolves every embedding alias
# except `minilm` (already in krick's TEI cache) to
#   models/onnx/<alias>/onnx/model.onnx   +   models/onnx/<alias>/tokenizer.json
# relative to the server's cwd — run this from the repo root (or the
# directory you start inferstream-nvidia from), then add the alias to
# `serve` in config/nvidia.toml.
#
# Usage:
#   scripts/fetch-embedding-models.sh <alias> [<alias> ...]
#   scripts/fetch-embedding-models.sh --all
#   scripts/fetch-embedding-models.sh --list
#
# Sources are official/community ONNX exports on Hugging Face (the
# sentence-transformers repos ship onnx/model.onnx themselves; the rest use
# Xenova's transformers.js exports; nomic ships its own). Downloads use the
# `hf` CLI (pip install -U huggingface_hub) or `huggingface-cli` if that is
# what's installed.
set -euo pipefail
cd "$(dirname "$0")/.."

# alias -> HF repo whose onnx/model.onnx + root tokenizer.json we fetch.
declare -A REPOS=(
    [minilm-l12]="sentence-transformers/all-MiniLM-L12-v2"
    [mpnet]="sentence-transformers/all-mpnet-base-v2"
    [bge-small]="Xenova/bge-small-en-v1.5"
    [bge-base]="Xenova/bge-base-en-v1.5"
    [bge-large]="Xenova/bge-large-en-v1.5"
    [bge-m3]="Xenova/bge-m3"
    [e5-small]="Xenova/multilingual-e5-small"
    [e5-base]="Xenova/multilingual-e5-base"
    [e5-large]="Xenova/multilingual-e5-large"
    [gte-small]="Xenova/gte-small"
    [gte-base]="Xenova/gte-base"
    [nomic-embed-text]="nomic-ai/nomic-embed-text-v1.5"
)

usage() {
    echo "usage: $0 <alias> [<alias> ...] | --all | --list" >&2
    echo "aliases: $(printf '%s ' "${!REPOS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')" >&2
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

if [ "$1" = "--list" ]; then
    for alias in $(printf '%s\n' "${!REPOS[@]}" | sort); do
        printf '%-18s %s\n' "$alias" "${REPOS[$alias]}"
    done
    exit 0
fi

if [ "$1" = "--all" ]; then
    set -- $(printf '%s\n' "${!REPOS[@]}" | sort)
fi

# Prefer the modern `hf` CLI; fall back to the legacy name.
if command -v hf >/dev/null 2>&1; then
    HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=(huggingface-cli download)
else
    echo "error: need the Hugging Face CLI: pip install -U huggingface_hub" >&2
    exit 1
fi

for alias in "$@"; do
    repo="${REPOS[$alias]:-}"
    if [ -z "$repo" ]; then
        echo "error: unknown alias '$alias'" >&2
        usage
        exit 1
    fi
    dest="models/onnx/$alias"
    echo "--- $alias  <-  $repo  ->  $dest ---"
    mkdir -p "$dest"
    # tokenizer.json sits at the repo root; the ONNX graph under onnx/.
    # config.json rides along for reference (dims, architecture).
    "${HF[@]}" "$repo" onnx/model.onnx tokenizer.json config.json \
        --local-dir "$dest"
    test -f "$dest/onnx/model.onnx" || { echo "error: $dest/onnx/model.onnx missing after download" >&2; exit 1; }
    test -f "$dest/tokenizer.json"  || { echo "error: $dest/tokenizer.json missing after download" >&2; exit 1; }
done

echo
echo "Done. Add the aliases to 'serve' in config/nvidia.toml and restart;"
echo "then verify with: scripts/smoke-embeddings.sh <host:port> <bearer-token>"
