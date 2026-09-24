#!/usr/bin/env bash
# Build the Inferstream service image (packaging/inferstream/Dockerfile).
#
# Usage: scripts/inferstream-image.sh [cpu|cuda] [--tag <image:tag>] [--toolchain <rust version>]
#
# cpu  (default): debian:bookworm-slim, llama.cpp CPU backend  -> turbo-inferstream:cpu
# cuda:           the NVIDIA CUDA devel and runtime bases, the ggml
#                 provider with llama.cpp's CUDA backend        -> turbo-inferstream:cuda
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
variant="cpu"
tag=""
toolchain="stable"
cuda_version="${CUDA_VERSION:-13.0.1}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        cpu|cuda) variant="$1" ;;
        --tag) tag="$2"; shift ;;
        --toolchain) toolchain="$2"; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
[[ -n "$tag" ]] || tag="turbo-inferstream:${variant}"
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
args=(--build-arg "RUST_TOOLCHAIN=$toolchain")
case "$variant" in
    cpu) ;;
    cuda)
        args+=(--build-arg "BUILD_BASE=nvidia/cuda:${cuda_version}-devel-ubuntu24.04"
               --build-arg "RUNTIME_BASE=nvidia/cuda:${cuda_version}-runtime-ubuntu24.04"
               --build-arg "GGML_FEATURES=cuda") ;;
esac
echo "==> building $tag ($variant)" >&2
DOCKER_BUILDKIT=1 docker build "${args[@]}" \
    --file "$root/packaging/inferstream/Dockerfile" \
    --tag "$tag" \
    "$root"
echo "==> $tag" >&2
docker image inspect "$tag" --format '    size {{.Size}} bytes, id {{.Id}}' >&2
