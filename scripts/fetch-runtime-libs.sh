#!/usr/bin/env bash
# fetch-runtime-libs.sh — reproducible, no-sudo download of the GPU user-space
# runtime libraries each arch binary needs, into ./.libs/<arch>/.
#
# Usage:
#   scripts/fetch-runtime-libs.sh nvidia   # CUDA 13 user-space libs for ort-cuda
#   scripts/fetch-runtime-libs.sh intel    # no-op (documented below)
#   scripts/fetch-runtime-libs.sh all
#
# nvidia: the `ort` crate's prebuilt CUDA bundle (ONNX Runtime 1.28) is built
#   against CUDA 13, so the CUDA EP dlopens libcublasLt.so.13, libcublas.so.13,
#   libcudart.so.13, libnvrtc.so.13 and a cuDNN 9 built for CUDA 13 at startup.
#   This script fetches them from NVIDIA's official pip wheels (the strategy
#   already validated on krick) into a throwaway venv and symlinks the lib
#   directories under .libs/nvidia — no sudo, no system CUDA install, and the
#   wheel versions are pinned below for reproducibility. libonnxruntime itself
#   is downloaded by `ort`'s `download-binaries` feature at cargo build time;
#   nothing to do here.
#
# intel: the OVMS path (`backend = "ovms"`) is a pure tonic/prost gRPC client
#   to a running OpenVINO Model Server — it links NO OpenVINO libraries, so
#   there is nothing to fetch. The in-process OpenVINO backend is still a stub;
#   when its FFI link lands, its runtime libs will be added here.
#
# TensorRT EP (feature ort-tensorrt) is deliberately NOT handled here: it needs
# multi-GB TensorRT 10 host libs (`sudo apt install tensorrt-libs` from the
# NVIDIA apt repo). Keep it opt-in on hosts that already carry TensorRT.
set -euo pipefail

# Pinned wheel versions (CUDA 13 line, for ort 2.0.0-rc.13 / ONNX Runtime
# 1.28). Note the naming: the CUDA-13 generation ships the core libs under the
# UNSUFFIXED wheel names (nvidia-cublas, nvidia-cuda-runtime, …, landing in
# site-packages/nvidia/cu13/lib), while cuDNN keeps the -cu13 suffix. Bump
# deliberately, together.
NVIDIA_CUBLAS_WHEEL="nvidia-cublas==13.6.2.16"
NVIDIA_CUDA_RUNTIME_WHEEL="nvidia-cuda-runtime==13.2.86"
NVIDIA_NVRTC_WHEEL="nvidia-cuda-nvrtc==13.2.86"
NVIDIA_CUDNN_WHEEL="nvidia-cudnn-cu13==9.26.0.51"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS_DIR="$ROOT/.libs"

fetch_nvidia() {
    local venv="$LIBS_DIR/.venv-cuda-libs"
    local dest="$LIBS_DIR/nvidia"
    echo "==> nvidia: fetching CUDA 13 user-space libs via pinned NVIDIA pip wheels (~1.6 GB)"
    mkdir -p "$LIBS_DIR"
    if [ ! -x "$venv/bin/pip" ]; then
        python3 -m venv "$venv"
    fi
    "$venv/bin/pip" install --quiet --only-binary :all: \
        "$NVIDIA_CUBLAS_WHEEL" \
        "$NVIDIA_CUDA_RUNTIME_WHEEL" \
        "$NVIDIA_CUDNN_WHEEL" \
        "$NVIDIA_NVRTC_WHEEL"

    # Collect every wheel lib dir under a stable path the run wrapper and CI
    # can point LD_LIBRARY_PATH at: .libs/nvidia/lib
    rm -rf "$dest"
    mkdir -p "$dest/lib"
    local found=0
    while IFS= read -r -d '' libdir; do
        found=1
        for so in "$libdir"/*.so*; do
            [ -e "$so" ] || continue
            ln -sf "$so" "$dest/lib/$(basename "$so")"
        done
    done < <(find "$venv"/lib/python*/site-packages/nvidia -type d -name lib -print0)
    if [ "$found" -eq 0 ]; then
        echo "error: no nvidia wheel lib directories found under $venv" >&2
        exit 1
    fi
    echo "==> nvidia: libs linked under $dest/lib"
    echo "    run with: scripts/run-nvidia.sh --config config/nvidia.toml"
}

fetch_intel() {
    echo "==> intel: nothing to fetch."
    echo "    backend = \"ovms\" is a pure tonic gRPC client to a running OpenVINO"
    echo "    Model Server; it links no OpenVINO libraries. The in-process"
    echo "    OpenVINO backend is a stub — its libs will be added here when the"
    echo "    FFI link lands."
}

case "${1:-all}" in
    nvidia) fetch_nvidia ;;
    intel)  fetch_intel ;;
    all)    fetch_nvidia; fetch_intel ;;
    *) echo "usage: $0 [nvidia|intel|all]" >&2; exit 2 ;;
esac
