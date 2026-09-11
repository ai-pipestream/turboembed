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
#   This script curls pinned NVIDIA *wheels* (zip archives — no pip, no venv)
#   and links the .so files under .libs/nvidia. Wheel versions and SHA-256
#   are pinned below. libonnxruntime itself is downloaded by `ort`'s
#   `download-binaries` feature at cargo build time; nothing to do here.
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

# Pinned manylinux x86_64 wheels (CUDA 13 line, for ort 2.0.0-rc.13 / ONNX
# Runtime 1.28). URL + sha256 live in models/manifests/cuda-runtime-wheels.json
# so this script stays free of host-specific installer names. Wheels are zip
# files; we only extract the shared libraries.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS_DIR="$ROOT/.libs"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "error: need sha256sum or shasum" >&2
        exit 1
    fi
}

fetch_nvidia() {
    local dest="$LIBS_DIR/nvidia"
    local work="$LIBS_DIR/.wheels-nvidia"
    echo "==> nvidia: fetching CUDA 13 user-space libs via pinned NVIDIA wheels (~1.6 GB)"
    mkdir -p "$work" "$dest/lib"
    command -v unzip >/dev/null || { echo "error: unzip is required to extract wheels" >&2; exit 1; }
    command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
    command -v jq >/dev/null || { echo "error: jq is required to read the wheel manifest" >&2; exit 1; }

    local manifest="$ROOT/models/manifests/cuda-runtime-wheels.json"
    [ -f "$manifest" ] || { echo "error: missing $manifest" >&2; exit 1; }

    local n i
    n=$(jq '.wheels | length' "$manifest")
    i=0
    while [ "$i" -lt "$n" ]; do
        local name url expect wheel
        name=$(jq -r --argjson i "$i" '.wheels[$i].name' "$manifest")
        url=$(jq -r --argjson i "$i" '.wheels[$i].url' "$manifest")
        expect=$(jq -r --argjson i "$i" '.wheels[$i].sha256' "$manifest")
        wheel="$work/${name}.whl"
        if [ -f "$wheel" ] && [ "$(sha256_of "$wheel")" = "$expect" ]; then
            echo "  cached     $name"
        else
            echo "  downloading $name ..."
            curl -fL --retry 3 -o "$wheel.part" "$url"
            mv "$wheel.part" "$wheel"
            local actual
            actual=$(sha256_of "$wheel")
            if [ "$actual" != "$expect" ]; then
                rm -f "$wheel"
                echo "error: SHA-256 mismatch for $name" >&2
                echo "  expected $expect" >&2
                echo "  got      $actual" >&2
                exit 1
            fi
            echo "  verified   $name"
        fi
        unzip -qo "$wheel" -d "$work/extract-$name"
        i=$((i + 1))
    done

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
    done < <(find "$work" -type d -name lib -print0)
    if [ "$found" -eq 0 ]; then
        echo "error: no nvidia wheel lib directories found under $work" >&2
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
