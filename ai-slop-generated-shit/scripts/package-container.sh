#!/usr/bin/env bash
# Build the Linux distribution archive inside the manylinux_2_28 container
# (packaging/Dockerfile) and write it to dist/.
#
# Usage: scripts/package-container.sh [x86_64|aarch64] [--toolchain <rust version>]
#
# aarch64 on an x86_64 host needs qemu user-mode emulation registered with
# binfmt_misc (for example the docker/binfmt or tonistiigi/binfmt image);
# the build then runs under emulation and takes correspondingly longer.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
arch="x86_64"
toolchain="stable"
while [[ $# -gt 0 ]]; do
    case "$1" in
        x86_64|aarch64) arch="$1" ;;
        --toolchain) toolchain="$2"; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
case "$arch" in
    x86_64) platform="linux/amd64" ;;
    aarch64) platform="linux/arm64" ;;
esac
base="quay.io/pypa/manylinux_2_28_${arch}"
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
echo "==> building the ${arch} archive on ${base} with Rust ${toolchain}" >&2
mkdir -p "$root/dist"
DOCKER_BUILDKIT=1 docker build \
    --platform "$platform" \
    --build-arg "BASE=$base" \
    --build-arg "RUST_TOOLCHAIN=$toolchain" \
    --file "$root/packaging/Dockerfile" \
    --target export \
    --output "type=local,dest=$root/dist" \
    "$root"
echo "==> archives in dist/:" >&2
ls -la "$root"/dist/turbo-*-"${arch}"-unknown-linux-gnu.tar.gz >&2
