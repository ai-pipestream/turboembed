#!/usr/bin/env bash
# M4 NVIDIA acceptance: a clean consumer project uses the released CUDA SDK
# archive without a server or source checkout.
#
#   usage: nvidia-sdk-consumer-acceptance.sh <release-tarball> <model-dir> \
#            <gpu|cpu-only> [cuda-libs-dir]
#
# <model-dir> is a provisioned MiniLM directory (for example the output of
# `cargo run -p inferstream-fetch -- --embeddings minilm`, i.e.
# models/onnx/minilm) containing onnx/model.onnx and tokenizer.json.
# [cuda-libs-dir] holds the CUDA user-space libraries for the ORT CUDA EP
# (cuBLAS / cuDNN, e.g. .libs/nvidia/lib); required in gpu mode.
#
# In a fresh temporary directory this verifies the archive hash and internal
# file manifest, verifies the model files against the packaged SHA-256 pins
# (a tampered model must be refused here — the load path does not re-hash),
# builds the installed C example as a separate CMake project against the
# extracted prefix, checks loader resolution, writes an explicit catalog file,
# and runs the device-policy matrix:
#   gpu:      default CUDA selection must succeed; explicit CPU must succeed
#   cpu-only: CUDA selection must fail loudly (no silent CPU fallback);
#             explicit CPU must succeed
#
# Consumer configure/build/run use a minimal environment so nothing resolves
# from the caller's shell.
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ] || { [ "$3" != gpu ] && [ "$3" != cpu-only ]; }; then
  echo "usage: $0 <release-tarball> <model-dir> <gpu|cpu-only> [cuda-libs-dir]" >&2
  exit 2
fi
TARBALL="$(readlink -f "$1")"
MODEL_DIR="$(readlink -f "$2")"
MODE="$3"
CUDA_LIBS="${4:-}"
if [ "$MODE" = gpu ] && [ -z "$CUDA_LIBS" ]; then
  echo "error: gpu mode needs the CUDA user-space libs dir (cuBLAS/cuDNN)" >&2
  exit 2
fi
[ -z "$CUDA_LIBS" ] || CUDA_LIBS="$(readlink -f "$CUDA_LIBS")"
RUN_TIMEOUT="${TE_ACCEPTANCE_TIMEOUT:-180}"

WORK="$(mktemp -d /tmp/te-cuda-consumer.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "== verify release archive hash =="
if [ -f "$TARBALL.sha256" ]; then
  (cd "$(dirname "$TARBALL")" && sha256sum -c "$(basename "$TARBALL").sha256")
else
  echo "note: no .sha256 sibling next to the archive; skipping archive hash check"
fi
tar -xzf "$TARBALL"
PREFIX="$(readlink -f "$WORK"/turboembed-cuda-sdk-*)"
[ -d "$PREFIX" ] || { echo "error: archive did not extract a turboembed-cuda-sdk-* directory" >&2; exit 1; }

echo "== verify installed files against sdk-manifest.json =="
python3 - "$PREFIX" <<'EOF'
import hashlib, json, sys
from pathlib import Path

prefix = Path(sys.argv[1])
manifest = json.loads((prefix / "share/turboembed/sdk-manifest.json").read_text())
problems = []
for rel, expected in manifest["files"].items():
    if rel == "share/turboembed/sdk-manifest.json":
        continue
    path = prefix / rel
    if not path.is_file():
        problems.append(f"missing: {rel}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        problems.append(f"sha256 mismatch: {rel}")
for rel, target in manifest["symlinks"].items():
    path = prefix / rel
    if not path.is_symlink() or str(path.readlink()) != target:
        problems.append(f"symlink mismatch: {rel}")
if problems:
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    raise SystemExit(1)
print(f"ok: {len(manifest['files'])} files and {len(manifest['symlinks'])} symlinks verified")
print("sdk:", manifest["version"], manifest["git_commit"])
print("ort build:", manifest["ort_build"])
EOF

echo "== verify the provisioned model against the packaged pins =="
python3 - "$PREFIX" "$MODEL_DIR" <<'EOF'
import hashlib, json, sys
from pathlib import Path

prefix, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
pins = json.loads((prefix / "share/turboembed/model-pins/minilm.json").read_text())
problems = []
for rel, meta in pins["files"].items():
    path = model_dir / rel
    if not path.is_file():
        problems.append(f"missing: {rel}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        problems.append(f"sha256 mismatch: {rel}")
if problems:
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    print("error: model files do not match the qualified pins; refusing to run",
          file=sys.stderr)
    raise SystemExit(1)
print(f"ok: {len(pins['files'])} model files match revision {pins['revision']}")
EOF

echo "== a tampered model is refused by the pin verification =="
cp -r "$MODEL_DIR" tampered-model
printf '\x00' | dd of=tampered-model/onnx/model.onnx bs=1 seek=100 count=1 conv=notrunc status=none
set +e
python3 - "$PREFIX" "$WORK/tampered-model" <<'EOF'
import hashlib, json, sys
from pathlib import Path
prefix, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
pins = json.loads((prefix / "share/turboembed/model-pins/minilm.json").read_text())
for rel, meta in pins["files"].items():
    path = model_dir / rel
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise SystemExit(1)
EOF
TAMPER_STATUS=$?
set -e
if [ "$TAMPER_STATUS" -eq 0 ]; then
  echo "error: a tampered model passed pin verification" >&2
  exit 1
fi
echo "ok: tampered model rejected by pin verification"

echo "== build the external C example against the extracted prefix =="
cp -r "$PREFIX/share/turboembed/examples" consumer
env -i PATH=/usr/bin:/bin cmake -S consumer -B consumer/build \
  -DCMAKE_PREFIX_PATH="$PREFIX" >cmake-configure.log
env -i PATH=/usr/bin:/bin cmake --build consumer/build >cmake-build.log
EXAMPLE="$WORK/consumer/build/turboembed_embed"

echo "== loader resolves the SDK's own library =="
ldd "$EXAMPLE" | tee ldd.txt
if grep -q "not found" ldd.txt; then
  echo "error: unresolved consumer dependencies" >&2
  exit 1
fi
grep "libturboembed" ldd.txt | grep -q "$PREFIX/lib/" || {
  echo "error: libturboembed did not resolve from the extracted SDK prefix" >&2
  exit 1
}

echo "== write the explicit consumer catalog =="
cat > catalog.toml <<EOF
[models.minilm]
description = "all-MiniLM-L6-v2 sentence embeddings (384 dims, mean pooling)"

[models.minilm.nvidia]
backend = "ort"
device = "cuda"
path = "$MODEL_DIR/onnx/model.onnx"
tokenizer_dir = "$MODEL_DIR"
pooling = "mean"
normalize = true
max_seq_len = 256
max_batch_size = 32
EOF

RUN_ENV=(env -i PATH=/usr/bin:/bin)
if [ -n "$CUDA_LIBS" ]; then
  RUN_ENV=(env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH="$CUDA_LIBS")
fi

echo "== explicit CPU run =="
"${RUN_ENV[@]}" timeout "$RUN_TIMEOUT" "$EXAMPLE" catalog.toml minilm cpu | tee cpu-run.txt
grep -q "embed=PASS device=cpu" cpu-run.txt

echo "== device policy: CUDA selection ($MODE host) =="
set +e
"${RUN_ENV[@]}" timeout "$RUN_TIMEOUT" "$EXAMPLE" catalog.toml minilm cuda \
  >gpu-run.txt 2>gpu-run.err
GPU_STATUS=$?
set -e
if [ "$MODE" = cpu-only ]; then
  if [ "$GPU_STATUS" -eq 0 ]; then
    echo "error: CUDA selection succeeded on a host declared cpu-only" >&2
    exit 1
  fi
  grep -qiE "failed|unavailable|CUDA" gpu-run.err || {
    echo "error: missing loud CUDA failure message" >&2
    cat gpu-run.err >&2
    exit 1
  }
  echo "ok: absent CUDA failed loudly (exit $GPU_STATUS): $(head -n1 gpu-run.err)"
else
  [ "$GPU_STATUS" -eq 0 ] || { echo "error: CUDA run failed" >&2; cat gpu-run.err >&2; exit 1; }
  grep -q "embed=PASS device=cuda" gpu-run.txt
  echo "ok: CUDA run passed"
fi

echo "== bounded wall-clock check (5 repeats, finish line ${RUN_TIMEOUT}s each) =="
RUN_DEVICE=cpu
[ "$MODE" = gpu ] && RUN_DEVICE=cuda
for i in 1 2 3 4 5; do
  START=$(date +%s.%N)
  "${RUN_ENV[@]}" timeout "$RUN_TIMEOUT" "$EXAMPLE" catalog.toml minilm "$RUN_DEVICE" >/dev/null
  END=$(date +%s.%N)
  echo "run $i ($RUN_DEVICE): $(echo "$END $START" | awk '{printf "%.2f", $1-$2}')s"
done
echo "note: times are this machine's measurements, not a cross-host performance claim"

echo "ACCEPTANCE PASS ($MODE): archive verified, model pins enforced, consumer built and ran offline from the package"
