#!/usr/bin/env bash
# M4 Apple acceptance: a clean consumer project uses the released Metal SDK
# archive without a server or source checkout. Apple analog of
# scripts/nvidia-sdk-consumer-acceptance.sh.
#
#   usage: apple-sdk-consumer-acceptance.sh <release-tarball> <model-dir> \
#            <metal|no-metal>
#
# <model-dir> is a provisioned MiniLM MLX directory (for example the output
# of `cargo xtask fetch --mlx minilm`, i.e. models/mlx/minilm) containing
# model.safetensors, tokenizer.json and the config files pinned in the
# packaged share/turboembed/model-pins/minilm.json.
#
# In a fresh temporary directory this verifies the archive hash and internal
# file manifest, verifies the model files against the packaged SHA-256 pins
# (a tampered model must be refused here — the load path does not re-hash),
# builds the installed C example as a separate CMake project against the
# extracted prefix, builds the documented Swift consumer with swiftc against
# the installed header and dylib, checks loader resolution, writes an
# explicit catalog file, and runs the device-policy matrix:
#   metal:    METAL selection must succeed; AUTO must resolve to Metal;
#             explicit CPU with a catalog alias must fail loudly (CPU never
#             silently serves catalog models on this dylib); explicit MOCK
#             serves only the mock-embed smoke alias
#   no-metal: METAL selection must fail loudly (no silent CPU fallback);
#             the MOCK smoke path must still work
#
# Consumer configure/build/run use a minimal environment so nothing resolves
# from the caller's shell.
set -euo pipefail

if [ $# -ne 3 ] || { [ "$3" != metal ] && [ "$3" != no-metal ]; }; then
  echo "usage: $0 <release-tarball> <model-dir> <metal|no-metal>" >&2
  exit 2
fi
if [ "$(uname -s)" != Darwin ]; then
  echo "error: the Apple SDK acceptance runs on macOS (the dylib is Mach-O arm64)." >&2
  exit 1
fi
TARBALL="$(readlink -f "$1")"
MODEL_DIR="$(readlink -f "$2")"
MODE="$3"
RUN_TIMEOUT="${TE_ACCEPTANCE_TIMEOUT:-180}"

# macOS ships no GNU `timeout`; bound runs with it only when present.
TIMEOUT_BIN="$(command -v timeout || true)"

# Run a consumer binary in the minimal environment (CONSUMER_ENV, defined
# after extraction), bounded by TIMEOUT_BIN when available.
run_consumer() {
  if [ -n "$TIMEOUT_BIN" ]; then
    "${CONSUMER_ENV[@]}" "$TIMEOUT_BIN" "$RUN_TIMEOUT" "$@"
  else
    "${CONSUMER_ENV[@]}" "$@"
  fi
}

WORK="$(mktemp -d /tmp/te-metal-consumer.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "== verify release archive hash =="
if [ -f "$TARBALL.sha256" ]; then
  (cd "$(dirname "$TARBALL")" && shasum -a 256 -c "$(basename "$TARBALL").sha256")
else
  echo "note: no .sha256 sibling next to the archive; skipping archive hash check"
fi
tar -xzf "$TARBALL"
PREFIX="$(readlink -f "$WORK"/turboembed-metal-sdk-*)"
[ -d "$PREFIX" ] || { echo "error: archive did not extract a turboembed-metal-sdk-* directory" >&2; exit 1; }

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
print("built with:", manifest["swift_version"])
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
cp -R "$MODEL_DIR" tampered-model
printf '\x00' | dd of=tampered-model/model.safetensors bs=1 seek=100 count=1 conv=notrunc status=none
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
rm -rf tampered-model

# CMake / swiftc / cc live outside /usr/bin on some hosts; keep the consumer
# environment minimal but resolvable. DEVELOPER_DIR is passed through when
# set so xcrun-backed shims work in the stripped environment.
CMAKE_BIN_DIR="$(dirname "$(command -v cmake)")"
SWIFTC_BIN_DIR="$(dirname "$(command -v swiftc)")"
MINIMAL_PATH="/usr/bin:/bin:$CMAKE_BIN_DIR:$SWIFTC_BIN_DIR"
CONSUMER_ENV=(env -i PATH="$MINIMAL_PATH" HOME="$WORK")
if [ -n "${DEVELOPER_DIR:-}" ]; then
  CONSUMER_ENV+=(DEVELOPER_DIR="$DEVELOPER_DIR")
fi

echo "== build the external C example against the extracted prefix =="
cp -R "$PREFIX/share/turboembed/examples" consumer
"${CONSUMER_ENV[@]}" cmake -S consumer -B consumer/build \
  -DCMAKE_PREFIX_PATH="$PREFIX" >cmake-configure.log
"${CONSUMER_ENV[@]}" cmake --build consumer/build >cmake-build.log
EXAMPLE="$WORK/consumer/build/turboembed_embed"

echo "== build the documented Swift consumer with swiftc =="
"${CONSUMER_ENV[@]}" swiftc "$PREFIX/share/turboembed/examples/swift/main.swift" \
  -import-objc-header "$PREFIX/include/turboembed.h" \
  -L "$PREFIX/lib" -lTurboEmbed \
  -Xlinker -rpath -Xlinker "$PREFIX/lib" \
  -o "$WORK/turboembed_embed_swift"
SWIFT_EXAMPLE="$WORK/turboembed_embed_swift"

echo "== loader resolves the SDK's own dylib =="
otool -L "$EXAMPLE" | tee otool.txt
grep -q "libTurboEmbed.dylib" otool.txt || {
  echo "error: consumer does not link libTurboEmbed.dylib" >&2
  exit 1
}
otool -l "$EXAMPLE" | grep -A2 LC_RPATH | grep -q "$PREFIX/lib" || {
  echo "error: consumer rpath does not point at the extracted SDK prefix" >&2
  exit 1
}

echo "== write the explicit consumer catalog =="
cat > catalog.toml <<EOF
[models.minilm]
description = "all-MiniLM-L6-v2 sentence embeddings (384 dims, mean pooling)"

[models.minilm.apple]
backend = "mlx"
path = "$MODEL_DIR"
tokenizer_dir = "$MODEL_DIR"
pooling = "mean"
normalize = true
max_seq_len = 256
max_batch_size = 32
EOF

echo "== explicit MOCK smoke run (ABI smoke alias only) =="
run_consumer "$EXAMPLE" catalog.toml mock-embed mock | tee mock-run.txt
grep -q "embed=PASS device=mock" mock-run.txt

echo "== device policy: explicit CPU must not serve catalog models =="
set +e
run_consumer "$EXAMPLE" catalog.toml minilm cpu \
  >cpu-run.txt 2>cpu-run.err
CPU_STATUS=$?
set -e
if [ "$CPU_STATUS" -eq 0 ]; then
  echo "error: explicit CPU served the minilm catalog alias — Metal MiniLM must not be silently substituted" >&2
  exit 1
fi
grep -qiE "METAL/AUTO|not served" cpu-run.err || {
  echo "error: missing loud CPU refusal message" >&2
  cat cpu-run.err >&2
  exit 1
}
echo "ok: explicit CPU refused the catalog alias loudly (exit $CPU_STATUS): $(head -n1 cpu-run.err)"

echo "== device policy: Metal selection ($MODE host) =="
set +e
run_consumer "$EXAMPLE" catalog.toml minilm metal \
  >metal-run.txt 2>metal-run.err
METAL_STATUS=$?
set -e
if [ "$MODE" = no-metal ]; then
  if [ "$METAL_STATUS" -eq 0 ]; then
    echo "error: Metal selection succeeded on a host declared no-metal" >&2
    exit 1
  fi
  grep -qiE "failed|unavailable|metal" metal-run.err || {
    echo "error: missing loud Metal failure message" >&2
    cat metal-run.err >&2
    exit 1
  }
  echo "ok: absent Metal failed loudly (exit $METAL_STATUS): $(head -n1 metal-run.err)"
else
  [ "$METAL_STATUS" -eq 0 ] || { echo "error: Metal run failed" >&2; cat metal-run.err >&2; exit 1; }
  grep -q "embed=PASS device=metal" metal-run.txt
  echo "ok: Metal run passed"

  echo "== AUTO resolves to the host GPU (Metal), never CPU/mock =="
  run_consumer "$EXAMPLE" catalog.toml minilm auto | tee auto-run.txt
  grep -q "embed=PASS device=auto" auto-run.txt
  grep -q "device=metal" auto-run.txt || {
    echo "error: AUTO did not list the catalog model on Metal" >&2
    exit 1
  }

  echo "== Swift consumer run (Metal) =="
  run_consumer "$SWIFT_EXAMPLE" catalog.toml minilm metal | tee swift-run.txt
  grep -q "embed=PASS device=metal" swift-run.txt
fi

echo "== bounded wall-clock check (5 repeats, finish line ${RUN_TIMEOUT}s each) =="
RUN_DEVICE=mock
RUN_ALIAS=mock-embed
if [ "$MODE" = metal ]; then
  RUN_DEVICE=metal
  RUN_ALIAS=minilm
fi
for i in 1 2 3 4 5; do
  START=$(date +%s)
  run_consumer "$EXAMPLE" catalog.toml "$RUN_ALIAS" "$RUN_DEVICE" >/dev/null
  END=$(date +%s)
  echo "run $i ($RUN_DEVICE): $((END - START))s"
done
echo "note: times are this machine's measurements, not a cross-host performance claim"

echo "ACCEPTANCE PASS ($MODE): archive verified, model pins enforced, C and Swift consumers built and ran offline from the package"
