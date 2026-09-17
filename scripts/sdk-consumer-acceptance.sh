#!/usr/bin/env bash
# M2 acceptance: a clean consumer project uses the released SDK archive
# without a server or source checkout.
#
#   usage: sdk-consumer-acceptance.sh <release-tarball> <bundle-dir> <cpu-only|gpu>
#
# In a fresh temporary directory this script verifies the archive hash and its
# internal file manifest, builds the installed C example as a separate CMake
# project against the extracted prefix, checks the loader resolves TurboEmbed,
# OpenVINO, and TBB from that prefix, embeds text and prepared tokens on the
# explicitly selected CPU, and proves a corrupted bundle cannot load.
#
# Device policy: with `cpu-only`, the example's default GPU selection must
# fail loudly (no silent CPU fallback); with `gpu`, it must succeed.
# The example itself verifies prepared-token/text agreement and the output
# norm; run wall-clock times are reported with a fixed finish line.
#
# All consumer configure/build/run steps use a minimal environment
# (env -i PATH=/usr/bin:/bin) so nothing resolves from the caller's shell.
set -euo pipefail

if [ $# -ne 3 ] || { [ "$3" != cpu-only ] && [ "$3" != gpu ]; }; then
  echo "usage: $0 <release-tarball> <bundle-dir> <cpu-only|gpu>" >&2
  exit 2
fi
TARBALL="$(readlink -f "$1")"
BUNDLE="$(readlink -f "$2")"
MODE="$3"
# Finish line for each single-example run (seconds). Generous by design: this
# bounds hangs, it is not a performance target.
RUN_TIMEOUT="${TE_ACCEPTANCE_TIMEOUT:-180}"

WORK="$(mktemp -d /tmp/te-consumer.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "== verify release archive hash =="
if [ -f "$TARBALL.sha256" ]; then
  (cd "$(dirname "$TARBALL")" && sha256sum -c "$(basename "$TARBALL").sha256")
else
  echo "note: no .sha256 sibling next to the archive; skipping archive hash check"
fi
tar -xzf "$TARBALL"
PREFIX="$(readlink -f "$WORK"/turboembed-prepared-sdk-*)"
[ -d "$PREFIX" ] || { echo "error: archive did not extract a turboembed-prepared-sdk-* directory" >&2; exit 1; }

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
print("openvino runtime:", manifest["openvino_build"])
EOF

echo "== build the external C example against the extracted prefix =="
cp -r "$PREFIX/share/turboembed/examples" consumer
env -i PATH=/usr/bin:/bin cmake -S consumer -B consumer/build \
  -DCMAKE_PREFIX_PATH="$PREFIX" >cmake-configure.log
env -i PATH=/usr/bin:/bin cmake --build consumer/build >cmake-build.log
EXAMPLE="$WORK/consumer/build/turboembed_prepared_embed"

echo "== loader resolves the SDK's own runtime =="
ldd "$EXAMPLE" | tee ldd.txt
if grep -q "not found" ldd.txt; then
  echo "error: unresolved consumer dependencies" >&2
  exit 1
fi
for lib in libturboembed_prepared libopenvino.so libtbb; do
  if ! grep "$lib" ldd.txt | grep -q "$PREFIX/lib/"; then
    echo "error: $lib did not resolve from the extracted SDK prefix" >&2
    exit 1
  fi
done

echo "== explicit CPU run (text + prepared tokens) =="
env -i PATH=/usr/bin:/bin timeout "$RUN_TIMEOUT" "$EXAMPLE" "$BUNDLE" cpu | tee cpu-run.txt
grep -q "prepared/text=PASS" cpu-run.txt

echo "== device policy: default GPU selection ($MODE host) =="
set +e
env -i PATH=/usr/bin:/bin timeout "$RUN_TIMEOUT" "$EXAMPLE" "$BUNDLE" >gpu-run.txt 2>gpu-run.err
GPU_STATUS=$?
set -e
if [ "$MODE" = cpu-only ]; then
  if [ "$GPU_STATUS" -eq 0 ]; then
    echo "error: GPU selection succeeded on a host declared cpu-only" >&2
    exit 1
  fi
  grep -qi "failed" gpu-run.err || { echo "error: missing loud GPU failure message" >&2; cat gpu-run.err >&2; exit 1; }
  echo "ok: absent GPU failed loudly (exit $GPU_STATUS): $(head -n1 gpu-run.err)"
else
  [ "$GPU_STATUS" -eq 0 ] || { echo "error: GPU run failed" >&2; cat gpu-run.err >&2; exit 1; }
  grep -q "prepared/text=PASS" gpu-run.txt
  echo "ok: GPU run passed"
fi

echo "== corrupted bundle cannot load =="
cp -r "$BUNDLE" tampered-bundle
printf '\x00' | dd of=tampered-bundle/openvino_model.bin bs=1 seek=100 count=1 conv=notrunc status=none
set +e
env -i PATH=/usr/bin:/bin timeout "$RUN_TIMEOUT" "$EXAMPLE" "$WORK/tampered-bundle" cpu >tamper.txt 2>tamper.err
TAMPER_STATUS=$?
set -e
if [ "$TAMPER_STATUS" -eq 0 ]; then
  echo "error: a tampered bundle loaded and executed" >&2
  exit 1
fi
grep -q "model load failed" tamper.err || { echo "error: tampering was not rejected at model load" >&2; cat tamper.err >&2; exit 1; }
echo "ok: tampered bundle rejected at load (exit $TAMPER_STATUS)"

echo "== bounded wall-clock check (5 repeats, CPU, finish line ${RUN_TIMEOUT}s each) =="
for i in 1 2 3 4 5; do
  START=$(date +%s.%N)
  env -i PATH=/usr/bin:/bin timeout "$RUN_TIMEOUT" "$EXAMPLE" "$BUNDLE" cpu >/dev/null
  END=$(date +%s.%N)
  echo "run $i: $(echo "$END $START" | awk '{printf "%.2f", $1-$2}')s"
done
echo "note: times are this machine's measurements, not a cross-host performance claim"

echo "ACCEPTANCE PASS ($MODE): archive verified, consumer built and ran offline from the package"
