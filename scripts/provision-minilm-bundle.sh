#!/usr/bin/env bash
# Provision the qualified MiniLM bundle from fetched prepared-SDK sources.
#
# Verifies the fetched source files against models/manifests/prepared-sources.json,
# then runs the installed SDK's prepare-native-bundle.py with the pinned model
# identity, revision, and license from that manifest. Fetch the sources first:
#
#   cargo run -p inferstream-fetch -- --prepared minilm
#
# No network access happens here; inference never downloads models.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <sdk-prefix> <output-bundle-dir>" >&2
  exit 2
fi
SDK_PREFIX="$(cd "$1" && pwd)"
OUTPUT="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/models/manifests/prepared-sources.json"
SOURCES="$ROOT/models/prepared-src/minilm"

for tool in "$SDK_PREFIX/bin/prepare-native-bundle.py" "$SDK_PREFIX/bin/turboembed-export-model"; do
  [ -f "$tool" ] || { echo "error: missing SDK tool: $tool" >&2; exit 1; }
done
[ -f "$MANIFEST" ] || { echo "error: missing manifest: $MANIFEST" >&2; exit 1; }

read -r MODEL_ID REVISION < <(python3 - "$MANIFEST" <<'EOF'
import json, sys
entry = json.load(open(sys.argv[1]))["models"]["minilm"]
print(entry["repo"], entry["revision"])
EOF
)
# SPDX identifier declared by the pinned model card; the qualification pin
# lives in PREPARED_SOURCES (crates/fetch/src/lib.rs).
LICENSE="Apache-2.0"

# Hash-verify each fetched source against the committed manifest before
# handing it to provisioning. A partial or tampered fetch stops here.
python3 - "$MANIFEST" "$SOURCES" <<'EOF'
import hashlib, json, sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))["models"]["minilm"]
sources = Path(sys.argv[2])
problems = []
for entry in manifest["files"]:
    path = sources / entry["path"]
    if not path.is_file():
        problems.append(f"missing: {path}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        problems.append(f"sha256 mismatch: {path}")
if problems:
    print("error: prepared sources failed verification "
          "(re-run: cargo run -p inferstream-fetch -- --prepared minilm)",
          file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    raise SystemExit(1)
print("prepared sources verified against models/manifests/prepared-sources.json")
EOF

python3 "$SDK_PREFIX/bin/prepare-native-bundle.py" \
  --source-onnx "$SOURCES/model.onnx" \
  --tokenizer "$SOURCES/tokenizer.json" \
  --config "$SOURCES/config.json" \
  --model-card "$SOURCES/MODEL_CARD.md" \
  --model-id "$MODEL_ID" \
  --revision "$REVISION" \
  --license "$LICENSE" \
  --exporter "$SDK_PREFIX/bin/turboembed-export-model" \
  --output-dir "$OUTPUT"
echo "bundle ready: $OUTPUT"
