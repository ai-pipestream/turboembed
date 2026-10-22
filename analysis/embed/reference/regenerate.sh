#!/usr/bin/env bash
# Regenerates analysis/embed/tests/data/sentence_vectors_dl_golden.json from
# testdata/tiny-bert-bundle/weights/model.safetensors and verifies that the
# regenerated file matches the checked-in fixture byte-for-byte.
#
# Run from the repository root. Uses the two pinned containers:
#   - turbo-reference (built with: docker build -t turbo-reference bundle/reference)
#   - turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1
#     (built with: cargo run -p turbo-goldens-tool -- image)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

ref_image="${TURBO_REFERENCE_IMAGE:-turbo-reference:latest}"
java_image="${TURBO_GOLDENS_IMAGE:-turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1}"
opennlp_commit="cdf33c653d5d61d5af7d37ffee189a83bd8392ea"
expected_mean_sha256="57ec273eae030d2c810d2e1df214684f49d9c0b81da08cc52de952aa985fed49"
expected_cls_sha256="7dfd15ea3e880ed3a6808df03b442b07a1d84cd55afeadd758a6108338176f0d"

work="target/sentence-vectors-dl-reference"
rm -rf "$work"
mkdir -p "$work/classes"
work_abs="$(cd "$work" && pwd)"

user_arg=()
if command -v id >/dev/null 2>&1; then
  user_arg=(--user "$(id -u):$(id -g)")
fi

# 1. Export mean_l2.onnx and cls.onnx from testdata/tiny-bert-bundle/weights/model.safetensors
#    in the reference container with --network none.
docker run --rm --network none --entrypoint python3 "${user_arg[@]}" \
  --mount "type=bind,src=$repo_root,dst=/repo,readonly" \
  --mount "type=bind,src=$work_abs,dst=/work" \
  "$ref_image" \
  /repo/analysis/embed/reference/export_onnx.py \
  /repo/testdata/tiny-bert-bundle/weights/model.safetensors \
  /work/mean_l2.onnx \
  /work/cls.onnx

mean_sha="$(sha256sum "$work/mean_l2.onnx" | awk '{print $1}')"
cls_sha="$(sha256sum "$work/cls.onnx" | awk '{print $1}')"
test "$mean_sha" = "$expected_mean_sha256" || {
  echo "mean_l2.onnx sha256 mismatch: got $mean_sha, expected $expected_mean_sha256" >&2
  exit 1
}
test "$cls_sha" = "$expected_cls_sha256" || {
  echo "cls.onnx sha256 mismatch: got $cls_sha, expected $expected_cls_sha256" >&2
  exit 1
}

# 2. Compile and run SentenceVectorsDlGolden in the pinned goldens JVM container
#    against OpenNLP's opennlp-dl and ONNX Runtime jars with --network none.
#    No --user flag is passed here so the container's JDK can read /root/.m2.
docker run --rm --network none --entrypoint sh \
  --mount "type=bind,src=$repo_root,dst=/repo,readonly" \
  --mount "type=bind,src=$work_abs,dst=/work" \
  "$java_image" -c "
    set -eu
    test \"\$(cat /opennlp.commit)\" = \"$opennlp_commit\"
    m2=/root/.m2/repository
    cp=\"/generator/target/lib/*:\$m2/org/apache/opennlp/opennlp-dl/3.0.0-SNAPSHOT/opennlp-dl-3.0.0-SNAPSHOT.jar:\$m2/com/microsoft/onnxruntime/onnxruntime/1.29.0/onnxruntime-1.29.0.jar\"
    javac -encoding UTF-8 -cp \"\$cp\" -d /work/classes /repo/analysis/embed/reference/SentenceVectorsDlGolden.java
    java -cp \"/work/classes:\$cp\" SentenceVectorsDlGolden \
      /work/mean_l2.onnx \
      /work/cls.onnx \
      /repo/analysis/goldens/data/fixtures/all-minilm-l6-v2-vocab.txt \
      /repo/testdata/tiny-bert-bundle/weights/model.safetensors \
      /repo/analysis/goldens/data/inputs \
      \"$opennlp_commit\" \
      /work/sentence_vectors_dl_golden.json
  "

if [[ "${1:-}" == "--write" ]]; then
  cp "$work/sentence_vectors_dl_golden.json" analysis/embed/tests/data/sentence_vectors_dl_golden.json
fi

cmp analysis/embed/tests/data/sentence_vectors_dl_golden.json "$work/sentence_vectors_dl_golden.json"
echo "verified: analysis/embed/tests/data/sentence_vectors_dl_golden.json matches regenerated output (mean_l2.onnx=$mean_sha, cls.onnx=$cls_sha)"
