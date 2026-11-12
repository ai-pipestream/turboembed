# Hailo graph comparison validation

Part of #1982. This receipt covers the CPU graph comparison prerequisite,
not a compiled HEF, quantized NER accuracy, or device execution.

## Regression

At main `b82dba33bc7e7a3dafc4de7aa1762df2292e1eca`, `check_cut`
can report zero error for NaN outputs because `max(0.0, nan)` is zero.
It also accepts equally wrong output widths and silently discards extra
batch entries. Empty calibration can pass without any comparison.

The new `bundle/hailo/test_cut_validation.py` builds real ONNX graphs and
runs both ONNX Runtime sessions. Against the original script, its 11 tests
produce 2 passes, 8 failures and 1 error. With the corrected script, all
11 pass in both compiler images below. The finite-difference threshold
remains `1e-4`; differences are computed in float64 after rejecting
nonfinite values. Each session uses two CPU threads and one inter-op
thread. No accelerator execution is claimed by these tests.

| Environment | Pinned image |
| --- | --- |
| DFC 5.4 | `turbo-hailo-dfc@sha256:4bdfc1c1e40727e6c4366c9123de65700633556e476b17a435dc8ad46f653a72` |
| DFC 3.x | `turbo-hailo-dfc3@sha256:aca2fd480f74d9173beba1b7af5a05702d76169d45070c93100653908e970f12` |

Run from the repository root, setting `IMAGE` to either image above:

```sh
docker run --rm --network none --cpus 2 --memory 4g \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/bundle/hailo:/tests:ro" -w /tests \
  --entrypoint python "$IMAGE" -m unittest -v test_cut_validation
```

For the red run, use the same test file with `hef_compile.py` from the
main commit above in a separate directory and mount that directory at
`/tests`. Both Dockerfiles now run this suite when building a new image.
Existing recipe image digests are unchanged and retain their old script.
A mounted script test does not rebuild those images.

## Real BERT encoder check

The privately cached `dslim/bert-base-NER` source revision is
`d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`. Its upstream ONNX hash was
checked against the pinned revision's LFS metadata. It exports nine
classifier logits, which are not the hidden states the Hailo backend
requires. The feasibility probe extracted the input of the final
classifier MatMul, preserving all 12 encoder layers and 768 hidden
channels. The transposed classifier weight and bias are byte-identical
to the pinned safetensors head.

Reconstructing logits with those unchanged classifier tensors on four
texts, including empty input and Unicode, gives maximum absolute error
`9.5367431640625e-6` against the original ONNX graph. This is an extraction
check, not a held-out NER evaluation.

The extracted encoder and its fixed-shape Hailo input adaptation were
then compared on all 180 existing calibration texts, batch 1, sequence
128, 12 attention heads. The updated validator checks exact output shape
`[1, 128, 768]` and finite values on every kept position. Maximum absolute
difference is `0.0`. Padding remains outside the comparison, as before.

| Artifact | SHA-256 |
| --- | --- |
| Original ONNX | `963039b81eec5b33e23d84826ccdf1e8f8ada776f320e692113034cfae384617` |
| Extracted encoder ONNX | `c3116aef7e29f4c541d76dccc7da41bb9b69fb7a9df802e5ca6f7f4c6c9aca10` |
| Fixed-shape encoder ONNX | `2d1fa467e39ddfd47b76cf6fde7a23a4d7f45da0e130bc9aff9efb124833ea29` |
| Calibration texts | `bde646a9a523e6dbb93315343bd363890ef31510625aee4a2e8c7a6fda65c3b4` |

Raw logs, source hashes, extraction and recheck scripts, and private model
artifacts are retained in the NAS evidence collection
`turboembed/1982-hailo10h-ner`, outside Cargo targets. No model or corpus
text is distributed by this change. #1982 remains open for compiler
placement, quantization quality, matched CPU/Hailo chunking, and real
Hailo-10H Word/Piece entity and offset parity.
