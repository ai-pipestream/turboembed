# Bundle design, candidate A (written fresh, before reading the previous one)

A bundle is a directory. It carries one model so the same program gives
the same answer on every machine. One manifest, `manifest.json`, names
every file with its hash, says what the model is, how to tokenize for it,
what it computes, and which artifact each backend runs.

## Directory

```
all-minilm-l6-v2/
  manifest.json
  tokenizer.json                 the upstream tokenizer file, unchanged
  model.safetensors              raw weights, fp32, for the CUDA and Metal kernels
  openvino/model.xml
  openvino/model.bin
  hailo10h/model.hef
  onnx/model.onnx                what the IR and HEF were compiled from; the fallback engine's input
  reference/inputs.txt           one input per line, UTF-8
  reference/embeddings.f32       [n_inputs, dim] float32, little-endian, row-major
```

Nothing outside the directory is referenced. Paths in the manifest are
relative and use `/`.

## Manifest

```json
{
  "format": 1,
  "id": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
  "source": {
    "kind": "huggingface",
    "repo": "sentence-transformers/all-MiniLM-L6-v2",
    "commit": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
    "licence": "Apache-2.0"
  },
  "task": "embed",
  "embed": {
    "dim": 384,
    "pooling": "mean",
    "normalize": "l2",
    "max_seq": 256,
    "max_batch": 64,
    "prefix_query": "",
    "prefix_document": ""
  },
  "tokenizer": {
    "kind": "wordpiece",
    "file": "tokenizer.json",
    "lowercase": true,
    "strip_accents": true,
    "truncate": "right",
    "pad": "[PAD]", "bos": "[CLS]", "eos": "[SEP]", "unk": "[UNK]"
  },
  "architecture": {
    "family": "bert",
    "layers": 6,
    "hidden": 384,
    "heads": 12,
    "intermediate": 1536,
    "activation": "gelu_erf",
    "layer_norm_eps": 1e-12,
    "max_position": 512,
    "type_vocab": 2,
    "vocab": 30522,
    "weight_names": "huggingface_bert"
  },
  "artifacts": [
    { "format": "safetensors", "file": "model.safetensors", "dtype": "f32",
      "backends": ["cuda", "metal"],
      "produced_by": "huggingface upload, unchanged" },
    { "format": "openvino_ir", "files": ["openvino/model.xml", "openvino/model.bin"], "dtype": "f32",
      "backends": ["openvino"],
      "produced_by": "openvino 2026.3.1 ovc, from onnx/model.onnx, container sha256:...", "reproducible": true },
    { "format": "hef", "file": "hailo10h/model.hef", "dtype": "a8w8", "target": "hailo10h", "seq": 128,
      "backends": ["hailo"],
      "produced_by": "hailo dataflow compiler 5.1.0, from onnx/model.onnx, calibration reference/inputs.txt", "reproducible": false },
    { "format": "onnx", "file": "onnx/model.onnx", "dtype": "f32",
      "backends": ["onnx"],
      "produced_by": "optimum 2.1 export, opset 17" }
  ],
  "reference": {
    "inputs": "reference/inputs.txt",
    "embeddings": "reference/embeddings.f32",
    "produced_by": "sentence-transformers 5.1.0, torch 2.9.0, cpu, fp32"
  },
  "files": {
    "tokenizer.json":        { "sha256": "…", "size": 711396 },
    "model.safetensors":     { "sha256": "…", "size": 90864192 },
    "openvino/model.xml":    { "sha256": "…", "size": 412331 },
    "openvino/model.bin":    { "sha256": "…", "size": 90862080 },
    "hailo10h/model.hef":    { "sha256": "…", "size": 24117248 },
    "onnx/model.onnx":       { "sha256": "…", "size": 90984734 },
    "reference/inputs.txt":  { "sha256": "…", "size": 4096 },
    "reference/embeddings.f32": { "sha256": "…", "size": 49152 }
  }
}
```

## Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `format` | int | yes | Manifest format version. The loader rejects any it does not know. |
| `id` | string | yes | Model identifier, shown in `turbo_model_info.model_id`. |
| `revision` | string | yes | Upstream commit or version, shown in `revision`. |
| `source.kind` | string | yes | `huggingface`, `local`, `url`. |
| `source.repo`, `source.commit` | string | when kind is huggingface | Where the checkpoint came from. |
| `source.licence` | string | yes | SPDX identifier. Nothing is redistributed without one. |
| `task` | string | yes | `embed` for now. One task per bundle. |
| `embed.dim` | int | yes | Vector width. |
| `embed.pooling` | string | yes | `mean`, `cls`, `last`. |
| `embed.normalize` | string | yes | `l2` or `none`. |
| `embed.max_seq` | int | yes | Tokens per row the model was trained for. |
| `embed.max_batch` | int | yes | Rows the model is qualified for in one run. |
| `embed.prefix_query`, `embed.prefix_document` | string | yes, may be empty | Prepended when the caller asks for that role. |
| `tokenizer.kind` | string | yes | `wordpiece`, `bpe`, `unigram`. |
| `tokenizer.file` | string | yes | The upstream tokenizer file, byte for byte. Its hash is `tokenizer_sha256`. |
| `tokenizer.lowercase`, `strip_accents` | bool | yes | The normalizer settings the native tokenizer applies. |
| `tokenizer.truncate` | string | yes | `right` or `left`: what `TURBO_TRUNCATE_MODEL` means. |
| `tokenizer.pad`, `bos`, `eos`, `unk` | string | yes | Special token strings; ids are looked up in the file. |
| `architecture.*` | object | when an artifact is `safetensors` | Everything a kernel path needs that the weights file does not carry. `weight_names` names the tensor naming scheme. |
| `artifacts[].format` | string | yes | `safetensors`, `openvino_ir`, `hef`, `gguf`, `onnx`. |
| `artifacts[].file` or `files` | string or list | yes | Relative paths, each present in `files`. |
| `artifacts[].dtype` | string | yes | `f32`, `f16`, `bf16`, or a quantization label such as `a8w8`. |
| `artifacts[].backends` | list | yes | Backend ids that can load it. |
| `artifacts[].target` | string | hef only | `hailo8`, `hailo8l`, `hailo10h`. Matched against the device's architecture label. |
| `artifacts[].seq` | int | hef only | The fixed sequence length compiled in. |
| `artifacts[].produced_by` | string | yes | Tool, version, input, and the container digest when one was used. |
| `artifacts[].reproducible` | bool | optional | Whether running `produced_by` again yields the same bytes. |
| `reference.inputs`, `reference.embeddings` | string | yes | The numeric check. |
| `reference.produced_by` | string | yes | What computed the reference vectors. |
| `files` | map | yes | Every file in the directory except the manifest, with SHA-256 and size. |

## Loader rules

1. Read `manifest.json`. Unknown `format`: `TURBO_E_BUNDLE_INVALID`. A
   missing required field, a value outside its set, a path that leaves
   the directory, or an artifact path absent from `files`:
   `TURBO_E_BUNDLE_INVALID` with the field named in the message.
2. Every file in `files` must exist: otherwise `TURBO_E_BUNDLE_NOT_FOUND`
   naming it. A file is hashed when it is opened, before any byte is used;
   a mismatch is `TURBO_E_BUNDLE_INTEGRITY` naming the file. A file the
   loader does not open is not hashed (the ONNX file on a machine that
   runs the safetensors).
3. Artifact choice: the backend of the context's device takes the first
   artifact whose `backends` contains its id and, for `hef`, whose
   `target` equals the device's architecture label. None:
   `TURBO_E_BUNDLE_NO_ARTIFACT`, message listing the formats present.
4. The artifact's hash becomes `turbo_model_info.artifact_sha256`; for a
   two-file artifact, the hash of the first file. The tokenizer file's
   hash becomes `tokenizer_sha256`.
5. `max_seq` and `max_batch` are limits, not defaults: a session larger
   than either is `TURBO_E_CAPACITY`. A HEF's `seq` caps `max_seq` on
   that device, and the cell says so.
6. The tokenizer is built from `tokenizer.file` plus the manifest's
   normalizer fields. If the file's own normalizer disagrees with the
   manifest, the bundle is invalid: there is one truth.

## Left out, and why

- No model kind, modality, or list of tasks. One task per bundle keeps
  the contract flat; a reranker is a different bundle.
- No per-backend options. A backend that needs a knob reads it from the
  artifact entry, and the knob is named in the manifest, not passed as a
  string at load time.
- No signatures, mirrors or download locations. The bundle is a
  directory; how it arrived is not its business.
- No labels, chat templates, generation settings. They come with the
  tasks that need them.
- No file for the architecture separate from the manifest. One file to
  hash, one file to read.

## Open questions

1. Should `max_batch` live in the bundle at all, or is it a device fact?
   I put it in because the reference vectors were produced at some
   batch and numerics can drift with batch on some kernels.
2. `reference.embeddings` at fp32 for a quantized HEF: the cosine floor
   is the cell's, not the bundle's, so one reference serves all. Agreed?
3. Is `source.licence` enough, or do we need the licence text in the
   directory for redistribution?
