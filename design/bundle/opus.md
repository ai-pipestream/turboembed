# Model bundle

The manifest is the proto3 JSON encoding of a `Bundle` message: snake_case
names, enums by value name (the header constant without `TURBO_`), lists
not maps except `tensor_names`. Absent means the proto3 default.

## 1. Layout

```
all-minilm-l6-v2/
  manifest.json
  LICENSE, tokenizer.json        upstream, unchanged
  weights/model.safetensors      upstream, unchanged
  onnx/model.onnx                upstream, unchanged
  openvino/model.xml, model.bin  converted
  hailo/model-hailo10h-s128.hef  compiled
  calibration/texts.txt          Hailo quantization inputs
  reference/reference.safetensors
```

Paths are bundle-relative, use `/`, never `..` or a leading `/`.

## 2. Example

Hashes, sizes, digests, commit and versions are placeholders.

```json
{
  "bundle_version": 1,
  "model": {
    "id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "3",
    "source": {
      "repository": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
      "commit": "<40-hex upstream commit>"
    },
    "license": "Apache-2.0",
    "license_file": "LICENSE"
  },
  "task": "TASK_EMBED",
  "embed": {
    "dim": 384,
    "pooling": "POOLING_MEAN",
    "normalize": "NORMALIZE_L2",
    "max_seq": 256,
    "max_batch": 256,
    "prefix_query": "",
    "prefix_document": "",
    "output_dims": []
  },
  "tokenizer": {
    "file": "tokenizer.json",
    "normalizer": {
      "clean_text": true,
      "lowercase": true,
      "strip_accents": true,
      "split_cjk": true,
      "unicode_form": "UNICODE_NONE"
    },
    "wordpiece": { "continuing_prefix": "##", "max_chars_per_word": 100 },
    "special_tokens": [
      { "role": "SPECIAL_PAD",  "content": "[PAD]",  "id": 0 },
      { "role": "SPECIAL_UNK",  "content": "[UNK]",  "id": 100 },
      { "role": "SPECIAL_BOS",  "content": "[CLS]",  "id": 101 },
      { "role": "SPECIAL_EOS",  "content": "[SEP]",  "id": 102 },
      { "role": "SPECIAL_MASK", "content": "[MASK]", "id": 103 }
    ],
    "template": ["[CLS]", "$TEXT", "[SEP]"],
    "truncation": "TRUNCATE_RIGHT"
  },
  "architecture": {
    "family": "FAMILY_BERT",
    "layers": 6,
    "hidden": 384,
    "heads": 12,
    "intermediate": 1536,
    "activation": "ACTIVATION_GELU_ERF",
    "layer_norm_eps": 1e-12,
    "position_embedding": "POSITION_ABSOLUTE",
    "max_positions": 512,
    "token_types": 2,
    "vocab_size": 30522
  },
  "artifacts": [
    {
      "name": "weights-f32",
      "format": "FORMAT_SAFETENSORS",
      "files": ["weights/model.safetensors"],
      "backends": ["cuda", "metal"],
      "graph_input": "INPUT_TOKEN_IDS",
      "graph_output": "OUTPUT_HIDDEN_STATES",
      "min_cosine": 0.999,
      "tensor_names": {
        "word_embeddings": "embeddings.word_embeddings.weight",
        "position_embeddings": "embeddings.position_embeddings.weight",
        "token_type_embeddings": "embeddings.token_type_embeddings.weight",
        "embeddings_ln_weight": "embeddings.LayerNorm.weight",
        "embeddings_ln_bias": "embeddings.LayerNorm.bias",
        "q_weight": "encoder.layer.{layer}.attention.self.query.weight",
        "q_bias": "encoder.layer.{layer}.attention.self.query.bias",
        "k_weight": "encoder.layer.{layer}.attention.self.key.weight",
        "k_bias": "encoder.layer.{layer}.attention.self.key.bias",
        "v_weight": "encoder.layer.{layer}.attention.self.value.weight",
        "v_bias": "encoder.layer.{layer}.attention.self.value.bias",
        "attn_out_weight": "encoder.layer.{layer}.attention.output.dense.weight",
        "attn_out_bias": "encoder.layer.{layer}.attention.output.dense.bias",
        "attn_ln_weight": "encoder.layer.{layer}.attention.output.LayerNorm.weight",
        "attn_ln_bias": "encoder.layer.{layer}.attention.output.LayerNorm.bias",
        "ffn_in_weight": "encoder.layer.{layer}.intermediate.dense.weight",
        "ffn_in_bias": "encoder.layer.{layer}.intermediate.dense.bias",
        "ffn_out_weight": "encoder.layer.{layer}.output.dense.weight",
        "ffn_out_bias": "encoder.layer.{layer}.output.dense.bias",
        "ffn_ln_weight": "encoder.layer.{layer}.output.LayerNorm.weight",
        "ffn_ln_bias": "encoder.layer.{layer}.output.LayerNorm.bias"
      }
    },
    {
      "name": "openvino-f16",
      "format": "FORMAT_OPENVINO_IR",
      "files": ["openvino/model.xml", "openvino/model.bin"],
      "backends": ["openvino"],
      "device_kinds": ["DEVICE_CPU", "DEVICE_GPU", "DEVICE_IGPU"],
      "compute_dtype": "DTYPE_F16",
      "graph_input": "INPUT_TOKEN_IDS",
      "graph_output": "OUTPUT_HIDDEN_STATES",
      "min_cosine": 0.999,
      "produced_by": {
        "tool": "ovc",
        "tool_version": "<version>",
        "container": "ghcr.io/<org>/openvino-tools@sha256:<digest>",
        "from": "onnx-f32",
        "args": ["onnx/model.onnx", "--compress_to_fp16=True",
                 "--output_model", "openvino/model.xml"],
        "reproducible": true
      }
    },
    {
      "name": "hef-hailo10h-s128",
      "format": "FORMAT_HEF",
      "files": ["hailo/model-hailo10h-s128.hef"],
      "backends": ["hailo"],
      "target": "hailo10h",
      "fixed_seq": 128,
      "compute_dtype": "DTYPE_I8",
      "graph_input": "INPUT_EMBEDDINGS",
      "host_weights": "weights-f32",
      "graph_output": "OUTPUT_HIDDEN_STATES",
      "min_cosine": 0.98,
      "produced_by": {
        "tool": "hailo-dataflow-compiler",
        "tool_version": "<version>",
        "container": "ghcr.io/<org>/hailo-dfc@sha256:<digest>",
        "from": "onnx-f32",
        "inputs": ["calibration/texts.txt"],
        "args": ["--hw-arch", "hailo10h", "--seq", "128",
                 "--calib", "calibration/texts.txt"],
        "reproducible": false
      }
    },
    {
      "name": "onnx-f32",
      "format": "FORMAT_ONNX",
      "files": ["onnx/model.onnx"],
      "backends": [],
      "graph_input": "INPUT_TOKEN_IDS",
      "graph_output": "OUTPUT_HIDDEN_STATES"
    }
  ],
  "reference": {
    "file": "reference/reference.safetensors",
    "cases": [
      { "text": "", "prompt_role": "PROMPT_NONE" },
      { "text": "The quick brown fox jumps over the lazy dog.", "prompt_role": "PROMPT_NONE" },
      { "text": "Café naïve RÉSUMÉ", "prompt_role": "PROMPT_NONE" },
      { "text": "东京是日本的首都。", "prompt_role": "PROMPT_NONE" },
      { "text": "emoji 🙂 and tabs\tand\nnewlines", "prompt_role": "PROMPT_NONE" },
      { "text": "how do I reset a password", "prompt_role": "PROMPT_QUERY" },
      { "text": "To reset a password, open Settings and choose Security.", "prompt_role": "PROMPT_DOCUMENT" },
      { "text": "<a paragraph of about 120 tokens>", "prompt_role": "PROMPT_NONE" }
    ],
    "produced_by": {
      "tool": "sentence-transformers",
      "tool_version": "<version>",
      "container": "ghcr.io/<org>/st-reference@sha256:<digest>",
      "args": ["--device", "cpu", "--dtype", "float32"],
      "reproducible": false
    }
  },
  "files": [
    { "path": "LICENSE", "size": 11357, "sha256": "<64 hex>" },
    { "path": "tokenizer.json", "size": 466247, "sha256": "<64 hex>" },
    { "path": "weights/model.safetensors", "size": 90868376, "sha256": "<64 hex>" },
    { "path": "onnx/model.onnx", "size": 90405214, "sha256": "<64 hex>" },
    { "path": "openvino/model.xml", "size": 412345, "sha256": "<64 hex>" },
    { "path": "openvino/model.bin", "size": 45212345, "sha256": "<64 hex>" },
    { "path": "hailo/model-hailo10h-s128.hef", "size": 31234567, "sha256": "<64 hex>" },
    { "path": "calibration/texts.txt", "size": 204800, "sha256": "<64 hex>" },
    { "path": "reference/reference.safetensors", "size": 16384, "sha256": "<64 hex>" }
  ]
}
```

## 3. Fields

| Field | Type | Req | Meaning |
|---|---|---|---|
| bundle_version | uint32 | yes | Schema version, 1. |
| model.id, .revision | string | yes | For `model_info`; revision changes with any file. |
| model.source.repository, .commit, .license | string | yes | Origin; SPDX id. |
| model.license_file | path | no | Licence text. |
| task | enum | yes | `TASK_EMBED`; names the task block present. |
| embed.dim, .pooling, .normalize | uint32, enum | yes | Length; defaults for options. |
| embed.max_seq, .max_batch | uint32 | yes | Tokens per row with specials; largest batch checked. |
| embed.prefix_query, .prefix_document | string | no | Prepended per role. |
| embed.output_dims | uint32[] | no | Trained cut lengths; others refused. |
| tokenizer.file | path | yes | Vocabulary source; `tokenizer_sha256`. |
| tokenizer.normalizer.* | bool, enum | yes | Applied in listed order. |
| tokenizer.wordpiece, bpe, unigram | message | one | Kind and parameters. |
| tokenizer.special_tokens[] | role, content, id | yes | Fills pad, bos, eos, unk ids. |
| tokenizer.template, .truncation | string[], enum | yes | Row layout; `TRUNCATE_MODEL`. |
| architecture.* | | raw weights | What kernels need that safetensors lacks. |
| artifacts[].name | string | yes | Unique; referenced by `from`, `host_weights`. |
| .format | enum | yes | SAFETENSORS, OPENVINO_IR, HEF, GGUF, ONNX. |
| .files | path[] | yes | First file gives `artifact_sha256`. |
| .backends | string[] | yes | `turbo_device_info.backend` values; empty never loads. |
| .device_kinds, .target | enum[], string | no | Device kinds; compiler target. Empty is any. |
| .fixed_seq, .fixed_batch | uint32 | no | Compiled shape; 0 is dynamic. |
| .compute_dtype | enum | no | Fixed by compilation; else the backend's. |
| .graph_input, .graph_output | enum | yes | Which stages the backend adds. |
| .host_weights | string | if embeddings in | Embedding tensors for the host. |
| .tensor_names | map | safetensors | Role to name; `{layer}` is the index. |
| .min_cosine | float | no | Floor against the reference. |
| .produced_by | message | no | Absent: upstream file unchanged. |
| produced_by.tool, .tool_version, .container, .reproducible | string, bool | yes | What ran; two runs gave identical bytes. |
| produced_by.from, .inputs, .args | | no | Source, files read, arguments. |
| files[] | path, uint64, hex | yes | Every file, size, SHA-256. |

**Reference.** `reference.safetensors` holds `ids` (I32 `[n, L]`, padded
with `pad_id`), `lengths` (I32 `[n]`) and `embeddings` (F32 `[n, dim]`),
row i for `cases[i]`: upstream's full pipeline, fp32 on CPU, in the pinned
container. The core already parses safetensors. Cases fit every `fixed_seq`.

## 4. Loader rules

Codes are `TURBO_E_*`.

1. No directory or manifest: `BUNDLE_NOT_FOUND`.
2. Unknown field or enum, missing field, string over its header buffer,
   bad path, path not in `files[]`: `BUNDLE_INVALID`. Unbuilt task:
   `UNSUPPORTED_TASK`.
3. Verify tokenizer and reference files: size, then SHA-256.
4. Encode the reference cases; compare with `ids` exactly. Any difference
   is `BUNDLE_INVALID` naming case and position. `turbo_tokenizer_create`
   stops here.
5. Select the first artifact, in manifest order, whose `backends` holds
   the device's backend, whose `device_kinds` and `target` match or are
   empty, and whose format and family the backend implements. None:
   `BUNDLE_NO_ARTIFACT`, saying why each was skipped.
6. Verify the chosen artifact's files and its `host_weights`, hashing the
   bytes handed to the backend where its API takes memory.
7. Check every templated tensor exists with the shape `architecture`
   implies (`[out, in]` for linear), else `BUNDLE_INVALID`.
8. Vendor load failure: `RUNTIME` with its text.

A size or hash mismatch or missing file is `BUNDLE_INTEGRITY`, naming the
path and both values. Nothing loads and no other artifact is tried.
Unlisted files are never opened.

`model_info.max_seq` and `max_batch` report a smaller fixed shape.
`TRUNCATE_MODEL` still cuts at `embed.max_seq`, so a longer row fails with
`CAPACITY`, never cut differently on one device. Asking for more is
`UNSUPPORTED_OPTION`.

## 5. Left out

- Signatures: hashes catch corruption; signing is distribution's job.
- Benchmark results: per machine and commit (rule 2).
- Machine tuning, names, cache paths (rule 8).
- Selection scores: manifest order suffices.
- A general tokenizer pipeline; BPE and unigram fields wait (rule 9).
- Pair templates, other tasks, dates, authors: nothing reads them.

## 6. Open questions

1. The README says ONNX never runs. Is there a fallback engine?
2. The header has no int8 dtype. What does a HEF report?
3. Multi-file `artifact_sha256`: first file, or hash of hashes?
4. Hashing large GGUF files every load: allow a cache?
5. Can the Hailo-10H embed on chip?
6. Tokenizer check on every load, or only in tests?
7. Add a reference case over `max_seq` for truncation?
8. May bundles redistribute upstream weights under each licence?
