# The model bundle

A bundle is a directory that carries one model so the same program gives
the same answer on every machine. One manifest, `manifest.json`, names
every file with its size and hash, says what the model is, exactly how
to tokenize for it, what it computes, which artifact each backend runs,
and carries a small reference the loader checks itself against on every
machine.

## Layout

```
all-minilm-l6-v2/
  manifest.json
  LICENSE                          upstream, unchanged
  tokenizer.json                   upstream, unchanged
  weights/model.safetensors        upstream, unchanged
  onnx/model.onnx                  upstream export; what the IR and HEF were compiled from
  openvino/model.xml, model.bin    converted
  hailo/model-hailo10h-s128.hef    compiled
  calibration/texts.txt            quantization inputs the HEF was calibrated with
  reference/reference.safetensors  ids, lengths and fp32 vectors for the reference cases
```

Paths in the manifest are relative to the directory, use `/`, and never
contain `..` or a leading `/`.

## Manifest

The manifest is JSON in one canonical form, the one the bundle tool
writes: field names in snake_case, enum values as the header constant
without the `TURBO_` prefix, 64-bit integers as JSON numbers, lists
rather than maps except `tensor_names`. A field the table below marks
required is present even when its value is empty, false or 0; a field
marked "no" may be left out. Nothing else is accepted: no lowerCamelCase
names, no integers as strings, no field filled in from a default. A
service that carries a `Bundle` message in another encoding converts to
this form before it is written or loaded.

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
    "max_batch": 64,
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
      "compute_dtype": "DTYPE_F16",
      "graph_input": "INPUT_TOKEN_IDS",
      "graph_output": "OUTPUT_HIDDEN_STATES",
      "produced_by": {
        "tool": "ovc",
        "tool_version": "<version>",
        "container": "<registry>/openvino-tools@sha256:<digest>",
        "from": "onnx-f32",
        "args": ["onnx/model.onnx", "--compress_to_fp16=True", "--output_model", "openvino/model.xml"],
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
      "produced_by": {
        "tool": "hailo-dataflow-compiler",
        "tool_version": "<version>",
        "container": "<registry>/hailo-dfc@sha256:<digest>",
        "from": "onnx-f32",
        "inputs": ["calibration/texts.txt"],
        "args": ["--hw-arch", "hailo10h", "--seq", "128", "--calib", "calibration/texts.txt"],
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
      { "text": "<a paragraph of about 120 tokens>", "prompt_role": "PROMPT_NONE" },
      { "text": "<a paragraph longer than max_seq tokens>", "prompt_role": "PROMPT_NONE" }
    ],
    "produced_by": {
      "tool": "sentence-transformers",
      "tool_version": "<version>",
      "container": "<registry>/st-reference@sha256:<digest>",
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

## Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `bundle_version` | uint32 | yes | Schema version, 1. Anything else is rejected. |
| `model.id`, `model.revision` | string | yes | Shown in `turbo_model_info`. The revision changes whenever any file changes. |
| `model.source.repository`, `.commit` | string | yes | Where the checkpoint came from. |
| `model.license` | string | yes | SPDX identifier. Nothing is redistributed without one. |
| `model.license_file` | path | no | The licence text. |
| `task` | enum | yes | `TASK_EMBED`. Names the task block that follows. Any other `TASK_*` name is a task this build does not have. |
| `embed.dim`, `.pooling`, `.normalize` | uint32, enum | yes | Vector width; what `TURBO_POOLING_MODEL` and `TURBO_NORMALIZE_MODEL` mean. With an `architecture`, `dim` equals `architecture.hidden`: the vectors are pooled hidden states. |
| `embed.max_seq` | uint32 | yes | Tokens per row including specials, the length the model was evaluated at. Never the positional table size. |
| `embed.max_batch` | uint32 | yes | The largest batch the reference was checked at. A session larger than it is refused. |
| `embed.prefix_query`, `.prefix_document` | string | no | Prepended for `TURBO_PROMPT_QUERY` and `TURBO_PROMPT_DOCUMENT`. |
| `embed.output_dims` | uint32[] | no | The widths the model was trained to be cut to. Any other `output_dim` is refused. The cut comes before normalize: an L2-normalized vector is unit length at `output_dim`. |
| `tokenizer.file` | path | yes | The upstream tokenizer file, unchanged. Its hash is `tokenizer_sha256`. |
| `tokenizer.normalizer.*` | bool, enum | yes | What the core applies to the text. The order is upstream BertNormalizer's, whatever the order of the fields: clean, split CJK, strip accents, lowercase. |
| `tokenizer.wordpiece`, `.bpe`, `.unigram` | message | one of | The kind and its parameters. Only wordpiece is defined in this cut. |
| `tokenizer.special_tokens[]` | role, content, id | yes | Fills `pad_id`, `bos_id`, `eos_id`, `unk_id`. |
| `tokenizer.template` | string[] | yes | The row layout around `$TEXT`. |
| `tokenizer.truncation` | enum | yes | What `TURBO_TRUNCATE_MODEL` means: `TRUNCATE_RIGHT` or `TRUNCATE_LEFT`. `TRUNCATE_NONE` is a caller option and is rejected here. |
| `architecture.*` | message | when an artifact is raw weights | Everything a kernel path needs that a weights file does not carry. |
| `artifacts[].name` | string | yes | Unique; referenced by `from` and `host_weights`. |
| `artifacts[].format` | enum | yes | `FORMAT_SAFETENSORS`, `FORMAT_OPENVINO_IR`, `FORMAT_HEF`, `FORMAT_GGUF`, `FORMAT_ONNX`. |
| `artifacts[].files` | path[] | yes | Each listed in `files`. |
| `artifacts[].backends` | string[] | yes | `turbo_device_info.backend` values that load it. Empty: nothing loads it, as for an ONNX file carried only for the reference programs and the converters. |
| `artifacts[].target` | string | compiled artifacts | The device architecture label the artifact was compiled for. Matched against `turbo_device_info.arch`. |
| `artifacts[].fixed_seq`, `.fixed_batch` | uint32 | no | The shape compiled in; 0 is dynamic. |
| `artifacts[].compute_dtype` | enum | no | Fixed by the compilation, so never on `FORMAT_SAFETENSORS`. Absent: the session's `precision` decides, and `TURBO_PRECISION_MODEL` computes in the dtype the weights are stored in. |
| `artifacts[].graph_input`, `.graph_output` | enum | yes | Where the artifact starts and stops, so the backend knows which stages it must add. Raw weights (`FORMAT_SAFETENSORS`) start at `INPUT_TOKEN_IDS`. |
| `artifacts[].host_weights` | string | when input is embeddings | The artifact whose embedding tensors the host lookup uses. |
| `artifacts[].tensor_names` | map | raw weights | Role to tensor name; `{layer}` is the layer index. |
| `artifacts[].produced_by` | message | no | Absent means the upstream file, unchanged. |
| `produced_by.tool`, `.tool_version`, `.container`, `.reproducible` | string, bool | yes when present | What ran, in which pinned container, and whether two runs give identical bytes. |
| `produced_by.from` | string | yes in an artifact, empty in `reference` | The artifact this one was converted from. The reference is made from the upstream model, so its `from` is empty. |
| `produced_by.inputs`, `.args` | path[], string[] | no | The other files read, the arguments. |
| `reference.file` | path | yes | A safetensors file with `ids` (I32 `[n, L]`, padded with `pad_id`), `lengths` (I32 `[n]`) and `embeddings` (F32 `[n, dim]`), row `i` for `cases[i]`. |
| `reference.cases[]` | text, prompt_role | yes | The exact bytes the core is handed. If a service normalizes text upstream, these are post-normalization. One case is longer than `max_seq`, so truncation is checked too. |
| `reference.produced_by` | message | yes | The upstream pipeline, fp32 on CPU, in a pinned container. |
| `files[]` | path, uint64, hex | yes | Every file in the directory except the manifest, with size and SHA-256. |

## Loader rules

Status codes are the header's `TURBO_E_*`.

1. No directory or no manifest: `BUNDLE_NOT_FOUND`.
2. The manifest is parsed strictly. An unknown field or enum value at
   any level, a missing required field, a string longer than its header
   buffer, a path with `..` or a leading `/`, or a path not present in
   `files`: `BUNDLE_INVALID`, naming the field. Any `TASK_*` name other
   than the ones this build has: `UNSUPPORTED_TASK`.
3. Every path is canonicalized and must resolve under the bundle
   directory's canonical path. A symlink that leaves the directory is
   `BUNDLE_INVALID`. A hash never vouches for a file elsewhere on disk.
4. The manifest bytes are hashed. That hash identifies the contract and
   is reported beside the artifact and tokenizer hashes, as
   `manifest_sha256` in `turbo_tokenizer_info`, `turbo_model_info` and
   `turbo_result_info`.
5. The tokenizer file and the reference file are verified by size, then
   SHA-256. Then the core encodes every reference case and compares the
   ids exactly with the reference's. Any difference is `BUNDLE_INVALID`
   naming the case and the position. `turbo_tokenizer_create` stops
   here.
6. The artifact is the first one, in manifest order, whose `backends`
   contains the device's backend, whose `target` is empty or equals the
   device's architecture label, and whose format and family the backend
   implements. Manifest order is the preference. None:
   `BUNDLE_NO_ARTIFACT`, with the message saying why each was skipped.
   A session's `precision` never picks another artifact; it says how the
   chosen one computes.
7. The chosen artifact's files, and its `host_weights` artifact's files,
   are verified by size, then SHA-256, before any byte is used. Where a
   backend's API takes memory, the bytes handed to it are the bytes
   hashed.
8. For raw weights, every tensor the `tensor_names` map implies must
   exist with the shape the architecture implies (`[out, in]` for a
   linear layer), be `F32`, `F16` or `BF16` with every such tensor the
   same, and start at a byte offset in its file that is a multiple of its
   element size; each file's header length is a multiple of 8. Else
   `BUNDLE_INVALID` naming the tensor or the file. Tensors the model does
   not use are not checked.
9. A vendor load failure is `RUNTIME` with the vendor's text.

A size mismatch or a hash mismatch is `BUNDLE_INTEGRITY`, naming the
path and both values. A listed file that is absent is
`BUNDLE_NOT_FOUND`, naming the path, as for a bundle shipped without
its weights (below). Either way nothing loads and no other artifact is
tried. A file not listed in `files` is never opened.

On a fixed-shape artifact, `turbo_model_info` reports the smaller
`max_seq` and `max_batch`. `TURBO_TRUNCATE_MODEL` still cuts at
`embed.max_seq`, so a row that fits the model but not the artifact fails
with `CAPACITY` rather than being cut differently on one device.

The artifact hash reported for a multi-file artifact is the SHA-256 of
the files' hashes, as the 64 hex digits `files` gives each, concatenated
in listed order with nothing between.

## Bundles without their weights

When a licence forbids redistribution, the bundle ships as a recipe: the
manifest is complete, including the sizes and hashes the files must
have, and the files are absent. Opening it is `BUNDLE_NOT_FOUND` naming
the first absent file and the bundle tool. The tool fetches the source
at the recorded commit, runs each `produced_by` in its container, and
refuses to finish unless every hash matches.

## Decisions taken with the design

- ONNX is a file in the bundle because the IR and the HEF are compiled
  from it and the manifest records that, and because the benchmark's
  reference programs (TensorRT's `trtexec`, OpenVINO's `benchmark_app`,
  docs/benchmarks.md) run it; they find it by its format,
  `FORMAT_ONNX`. The core never executes it: its `backends` is empty,
  so rule 6 skips it on every device. If a fallback engine is ever
  built it is a backend like any other and gets its name in `backends`.
  The MiniLM recipe carries upstream's `onnx/model.onnx` unchanged, as
  the artifact `onnx-f32`.
- The header has no int8 dtype today. It is added when the Hailo
  backend lands, not before; the example shows the value it will use.
- Every load verifies every file it opens. No hash cache. If a
  multi-gigabyte file makes that slow, it is measured then.
- The tokenizer check runs on every load. The cases are short and the
  cost is not measurable.
- Whether the Hailo-10H can run the embedding lookup on the chip is not
  known until its compiler says; the artifact can say either.
- No per-artifact quality floor in the manifest. How close a backend
  gets to the reference is measured on a machine and recorded with the
  benchmark, not written into the bundle ahead of time.

## Left out

- Signatures. Hashes catch corruption; signing belongs to distribution.
- Benchmark numbers. They are per machine and per commit.
- Machine names, tuning, cache paths.
- Selection scores. Manifest order is the preference.
- BPE and unigram parameters, pair templates, labels, chat templates,
  other tasks. They arrive with the code that reads them.
