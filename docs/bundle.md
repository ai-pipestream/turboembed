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
      "backends": ["npu"],
      "compute_dtype": "DTYPE_F16",
      "graph_input": "INPUT_TOKEN_IDS",
      "graph_output": "OUTPUT_HIDDEN_STATES",
      "fixed_seq": 128,
      "fixed_batch": 1,
      "produced_by": {
        "tool": "openvino.save_model",
        "tool_version": "2026.3.0",
        "container": "turbo-reference@sha256:<digest>",
        "from": "onnx-f32",
        "args": [
          "onnx/model.onnx",
          "openvino/model.xml",
          "openvino/model.bin",
          "seq=128",
          "batch=1",
          "max_opset=11",
          "compress_to_fp16=True",
          "output_precision=FP16"
        ],
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
      "fixed_batch": 1,
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
| `model.license_class` | enum | no | The class the model zoo puts `model.license` in (docs/zoo.md): `CLASS_PERMISSIVE`, `CLASS_LGPL`, `CLASS_SHARE_ALIKE`, `CLASS_GPL` or `CLASS_LOCAL_ONLY`. The bundle tool writes it from the licence and `turbo-bundle verify` checks it. The tool makes no bundle under a licence the zoo refuses, and fetches a model whose class needs acceptance only with `--accept-terms`. |
| `task` | enum | yes | `TASK_EMBED`, `TASK_RERANK`, or `TASK_EXTRACTIVE_QA`. Names the task block (`embed`, `rerank`, or `qa`) that follows. Any other `TASK_*` name is a task this build does not have. |
| `embed.dim`, `.pooling`, `.normalize` | uint32, enum | yes | Vector width; what `TURBO_POOLING_MODEL` and `TURBO_NORMALIZE_MODEL` mean. With an `architecture`, `dim` equals `architecture.hidden`: the vectors are pooled hidden states. |
| `embed.max_seq` | uint32 | yes | Tokens per row including specials, the length the model was evaluated at. Never the positional table size. |
| `embed.max_batch` | uint32 | yes | The largest batch the reference was checked at. A session larger than it is refused. |
| `embed.prefix_query`, `.prefix_document` | string | no | Prepended for `TURBO_PROMPT_QUERY` and `TURBO_PROMPT_DOCUMENT`. |
| `embed.output_dims` | uint32[] | no | The widths the model was trained to be cut to, each in 1 to `dim` and listed once, at most `TURBO_OUTPUT_DIMS_MAX` (16) of them. `turbo_model_info.output_dims` reports them ascending. An `output_dim` above `dim` is `INVALID_ARGUMENT`; one that is neither `dim` nor listed is `UNSUPPORTED_OPTION`. The cut comes before normalize: an L2-normalized vector is unit length at `output_dim`. Below, "Declared widths". |
| `embed.sparse` | message | no | A learned sparse output beside the dense vectors (SPLADE, arXiv 2403.06789). `pooling` is `SPARSE_POOLING_MAXLOG`: the MLM head named in `artifacts[].tensor_names` scores each of the text's tokens, and a term's weight is the max over them of `log(1 + relu(logit))`. The encoding is `(term id, weight)` pairs sorted by term id, cut by an optional top-k or weight floor the caller gives. Requires an encoder `architecture`; a raw-weights artifact names the head's tensors, and a missing decoder weight ties it to `word_embeddings`, as BERT does. Terms the input carries keep their source spans; terms the model adds carry none. |
| `tokenizer.file` | path | yes | The upstream tokenizer file, unchanged: a `tokenizer.json`, or with `sentencepiece` the model's SentencePiece `.model` file. Its hash is `tokenizer_sha256`. |
| `tokenizer.normalizer.*` | bool, enum | with `wordpiece` | What the core applies to the text. The order is upstream BertNormalizer's, whatever the order of the fields: clean, split CJK, strip accents, lowercase. A `unigram` tokenizer has none: it normalizes with the map its file carries. A `bpe` tokenizer has none either: it reads the text as it is, or in NFC with `bpe.nfc`. A `sentencepiece` tokenizer has none: its model holds its normalizer. |
| `tokenizer.wordpiece`, `.bpe`, `.unigram`, `.sentencepiece` | message | one of | The kind and its parameters: `wordpiece` (`continuing_prefix`, `max_chars_per_word`), `unigram`, `bpe` and `sentencepiece`, below. |
| `tokenizer.bpe.*` | bool, enum | with `bpe` | Byte-level BPE as upstream `tokenizers` runs it for GPT-2, RoBERTa, Llama-3 and Qwen2: the text is cut into words by `pattern`, each word's bytes are written as GPT-2's printable characters, and the file's merges join them pair by pair, lowest rank first. `pattern` (default `BPE_PATTERN_GPT2`) is GPT-2's (a contraction, an optional space and a run of letters, of numbers or of other characters, or whitespace), with the file's pre-tokenizer ByteLevel with that pattern; or `BPE_PATTERN_LLAMA3` or `BPE_PATTERN_QWEN2` (contractions in any case, letters after an optional character that is no letter, number or line break, numbers in runs of up to three for Llama-3 and one at a time for Qwen2, others with the line breaks after them, line breaks with the whitespace before them, or whitespace), with the file's pre-tokenizer a Sequence of a Split with exactly that pattern (Isolated, not inverted) and a ByteLevel without `use_regex`. The core matches each pattern by hand, with no regular expression. `nfc` (default false) puts the text in NFC first, as Qwen2's file says with an NFC normalizer; spans still point into the caller's text, a character composed from several covering all of them. `add_prefix_space` puts a space in front of each piece of text between special tokens that does not start with one, with GPT-2's pattern and no NFC only; `ignore_merges` (default false) takes a word that is a piece of the vocabulary as that piece. All must match the file, whose normalizer must be null or, with `nfc`, NFC, and model BPE with no dropout, prefix, suffix, byte fallback or fused unknowns. Merges may be written `"left right"` or `["left", "right"]`. |
| `tokenizer.unigram.*` | bool, string | with `unigram` | SentencePiece's Unigram as upstream `tokenizers` runs it (XLM-RoBERTa, BGE-M3): `precompiled_charsmap` applies the precompiled character map the file carries (SentencePiece's `nmt_nfkc` for XLM-RoBERTa), `collapse_spaces` folds runs of spaces to one, every space becomes `metaspace` (one character, `▁`) and each piece of text between special tokens gets one in front when `add_prefix_space`; the text is cut at each metaspace into words, and each word into the pieces of highest total score, a character no piece covers being the SPECIAL_UNK token, a run of them one token. After the collapse of spaces, as Model2Vec's multilingual models normalize: `space_punctuation` puts a space on each side of every ASCII punctuation character, `collapse_whitespace` folds each run of whitespace to one space, and `strip` trims the text; `whole_text` segments the text at once, uncut at the metaspace (a Metaspace pre-tokenizer with `split` false). `whitespace_split` reads `xlm-roberta-base`'s own file, whose model section names no type and whose pre-tokenizer is WhitespaceSplit then Metaspace: the normalized text is cut at every Unicode whitespace character, which is dropped, and each word gets a metaspace in front; it does not go with `whole_text`. Each switch defaults to false and must match the file. |
| `tokenizer.sentencepiece.algorithm` | enum | with `sentencepiece` | `unigram` or `bpe`: the algorithm the `.model` file must name. The file holds everything else: its pieces and their types, its normalizer and character map, byte fallback (docs/tokenizer.md). Each special token must be a piece of the model under its id, and SPECIAL_UNK the model's unknown piece; none is matched in the text, as the sentencepiece library matches none, so none may have `lstrip`. |
| `tokenizer.special_tokens[]` | role, content, id, lstrip, normalized, single_word, rstrip | yes | Fills `pad_id`, `bos_id`, `eos_id`, `unk_id`. `lstrip` (default false) says the token takes the whitespace before it in the text, as upstream's `lstrip` added tokens do (XLM-RoBERTa's `<mask>`). `normalized` (default false, WordPiece or BPE) matches the token in the normalized text, its content normalized too, only as a whole word with `single_word`, taking the whitespace after it with `rstrip` (Model2Vec's code models' `[PAD]`); `single_word` and `rstrip` are WordPiece only. A BPE tokenizer without `nfc` has no normalizer, so a `normalized` token there is matched in the text as it is (GPT-2's and RoBERTa's files mark theirs so); with `nfc` no token may be `normalized`. Each must match the file's added token; a `sentencepiece` file has no added tokens, only pieces. |
| `tokenizer.template` | string[] | yes | The row layout around `$TEXT`. |
| `tokenizer.truncation` | enum | yes | What `TURBO_TRUNCATE_MODEL` means: `TRUNCATE_RIGHT` or `TRUNCATE_LEFT`. `TRUNCATE_NONE` is a caller option and is rejected here. |
| `architecture.*` | message | when an artifact is raw weights, a HEF or an OpenVINO IR, except in a static model | Everything a kernel path needs that a weights file does not carry. `family` is `FAMILY_BERT`, `FAMILY_ROBERTA` or `FAMILY_DEBERTA_V2` (below). BERT and RoBERTa are the same encoder, RoBERTa counting positions from `position_offset` (its padding id plus one: 2 for XLM-RoBERTa, whose table of `max_positions` 8194 rows serves 8192 tokens). `position_offset` is 0 for BERT and required for RoBERTa; `max_seq` plus it must fit `max_positions`. `lookahead`, optional, for `FAMILY_BERT` and `FAMILY_ROBERTA` only and 1 or more, is limited lookahead: each layer's query attends to every key before it and to at most that many keys after it, as SaT's `sat-1l` to `sat-12l` were trained (wtpsplit's config `lookahead` divided by its layers, rounded down; seal checks a recipe's value against an upstream `config.json` and refuses `lookahead_split_layers` inside the encoder because later layers require a causal mask that this uniform band cannot express). Absent, attention reads the whole row. It reaches the backend as `turbo_backend_model.lookahead` (0 for none; the field was `reserved`, at the same offset), and only the CPU backend runs it: the others refuse the model at load with `TURBO_E_UNSUPPORTED`. |
| `architecture.relative.position_buckets`, `.max_relative_positions` | uint32 | with `FAMILY_DEBERTA_V2` | `FAMILY_DEBERTA_V2` is DeBERTa-v2 and v3: shared-key disentangled attention over log-bucketed relative positions (Hugging Face's `DebertaV2Model` with `relative_attention`, `share_att_key`, `pos_att_type` c2p and p2c, `norm_rel_ebd` layer_norm). It takes `position_embedding` `POSITION_RELATIVE`, `token_types` 0 and `position_offset` 0, and both fields here: `position_buckets` even, 4 to 4096 (the relative table has twice as many rows), and `max_relative_positions` past `position_buckets / 2 + 1` and at most 65536. Its artifacts are `FORMAT_SAFETENSORS` only, and its `tensor_names` name `word_embeddings`, `embeddings_ln_weight` and `embeddings_ln_bias`, `relative_embeddings` (`[2 * position_buckets, hidden]`), `relative_ln_weight` and `relative_ln_bias`, and the BERT layer roles, with no `position_embeddings` or `token_type_embeddings`. They reach the backend as `turbo_backend_model.position_buckets` and `.max_relative_positions`. The CPU, cuda, levelzero and metal backends run the family so far, the devices in F32 at MODEL and EXACT and refusing FASTEST with `TURBO_E_UNSUPPORTED_OPTION` (docs/cuda.md, docs/levelzero.md, docs/metal.md); the others refuse the model at load with `TURBO_E_UNSUPPORTED` (#883). Sealing such a recipe reads the upstream `config.json`: it must be the shared-key profile (`model_type` `deberta-v2`, relative attention, `share_att_key`, `pos_att_type` c2p and p2c, the relative table LayerNormed, no absolute positions or token types), and every field of `architecture` must be what it gives (`hidden`, `heads`, `intermediate`, `layers`, `vocab_size`, `max_positions`, `layer_norm_eps`, the two relative fields, `token_types` 0), or the seal is refused naming the field. A DeBERTa-v2 bundle has no rerank pair template yet. |
| `static_embedding.vocab_size` | uint32 | in a static model | A static model (docs/static.md): one vector per vocabulary entry and no encoder, in place of `architecture`; the two never appear together. The tokenizer's entries; every special token's id is under it. Its artifacts are `FORMAT_SAFETENSORS`, naming `static_embeddings` (`[vocab_size, dim]`, or `[rows, dim]` with a mapping) in F32, F16, BF16, F64 or I8, and optionally `static_weights` (`[vocab_size]`, F32, F16 or F64) and `static_mapping` (`[vocab_size]`, I32 or I64, each value under `rows`). |
| `static_embedding.rows` | uint32 | with `static_mapping` | The table's rows when a token mapping picks them, as Model2Vec's vocabulary quantization stores the table; absent (0) without one. |
| `static_embedding.max_length` | uint32 | in a static model | The tokens `TRUNCATE_MODEL` keeps, StaticModel's `max_length`: 1 to `embed.max_seq`. The template must be `["$TEXT"]` and the truncation `TRUNCATE_RIGHT`. |
| `static_embedding.distilled_from.model_id`, `.revision`, `.manifest_sha256` | string | in a distilled static model | The base bundle the table was distilled from, by its manifest's SHA-256; with `quality`, or neither for a model made elsewhere. |
| `static_embedding.quality.texts` | path | with `distilled_from` | The texts the cost was measured on, JSON lines of `text` and `group`. |
| `classifier.kind` | enum | with `classifier` | `CLASSIFIER_TOKEN` or `CLASSIFIER_SEQUENCE`: whether the classification head scores each token's hidden state (`AutoModelForTokenClassification`) or the pooled sequence (`AutoModelForSequenceClassification`). Requires an encoder `architecture`. |
| `classifier.labels` | string[] | with `classifier` | The output labels in id order (`0..labels.len()`), matching `id2label` in the upstream `config.json`. At least one non-empty label. |
| `projection.dim` | uint32 | with `projection` | Output width of a linear token-projection head (`[projection.dim, hidden]` weight, optional `[projection.dim]` bias) applied to unpooled token hidden states before L2 normalization, as ColBERT and late-interaction models carry. Requires an encoder `architecture`. |
| `artifacts[].name` | string | yes | Unique; referenced by `from` and `host_weights`. |
| `artifacts[].format` | enum | yes | `FORMAT_SAFETENSORS`, `FORMAT_OPENVINO_IR`, `FORMAT_HEF`, `FORMAT_ONNX`. (`FORMAT_GGUF` is a header and manifest name only; no backend loads it.) |
| `artifacts[].files` | path[] | yes | Each listed in `files`. A `FORMAT_HEF` artifact is one file. A `FORMAT_OPENVINO_IR` artifact is two: the xml, then its weights. A `FORMAT_ONNX` artifact's first file is the graph; any others are its external data, beside it in the same directory, as the exporter wrote them. |
| `artifacts[].backends` | string[] | yes | `turbo_device_info.backend` values that load it. Empty: nothing loads it, as for an ONNX file carried only for the reference programs and the converters. |
| `artifacts[].target` | string | compiled artifacts | The device architecture label the artifact was compiled for. Matched against `turbo_device_info.arch`. |
| `artifacts[].fixed_seq`, `.fixed_batch` | uint32 | for `FORMAT_HEF` and `FORMAT_OPENVINO_IR` | The shape compiled in; 0 is dynamic. A FORMAT_HEF and a FORMAT_OPENVINO_IR set both. `fixed_batch` is a frame, not a limit on a session's batch. |
| `artifacts[].compute_dtype` | enum | yes for `FORMAT_HEF` and `FORMAT_OPENVINO_IR` | Fixed by the compilation, so never on `FORMAT_SAFETENSORS`. Absent: the session's `precision` decides, and `TURBO_PRECISION_MODEL` computes in the dtype the weights are stored in. |
| `artifacts[].graph_input`, `.graph_output` | enum | yes | Where the artifact starts and stops, so the backend knows which stages it must add. Raw weights (`FORMAT_SAFETENSORS`) start at `INPUT_TOKEN_IDS`. `INPUT_EMBEDDINGS` is defined under "Graph inputs" below. `OUTPUT_HIDDEN_STATES` is the last layer's hidden states, before pooling; raw weights stop there, the backend pools. `OUTPUT_EMBEDDINGS` is the pooled, normalized vectors, for a graph that carries the embed block's pooling and normalization (a sentence-transformers export); only graphs no backend runs stop there. |
| `artifacts[].host_weights` | string | when input is embeddings | The `FORMAT_SAFETENSORS` artifact whose embedding tensors the host lookup uses. Its `tensor_names` must name the five embedding roles; the layer roles are not read. |
| `artifacts[].tensor_names` | map | raw weights | Role to tensor name; `{layer}` is the layer index. A static model's roles (`static_embeddings`, `static_weights`, `static_mapping`) and an encoder's are never mixed. With `classifier`, raw-weights artifacts also name `classifier_weight` (`[labels, hidden]`) and `classifier_bias` (`[labels]`), and for `CLASSIFIER_SEQUENCE` with a BERT pooler, `pooler_weight` (`[hidden, hidden]`) and `pooler_bias` (`[hidden]`). With `embed.sparse`, they also name the MLM head: `mlm_transform_weight` (`[hidden, hidden]`), `mlm_transform_bias` (`[hidden]`), `mlm_transform_ln_weight` and `mlm_transform_ln_bias` (`[hidden]`), `mlm_decoder_bias` (`[vocab_size]`), and, unless the head is tied, `mlm_decoder_weight` (`[vocab_size, hidden]`). With `projection`, raw-weights artifacts name `projection_weight` (`[projection.dim, hidden]`) and optionally `projection_bias` (`[projection.dim]`). |
| `artifacts[].produced_by` | message | no | Absent means the upstream file, unchanged. |
| `produced_by.tool`, `.tool_version`, `.container`, `.reproducible` | string, bool | yes when present | What ran, in which pinned container, and whether two runs give identical bytes. |
| `produced_by.from` | string | yes in a converted artifact but raw weights, empty in `reference` | The artifact this one was converted from. The reference, and raw weights the bundle tool wrote from an upstream PyTorch checkpoint the bundle does not carry, are made from the upstream model, so their `from` is empty; the weights' `args` name the checkpoint. |
| `produced_by.inputs`, `.args` | path[], string[] | no | The other files read, the arguments. |
| `reference.file` | path | yes | A safetensors file with `ids` (I32 `[n, L]`, padded with `pad_id`), `lengths` (I32 `[n]`) and `embeddings` (F32 `[n, dim]`), row `i` for `cases[i]`, plus `logits` (F32 `[n, labels]` for `CLASSIFIER_SEQUENCE`, or `[sum(lengths), labels]` for `CLASSIFIER_TOKEN`) when `classifier` is present. |
| `reference.cases[]` | text, prompt_role | yes | The exact bytes the core is handed. If a service normalizes text upstream, these are post-normalization. One case is longer than `max_seq`, so truncation is checked too. |
| `reference.produced_by` | message | yes | The upstream pipeline, fp32 on CPU, in a pinned container. |
| `files[]` | path, uint64, hex | yes | Every file in the directory except the manifest, with size and SHA-256. |

## Declared widths

A dense model trained to be cut to several widths (the Matryoshka
objective) declares them in its recipe's top-level `matryoshka` field,
a JSON list the recipe's author reads off the upstream model card and
pins beside the rest of the model's facts:

```json
"matryoshka": [64, 128, 256]
```

The bundle tool checks the list when the recipe is loaded, before
anything is fetched: every width a nonzero integer within the model's
`embed.dim`, the list ascending (so no width twice), and at most
`TURBO_OUTPUT_DIMS_MAX` of them. It then fills `embed.output_dims` with
it; a list the manifest itself carries is checked the same way, and
giving both is refused. The tool never guesses: a model whose recipe
declares no widths serves its own `dim` alone, and a card that lists
none is a model with none. The core validates the manifest field again
when it is parsed at seal.

A session asked for another `output_dim` is refused with
`UNSUPPORTED_OPTION`, the message naming the declared widths; one over
`dim` is `INVALID_ARGUMENT`. The write itself pools each row into the
first `output_dim` coordinates and normalizes them there: no full-width
vector is materialized first, and the result moves `output_dim` values
a vector, so the narrower write is no slower than the full-width one.
The width is part of the vector's identity in analysis: a `Vectors`
layer cut to one width does not compare with one cut to another, and
the exact vector index refuses them (`IndexError::Dimension`,
docs/analysis.md).

## Sparse outputs

`embed.sparse` turns the encoder into a SPLADE-style sparse encoder (arXiv
2403.06789): the host reads each token's final hidden state through the
range write, applies the declared MLM head to it, and pools
`log(1 + relu(logit))` over the text's tokens into one weight per
vocabulary term. The output of one text is its `(term id, weight)` pairs
sorted by id, cut by an optional top-k or weight floor; a term id is a
model vocabulary id, so the sparse vector stands beside an inverted
index the way BM25 does.

The declaration is the recipe's `sparse` field, checked and copied into
`manifest.embed.sparse` by the bundle tool the way the Matryoshka widths
field is, and the head's tensors are named in `tensor_names`, the way a
classifier's are. Seal loads the head through the core, so a declaration
the weights cannot serve is refused at seal with the tensor named.

In the analysis crates the embed step writes the document's terms as
`turbo:sparse-terms` (`SparseTerms<View>`), each input term with the span
of its pieces, each model-added term marked expanded with no span, and
the chunker's adapter writes the same terms per `Levels` unit beside
`TermStats`, so BM25 and learned sparse scores fuse per unit in the level
router.

## Graph inputs

`INPUT_TOKEN_IDS`: the artifact takes the rows as written, ids, mask and
token types, and does the embedding lookup itself.

`INPUT_EMBEDDINGS`: the artifact takes, for each token, the row of the
word-embedding table (`word_embeddings`, `[vocab_size, hidden]`) its id
selects, as stored, gathered by id and nothing more: the position
embeddings and the token type embeddings are not yet added and the
embeddings LayerNorm is not yet applied. The artifact owns everything
after that gather: the position and token type add, the embeddings
LayerNorm, every layer, up to its `graph_output`. The backend does the
gather on the host from the `host_weights` artifact's `word_embeddings`,
which the core hands it with the other four embedding tensors, and
passes the mask as the artifact takes it (an additive attention bias,
say).

Such an artifact is given no token type ids, so the type embedding it
adds is the one compiled in as a constant: row 0 of
`token_type_embeddings`. An `INPUT_EMBEDDINGS` artifact therefore
computes token type 0 only. That limit is not a manifest field: it
follows from `graph_input`, for every `INPUT_EMBEDDINGS` artifact. A
backend running one refuses rows whose token types hold any value but 0
with `UNSUPPORTED_OPTION`, the message naming the row and position, and
runs nothing. Rows written as text, and rows with `types` NULL, are all
type 0. An artifact that takes the type ids beside the word rows would be
another `graph_input` value, added when one is compiled.

## Loader rules

Status codes are the header's `TURBO_E_*`.

1. No directory or no manifest: `BUNDLE_NOT_FOUND`.
2. The manifest is parsed strictly. An unknown field or enum value at
   any level, a missing required field, a string longer than its header
   buffer, more `embed.output_dims` than `TURBO_OUTPUT_DIMS_MAX`, a path
   with `..` or a leading `/`, or a path not present in `files`:
   `BUNDLE_INVALID`, naming the field. Any `TASK_*` name other
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
   implements. The backend says which formats it loads in
   `turbo_backend.formats` (turbo_backend.h); a backend that does not say
   loads `FORMAT_SAFETENSORS` alone. The core hands a backend
   `FORMAT_SAFETENSORS`, `FORMAT_OPENVINO_IR` and `FORMAT_HEF` artifacts
   only, whatever it says. Manifest order is the preference. None:
   `BUNDLE_NO_ARTIFACT`, with the message saying why each was skipped.
   A session's `precision` never picks another artifact; it says how the
   chosen one computes.
7. The chosen artifact's files, and its `host_weights` artifact's files,
   are verified by size, then SHA-256, before any byte is used. Where a
   backend's API takes memory, the bytes handed to it are the bytes
   hashed: a HEF reaches `model_load` as the bytes read and hashed, in
   `turbo_backend_model.artifact`, an OpenVINO IR as its two files' bytes
   in `artifact` (the xml) and `artifact2` (the weights), and the
   embedding tensors as they lie in the verified `host_weights` file.
8. For raw weights, and for a `host_weights` artifact's embedding
   tensors, every tensor the `tensor_names` map implies must
   exist with the shape the architecture or the `static_embedding` block
   implies (`[out, in]` for a linear layer), be `F32`, `F16` or `BF16` with every such tensor the
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
`max_seq`. `fixed_batch` is the frame the artifact runs, handed to the
backend in `turbo_backend_model.fixed_batch`; the backend runs as many
frames as a batch needs, so `max_batch` stays `embed.max_batch`. `TURBO_TRUNCATE_MODEL` still cuts at
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
refuses to finish unless every hash matches. The model zoo
(docs/zoo.md) fetches the analysis add-ons' models, heads, lexicons
and datasets the same way, and adds a licence class per artifact and
the user's acceptance of the licences that need it.

`turbo-bundle catalogue` lists the models the tool makes by name
(`bundle/src/catalogue.rs`): the BGE v1.5 encoders, BGE-M3, the two
rerankers and the potion static models. Every other recipe in
`bundle/recipes/` is made by its path, for the reason `BY_PATH` gives
beside it: it reads a file kept next to the recipe, it is a second
build of a catalogued model, or its training data puts it behind the
zoo's acceptance step. A test fails when a recipe file is in neither
list.

## Decisions taken with the design

- ONNX is a file in the bundle because the IR and the HEF are compiled
  from it and the manifest records that, and because the benchmark's
  reference programs (TensorRT's `trtexec`, OpenVINO's `benchmark_app`,
  docs/benchmarks.md) run it; they find it by its format,
  `FORMAT_ONNX`. The core never executes it: its `backends` is empty,
  so rule 6 skips it on every device. If a fallback engine is ever
  built it is a backend like any other and gets its name in `backends`.
  The MiniLM recipe carries upstream's `onnx/model.onnx` unchanged, as
  the artifact `onnx-f32`, and an F16 copy of it, `onnx-f16`
  (`onnx/model-f16.onnx`, `compute_dtype` `DTYPE_F16`), for TensorRT,
  which from version 11 builds F16 only from a strongly typed F16 graph.
  The bundle tool makes the copy in the reference container with
  onnxconverter-common (`bundle/reference/onnx_f16.py`): every float
  becomes float16 except in the ops on the converter's default block
  list, which stay float32 with casts around them; constants are clamped
  to `max_finite_val` 1e4 and `min_positive_val` 1e-7 in magnitude (its
  defaults, passed explicitly and recorded in `args`); the inputs and
  the output keep their types. The converter leaves a Cast to float of
  an integer input (BERT's attention mask) as it was, feeding F16 ops,
  so the script makes each such Cast one to float16
  (`float_casts_into_f16_ops=FLOAT16` in `args`) and refuses a result in
  which any other op takes both float and float16 inputs. The recipe
  names only `produced_by.from`; the tool fills in the rest from
  the run, and runs it twice to say whether it is `reproducible`. An
  export with external data (BGE-M3's `onnx/model.onnx` and
  `onnx/model.onnx_data`) lists both files, the graph first; the F16
  copy comes out as one file. A graph whose F32 tensors serialize past
  protobuf's 2 GB message limit cannot go through the converter's shape
  inference, so it is converted without it (`disable_shape_infer=True`
  in `args`) on the graph as exported.
- A model whose repository ships a PyTorch checkpoint and no
  safetensors file (BAAI/bge-m3 at its pinned commit) gets its raw
  weights from the bundle tool: `produced_by` on the
  `FORMAT_SAFETENSORS` artifact names `upstream`, the checkpoint's path
  among the recipe's upstream files, fetched and not carried, and the
  reference container writes every tensor as stored
  (`bundle/reference/bin_to_safetensors.py`). The library never reads the
  checkpoint.
- `DTYPE_I8` is `TURBO_DTYPE_I8` (6), the slot the header's numbering
  leaves for it below `TURBO_DTYPE_I32` (8). It is a compute dtype: a
  session, a capability or a compiled artifact computes in it; no buffer
  holds it.
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
- Pair templates, chat templates, other tasks. They arrive with
  the code that reads them.

### Cross-encoder reranking (TASK_RERANK)

A reranking bundle has `task: "TASK_RERANK"` and a `rerank` block instead
of `embed`. The block contains positive `max_seq`, `max_batch`, and
`query_tokens` limits and `pair_template: "PAIR_BERT"` or `"PAIR_ROBERTA"`.
BERT uses `[CLS] query [SEP] passage [SEP]` with segment types 0 then 1.
RoBERTa uses `<s> query </s></s> passage </s>` with segment type 0.
The query is cut on the right before passage windowing, leaving at least
one passage position. Windows cover the complete passage; their raw
relevance logits are combined by maximum. No sigmoid is applied.

`classifier` must describe a sequence head with one label. Its existing
classifier/pooler tensor roles carry the relevance weights. RoBERTa's
classification dense/tanh layer maps to the pooler roles. Missing head
weights, an incompatible template, or an unsupported task is refused at
load with the relevant field name. CPU is the initial execution backend.
The result records the model, artifact, and tokenizer identity.
