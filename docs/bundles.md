# Bundles

A bundle is a directory with a `bundle.json` manifest (bundle format
version 2) plus the files it references. `Bundle::open` in
`crates/turbo-core/src/bundle.rs` parses and verifies a bundle before any
provider sees it; providers read the resulting `Bundle`/`Manifest` and never
infer per-model behavior from a name (`PLAN.md` section 2, item 6).

## Manifest fields

All fields below are read by `crates/turbo-core/src/bundle.rs`'s `Manifest`,
`TokenizerSpec`, `Contract`, `FileEntry`, `Prompts`, and `Limits` structs,
every one of which is `#[serde(deny_unknown_fields)]`: a field name the
struct does not recognize, at any level, is `TURBO_E_BUNDLE_INVALID` rather
than a silently ignored typo.

Top level:

| field | type | required | notes |
|---|---|---|---|
| `bundle_version` | u32 | yes | must equal `2`; anything else is `TURBO_E_BUNDLE_INVALID` |
| `model_id` | string | yes, non-empty | |
| `revision` | string | no (default `""`) | |
| `license` | string | no (default `""`) | |
| `task` | string | yes | one of `embed`, `rerank`, `classify`, `token_classify`, `generate`, `tokenize`, `run`, `chunk` (`task_from_name`) |
| `kind` | string | yes | one of `embedding`, `reranker`, `classifier`, `token_classifier`, `generative`, `generic` (`kind_from_name`) |
| `modality` | string | no (default `"text"`) | one of `text`, `audio`, `image`, `video` (`modality_from_name`) |
| `family` | string | no (default `""`) | informational only (e.g. `bert`, `xlm-roberta`) |
| `tokenizer` | object or absent | no | see below |
| `contract` | object | no (default empty) | see below; required sub-fields depend on `kind` |
| `artifacts` | map | no (default empty) | format name (`onnx`, `openvino_ir`, `gguf`, `hef`, `mlx_safetensors`, `static`, `mock`, ...) to `FileEntry`; the bundle must declare at least one artifact or at least one `tokenizer.files` entry |
| `limits` | object | no | see below |

`tokenizer` (`TokenizerSpec`):

| field | type | required | notes |
|---|---|---|---|
| `kind` | string | yes if `tokenizer` present | e.g. `wordpiece`, `bpe`, `unigram`, `sentencepiece`, `gguf`, `mock` |
| `files` | map | no (default empty) | role name (e.g. `tokenizer.json`) to `FileEntry` |
| `chat_template` | string | no | for generative models |

`contract` (`Contract`): every field below that names a set of values is
parsed and checked against that set whenever it is present, regardless of
`kind`. A misspelled value (`"Mean"`, `"soft-max"`) is `TURBO_E_BUNDLE_INVALID`
naming the field, never silently ignored because the current kind does not
strictly need it (`validate_contract` in `bundle.rs`).

| field | type | required | notes |
|---|---|---|---|
| `pooling` | string | embedders only | `mean`, `cls`, `last` (`Pooling::from_name`) |
| `normalize` | string | embedders only | `l2` or `none` |
| `max_seq` | u32 | yes, except `generic` | must be non-zero |
| `dim` | u32 | embedders only | must be non-zero |
| `truncate_dims` | array of u32 | no | each entry must be in `1..=dim` (Matryoshka truncation targets) |
| `prompts.query`, `prompts.document` | string | no | prefixes applied by `TURBO_PROMPT_QUERY`/`DOCUMENT` |
| `similarity_fn` | string | no | `cosine`, `dot`, `euclidean` |
| `dtype` | string | no | model compute dtype name |
| `vocab_size` | u32 | no | |
| `labels` | array of string | classifiers/token classifiers | must be non-empty for those kinds |
| `activation` | string | classifiers, token classifiers, rerankers | `softmax`, `sigmoid`, `none`; missing on one of those kinds is `TURBO_E_BUNDLE_INVALID` ("scored models must declare their activation") |
| `aggregation` | string | token classifiers | `none`, `simple`, `first`, `max` (`Aggregation::from_name`); missing on a token classifier is `TURBO_E_BUNDLE_INVALID` |
| `tagging` | string | no | `BIO`, `BILOU`, `IOB1` |

`FileEntry`: `path` (relative, no `..`/absolute), `sha256` (lowercase hex),
optional `provenance` free text.

`limits` (`Limits`): `max_batch` (u32, `0` = provider default),
`fixed_shape` (bool, `true` for NPU-style fixed-shape artifacts).

## Required fields per model kind

`validate_contract` in `bundle.rs` enforces:

- Every kind: any of `contract.pooling`, `contract.normalize`,
  `contract.aggregation`, `contract.activation`, `contract.tagging` that is
  present must parse to one of its known values, whether or not this kind
  needs that field.
- `embedding`: `contract.dim != 0`, `contract.pooling` present,
  `contract.normalize` present; every `truncate_dims` entry in `1..=dim`.
- `classifier`, `token_classifier`: `contract.labels` non-empty,
  `contract.activation` present; `token_classifier` additionally requires
  `contract.aggregation`.
- `reranker`: `contract.activation` present.
- `generative`, `generic`: no kind-specific contract fields required.
- All kinds except `generic`: `contract.max_seq != 0`.

Any violation is `TURBO_E_BUNDLE_INVALID`, raised before the manifest is
otherwise trusted.

## Verification

`Bundle::open` / `Bundle::from_manifest`:

1. The directory must exist (`TURBO_E_BUNDLE_NOT_FOUND` otherwise); the
   `bundle.json` file inside it must exist and parse as JSON matching
   `Manifest` (`TURBO_E_BUNDLE_NOT_FOUND` if missing, `TURBO_E_BUNDLE_INVALID`
   on a JSON/shape error surfaced as a `serde_json` error).
2. `bundle_version` and the required-field checks above run.
3. `artifacts` and `tokenizer.files` (or the absent tokenizer) cannot both be
   empty — a bundle needs at least an artifact or a tokenizer to be useful
   for anything (`TURBO_E_BUNDLE_INVALID` otherwise). A tokenizer-only
   bundle with no artifacts is valid as long as it declares tokenizer files;
   see the example at the end of this page.
4. Every path referenced by `tokenizer.files` and `artifacts` is resolved
   with `Bundle::resolve`, which rejects an absolute path or any path
   component that is `..`, a root, or a Windows prefix
   (`TURBO_E_BUNDLE_INVALID`), rejects a path that does not point to an
   existing file (`TURBO_E_BUNDLE_NOT_FOUND`), and then canonicalizes the
   resolved path and rejects it (`TURBO_E_BUNDLE_INVALID`) unless it still
   falls under the bundle directory's own canonical path. A symlink inside
   the bundle that points outside it cannot be used to make the manifest's
   recorded hashes vouch for a file elsewhere on disk.
5. Every one of those files is hashed with SHA-256 and compared
   case-insensitively against the manifest's recorded `sha256`; the first
   mismatch is `TURBO_E_BUNDLE_INTEGRITY` and the bundle does not load.

A bundle that passes all of this is still only a verified manifest — whether
a specific provider can actually run it is a separate question answered by
`Provider::can_run` (see `docs/architecture.md`'s capability matrix section).

## Bundle identity

`Bundle::manifest_sha256()` is the hex SHA-256 of the manifest bytes: the
bytes read from `bundle.json` when opened from disk, or the canonical
`serde_json` serialization when built from an already-parsed `Manifest`
(`Bundle::from_manifest`). The manifest is the frozen per-model contract
(pooling, normalization, labels, prefixes, dimension), so this hash, not
only the artifact and tokenizer file hashes, is what a caller pins to know
it is running the model it qualified. `PLAN.md` section 6 lists `source_sha256` as
part of the bundle identity; `Manifest` has no such field, and
`manifest_sha256` is the identity hash this tree actually implements.

## Mock bundle layout

`testdata/bundles/mock/` has one subdirectory per `MockBundleKind`
(`crates/turbo-core/src/mock.rs`): `embedding`, `reranker`, `classifier`,
`token-classifier`, `generative`, `generic`. Each contains `bundle.json` and
`mock.json` (`{"vocab_size": 1000, "salt": 7}`, the mock's only artifact,
referenced from `artifacts.mock`). These are committed fixtures, not
hand-written: regenerate them with

```bash
cargo run -p turbo-core --example write_mock_bundles
```

CI (`.github/workflows/ci.yml`) runs the same command and fails the build if
the working tree then differs, so the fixtures and `mock.rs` cannot drift
apart. Do not hand-edit files under `testdata/bundles/mock/`.

## Importer

`tools/turbo-bundle` (`cargo run -p turbo-bundle --`) has three subcommands.

```bash
turbo-bundle import --source <model dir> --output <bundle dir> --license Apache-2.0 \
    [--artifact onnx=model.onnx] [--artifact openvino_ir=model.xml] [--artifact gguf=model.gguf] \
    [--static-from model.safetensors[:tensor]] [--truncate-dims 256,128] [--max-batch 32]
turbo-bundle verify <bundle dir>
turbo-bundle inspect <bundle dir>
```

`import` derives the contract from the source model's own files
(`tools/turbo-bundle/src/import.rs`), never from its name, and refuses rather
than guesses when a file is ambiguous or missing a needed field:

- **Module stack** (`modules.json`) must be exactly one Transformer, one
  Pooling, and an optional Normalize; any other module (a Dense projection,
  for example) or a second Pooling module is refused naming the module type
  and path, because the importer cannot express what that module would do
  to the vectors and refuses to write a contract that lies about it.
- **Pooling and normalization** come from the Pooling module's directory
  `config.json`, in either the sentence-transformers v6 schema
  (`pooling_mode`) or the legacy six-boolean schema
  (`pooling_mode_cls_token`, `pooling_mode_mean_tokens`, ...; exactly one
  flag set, or none, meaning `mean`; more than one flag set is an error
  naming the file). A `Normalize` module sets `normalize = l2`. The same
  config's `word_embedding_dimension`, when present, must equal
  `config.json`'s `hidden_size`; a mismatch is refused, since the importer
  would otherwise write a `contract.dim` the export does not actually
  produce.
- **Prompts** (`contract.prompts.query`/`document`) and `similarity_fn` come
  from `config_sentence_transformers.json`'s `prompts` and
  `similarity_fn_name`; any prompt name other than `query`/`document`/
  `passage` is recorded as an ignored note, not silently dropped.
- **Sequence limit** comes from `--max-seq`, then
  `sentence_bert_config.json`'s `max_seq_length`, then
  `tokenizer_config.json`'s `model_max_length` (only when it parses as a
  positive integer no larger than 2^20); `config.json`'s
  `max_position_embeddings` is never used; it is the positional table size
  (514 for XLM-R, 8192 for ModernBERT), not the length the model was
  evaluated at. When both `sentence_bert_config.json` and
  `tokenizer_config.json` give a value and they disagree, the
  `sentence_bert_config.json` value is kept and the disagreement is
  recorded as a note. With no source for it and no `--max-seq`, the import
  refuses (except for a `generic` bundle, where `max_seq` is `0`).
- **Kind and task** come from `--kind`/`--task`, or are inferred from
  `config.json`'s `architectures` entry (`ForSequenceClassification`,
  `ForTokenClassification`, `ForCausalLM`) and the presence of pooling;
  a single-label `ForSequenceClassification` becomes `reranker`, otherwise
  `classifier`. An architecture the importer does not recognize is an error
  naming `--kind`.
- **Labels** come from `config.json`'s `id2label`, sorted by index.
- **Activation** follows the head configuration, never the model kind
  alone: for a classifier or token classifier, `config.json`'s
  `problem_type` maps `single_label_classification` (or absent) to
  `softmax`, `multi_label_classification` to `sigmoid`, and `regression` to
  `none`; any other value is refused naming it. For a reranker (cross-encoder),
  `sbert_ce_default_activation_function` maps a class name ending in
  `Sigmoid` to `sigmoid`, one ending in `Identity` to `none`, and absent to
  `sigmoid`; any other value is refused naming it.
- **Tokenizer**: a `tokenizer.json` in the source is copied as-is (its kind
  detected from the file's `model.type`: WordPiece, BPE, Unigram,
  WordLevel). Without one, a `vocab.txt` plus `tokenizer_config.json`'s
  `do_lower_case` builds a `tokenizer.json` for a BERT WordPiece tokenizer
  (`BertNormalizer` + `BertPreTokenizer` + `WordPiece` + `BertProcessing`,
  matching what `BertWordPieceTokenizer(vocab.txt).save()` produces).
  `[PAD]`/`[UNK]`/`[CLS]`/`[SEP]`/`[MASK]` must all be present in
  `vocab.txt`.
- **Artifacts** are copied into the bundle by `--artifact format=path`
  (repeatable; `onnx`, `openvino_ir`, `gguf`, `hef`, `mlx_safetensors`, ...);
  an `openvino_ir` artifact also requires and copies the sibling `.bin`
  weights file (recorded as `openvino_ir_weights`).
- **`--static-from path[:tensor]`** builds a `static` artifact (a
  little-endian f32 `[vocab_size, dim]` table) from a safetensors file
  (`tools/turbo-bundle/src/safetensors.rs`, supporting `F32`, `F16`, `BF16`
  tensors), for the `static` provider (`docs/providers.md`). Without a
  `:tensor` suffix, the tool picks the file's only tensor or one named
  `embeddings`. `config.json` must declare `normalize` (`true`/`false`)
  since a model2vec source has no pooling config to read.

An import stages its output into a sibling `.<name>.staging` directory and
renames it into place only after the manifest is written and re-verified
with `Bundle::open`, so an interrupted or failing import never leaves a
loadable-looking bundle at the target path; `--output` must not already
exist.

`verify` prints the model id, kind, and artifact formats after checking
hashes; `inspect` prints the full manifest as JSON. Both call the same
`Bundle::open` the runtime uses, so they report exactly what a load would
see.

Example: `testdata/bundles/minilm-tokenizer/` is a tokenizer-only bundle
(the fixture used by tokenizer and chunk-planner tests) — a `bundle.json`
whose `artifacts` map is empty and whose `tokenizer` block points at a real
MiniLM `tokenizer.json`. It loads and verifies like any other bundle;
callers that only need `turbo_tokenizer_create`/`turbo_chunk_plan_create`
never need a model artifact.
