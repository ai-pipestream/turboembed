# Bundles

A bundle is a directory with a `bundle.json` manifest (bundle format
version 2) plus the files it references. `Bundle::open` in
`crates/turbo-core/src/bundle.rs` parses and verifies a bundle before any
provider sees it; providers read the resulting `Bundle`/`Manifest` and never
infer per-model behavior from a name (`PLAN.md` section 2, item 6).

## Manifest fields

All fields below are read by `crates/turbo-core/src/bundle.rs`'s `Manifest`,
`TokenizerSpec`, `Contract`, `FileEntry`, and `Limits` structs.

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
| `artifacts` | map | yes, non-empty | format name (`onnx`, `openvino_ir`, `gguf`, `hef`, `mlx_safetensors`, `mock`, ...) to `FileEntry` |
| `limits` | object | no | see below |

`tokenizer` (`TokenizerSpec`):

| field | type | required | notes |
|---|---|---|---|
| `kind` | string | yes if `tokenizer` present | e.g. `wordpiece`, `bpe`, `unigram`, `sentencepiece`, `gguf`, `mock` |
| `files` | map | no (default empty) | role name (e.g. `tokenizer.json`) to `FileEntry` |
| `chat_template` | string | no | for generative models |

`contract` (`Contract`):

| field | type | required | notes |
|---|---|---|---|
| `pooling` | string | embedders only | `mean`, `cls`, `last` |
| `normalize` | string | embedders only | `l2` or `none` |
| `max_seq` | u32 | yes, except `generic` | must be non-zero |
| `dim` | u32 | embedders only | must be non-zero |
| `truncate_dims` | array of u32 | no | each entry must be in `1..=dim` (Matryoshka truncation targets) |
| `prompts.query`, `prompts.document` | string | no | prefixes applied by `TURBO_PROMPT_QUERY`/`DOCUMENT` |
| `similarity_fn` | string | no | `cosine`, `dot`, `euclidean` |
| `dtype` | string | no | model compute dtype name |
| `vocab_size` | u32 | no | |
| `labels` | array of string | classifiers/token classifiers | must be non-empty for those kinds |
| `activation` | string | no | `softmax`, `sigmoid`, `none` |
| `aggregation` | string | token classifiers | `none`, `simple`, `first`, `max` |
| `tagging` | string | no | `BIO`, `BILOU`, `IOB1` |

`FileEntry`: `path` (relative, no `..`/absolute), `sha256` (lowercase hex),
optional `provenance` free text.

`limits` (`Limits`): `max_batch` (u32, `0` = provider default),
`fixed_shape` (bool, `true` for NPU-style fixed-shape artifacts).

## Required fields per model kind

`validate_contract` in `bundle.rs` enforces:

- `embedding`: `contract.dim != 0`, `contract.pooling` present,
  `contract.normalize` present; every `truncate_dims` entry in `1..=dim`.
- `classifier`, `token_classifier`: `contract.labels` non-empty.
- `reranker`, `generative`, `generic`: no kind-specific contract fields
  required.
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
3. `artifacts` must be non-empty.
4. Every path referenced by `tokenizer.files` and `artifacts` is resolved
   with `Bundle::resolve`, which rejects an absolute path or any path
   component that is `..`, a root, or a Windows prefix
   (`TURBO_E_BUNDLE_INVALID`), and rejects a path that does not point to an
   existing file (`TURBO_E_BUNDLE_NOT_FOUND`).
5. Every one of those files is hashed with SHA-256 and compared
   case-insensitively against the manifest's recorded `sha256`; the first
   mismatch is `TURBO_E_BUNDLE_INTEGRITY` and the bundle does not load.

A bundle that passes all of this is still only a verified manifest — whether
a specific provider can actually run it is a separate question answered by
`Provider::can_run` (see `docs/architecture.md`'s capability matrix section).

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

`PLAN.md` section 6 describes a `turbo-bundle import` tool that derives a
bundle's contract from a source model's own files (`modules.json`,
`1_Pooling/config.json` in both the sentence-transformers v6 schema and the
legacy six-boolean schema, `config_sentence_transformers.json` prompts,
`sentence_bert_config.json`, GGUF metadata for generative models), refusing
ambiguous inputs rather than guessing. This tool does not exist yet in this
tree (`tools/turbo-bundle/` is not present); it is planned for P1 alongside
the bundle format's first non-mock provider (`PLAN.md` section 10, P1 gate:
"bundle importer round-trips both sentence-transformers schemas").
