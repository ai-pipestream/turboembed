# Bundle design, candidate C: the previous attempt, read after candidate A was written

Read from `ai-slop-generated-shit/docs/bundles.md`, the `Manifest` structs
in `crates/turbo-core/src/bundle.rs` (596 lines) and the committed
example `testdata/bundles/minilm-tokenizer/bundle.json`. Neither audit
flagged the bundle code itself; the importer is Rust, refuses rather
than guesses, and reads the contract from the model's own files. This is
the part of the old tree in best shape.

## What it is

`bundle.json`, format version 2. Top level: `bundle_version`, `model_id`,
`revision`, `license`, `task`, `kind`, `modality`, `family`, `tokenizer`,
`contract`, `artifacts`, `limits`. The `contract` holds pooling,
normalize, max_seq, dim, `truncate_dims`, prompts, similarity function,
dtype, vocab size, and the classifier fields (labels, activation,
aggregation, tagging). `artifacts` is a map from format name to one
`FileEntry` (`path`, `sha256`, optional free-text `provenance`).
`tokenizer` is a kind plus a map of role name to `FileEntry`, with an
optional chat template. `limits` has `max_batch` and `fixed_shape`.

Every struct is `deny_unknown_fields`, so a typo is an error. Paths are
checked against `..`, absolute paths and symlinks that leave the
directory, then canonicalized. Every referenced file is hashed at open,
before any provider sees the bundle. The manifest's own hash is the
bundle identity.

## What it got right, and candidate A should keep

- Refusing unknown fields and misspelled values at every level, with
  the field named. Candidate A says "invalid, field named" and should
  say this exactly.
- The symlink rule: canonicalize and require the result under the
  bundle directory, so a hash cannot vouch for a file elsewhere.
- Hashing the manifest bytes as the identity of the contract, separate
  from the artifact and tokenizer hashes. Candidate A has no manifest
  hash; it should.
- `truncate_dims`: the set of Matryoshka widths the model was trained
  for, each checked against `dim`. Candidate A's `output_dim` option has
  no such list, so any width would be accepted; it needs this.
- The importer's rule that `max_seq` is the length the model was
  evaluated at, never the positional table size, with its sources in
  order and a refusal when none exists.

## What it lacks for the restart

- No architecture description. The old tree ran ONNX and IR, which
  carry their own graphs; a kernel path over safetensors has nothing to
  tell it the layer count, hidden size, head count, activation or
  epsilon. Candidate A's `architecture` block is required for that.
- One artifact per format, keyed by format name. A HEF per Hailo target
  cannot be expressed (there is one `hef` slot), and nothing says which
  backend takes which artifact; that was a heuristic in each provider.
  Candidate A lists artifacts with `backends` and `target`.
- No reference outputs. The numeric check lived in test data outside the
  bundle, tied to one machine's ONNX run. Candidate A puts the inputs and
  the fp32 vectors in the bundle so every backend is checked against the
  same thing.
- `files` has no sizes, and files not referenced by an artifact or the
  tokenizer are not listed at all, so a stray file in the directory is
  invisible to verification.
- Six model kinds, four modalities and eight tasks in the schema, with
  `task` and `kind` overlapping (`embed` and `embedding`). The restart
  has one task; the schema should grow with the header, not ahead of it.
- Normalizer settings for the native tokenizer (lowercase, accent
  stripping, truncation side) were read from `tokenizer.json` at run
  time by each of the three tokenizer copies. With one tokenizer that is
  fine, but the manifest should state them so a disagreement is caught
  at load, not at the first wrong id.
- `provenance` is free text with no required content.

## Verdict on this one

Keep its verification rules and its importer discipline; drop its schema.
The schema was built for a runtime that executed graphs, and the
restart executes weights.
