# Optional source chunking

`turboembed::chunker` (in [`crates/turboembed/src/chunker.rs`](../crates/turboembed/src/chunker.rs))
plans model-sized chunks over caller-owned text. It is the optional utility
described in roadmap M6 and the
[library design](library-design.md#tokenization-chunking-and-future-analysis).
It performs no inference and no tokenization of its own; callers can always
skip it and provide their own chunks or prepared tokens.

## What it does

`chunk_source(source, text, &config, &counter)` returns a `ChunkPlan`
containing the source identity, the configuration that produced the plan, and
chunks in source order. Each `SourceChunk` reports:

- half-open UTF-8 **byte** offsets into the original, unmodified input
  (`&text[chunk.byte_range]` is exactly the chunk text — the input is never
  normalized or copied);
- the paragraph it belongs to (chunks never span paragraph boundaries);
- its content-token count under the supplied counter.

Configuration is explicit and echoed back in the plan:

- `max_tokens` — the model sequence capacity;
- `reserved_tokens` — tokens held back for special tokens and model prefixes
  (for example 2 for `[CLS]`/`[SEP]`); the usable content budget per chunk is
  `max_tokens - reserved_tokens`;
- `overlap_tokens` — an upper bound on counted tokens that consecutive chunks
  of the same paragraph may share (0 disables overlap; it must stay below the
  content budget so chunks always advance);
- `sentence_boundaries` — whether an over-budget paragraph is split at
  sentence boundaries before whitespace and character boundaries.

Splitting is layered and always makes forward progress: paragraphs (blank-line
separated) are the outer boundary; a paragraph over budget splits at sentence
boundaries (optional), then at whitespace, then at character boundaries, so a
long span without punctuation still chunks. If a single character exceeds the
budget under the supplied counter, chunking returns an explicit
`ChunkError::NoProgress` instead of looping or silently truncating.

## Token budgets come from the model tokenizer

The budget is meaningful only in the tokens the model will actually consume,
so token counts come from a caller-supplied `TokenCounter` — any
deterministic `Fn(&str) -> usize`, typically the loaded model tokenizer's
content-token count (excluding special tokens, which `reserved_tokens`
covers). This module deliberately does not bundle or guess a tokenizer;
pairing a chunk plan with a different tokenizer than the one that counted it
voids the budget guarantee. The counter must be deterministic, and chunk
boundaries can trigger many counts over overlapping spans; cache inside the
counter if that matters for your input sizes.

## What it is not

- **Not linguistic analysis.** Sentence segmentation is a deterministic
  terminator scan (`.`, `!`, `?`, closing quotes/brackets, a small English
  abbreviation list) shared with the E2E harness. It is a split preference,
  not a semantic claim, and is not configurable per language.
- **Not part of the C ABI or the Java surface.** This is host CPU Rust with
  no GPU work. Exposing it through the frozen C ABI or the Java adapters
  (including UTF-16 offset conversion, which the design assigns to Java
  adapters on request) is a separate, explicitly versioned extension.
- **Not a normalizer.** Offsets always index the original bytes, including
  CRLF and embedded NUL bytes inside a paragraph.

## Relationship to the E2E chunker

The E2E harness's [`chunker`](../crates/e2e/src/chunker.rs) keeps its own
purpose — stable positional ids over CR/LF-normalized text for repeatable
embed batches — but its paragraph and sentence segmentation now delegates to
`turboembed::chunker::paragraph_spans` / `sentence_spans`, so there is one
segmentation implementation in the repository.

## Validation

Contract tests with fixed fixtures live in
[`crates/turboembed/tests/chunker_contract.rs`](../crates/turboembed/tests/chunker_contract.rs):
determinism, offset fidelity (including CRLF/NUL/non-ASCII input), budget and
overlap bounds, forward progress on punctuation-free text, and explicit
config/no-progress errors. They use fixed deterministic counters and change no
embedding goldens.

```bash
cargo test --locked -p turboembed --test chunker_contract
cargo test --locked -p inferstream-e2e
```
