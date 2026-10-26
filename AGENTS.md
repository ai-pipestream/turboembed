# Turboembed engineering rules

These rules apply to the whole repository. Follow the pull request template and keep review evidence tied to an exact commit.

## Architecture

- Preserve 100% bidirectional offset tracking across every `View` and `Alignment` chain. Support conversion from `ViewId::SOURCE` to each child view and back, and among UTF-8 bytes, UTF-16 code units, and Unicode code points through `OffsetTable`.
- Do not add the `regex` crate anywhere in the workspace. Use `turbo-matcher` or direct structural checks instead.
- Do not add external NLP dependencies. `turbovec` is allowed; use the workspace's `turbo-unicode`, in-house `sha256`, and `turbo-zstd` implementations for their respective jobs.
- Keep the flow minimal: text or model input, normalization, tokenization, subwords, then embeddings. Reuse existing `View`, `Tokens`, `Subwords`, and span data. Do not re-tokenize compatible inputs or allocate synthetic joined strings merely to feed another stage.
- Prefer borrowed slices and zero-copy data flow where the existing interfaces permit them.

## Hygiene

- Every touched Rust source file must remain under 500 lines.
- No literal em dash characters, workstation paths, hostnames, generated footers, or AI attribution trailers in code, documentation, commits, or pull request comments.
- Keep documentation and mechanical tree checks in step with moved, added, or removed files.
- Never use destructive worktree cleanup or history rewriting to clear an obstacle. Fix forward with ordinary commits.

## Review and merge

- Any agent may push a fix commit directly to an open pull request branch.
- A merge requires two independent `PASS` verdicts naming the same exact 40-character head SHA and a green hosted `check` gate.
- A reviewer cannot approve a head containing that reviewer's own commit. Pushing a fix voids the pusher's own verdict and requires fresh independent review.
- Do not merge with an unresolved `BLOCK`, and do not treat skipped hardware evidence as a passing hardware result.
