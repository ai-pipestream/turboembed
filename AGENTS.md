# Turboembed engineering rules

These rules apply to the whole repository. Follow the pull request template and keep review evidence tied to an exact commit.

## Architecture

- Preserve 100% bidirectional offset tracking across every `View` and `Alignment` chain. Support conversion from `ViewId::SOURCE` to each child view and back, and among UTF-8 bytes, UTF-16 code units, and Unicode code points through `OffsetTable`.
- Do not add the `regex` crate anywhere in the workspace. Use `turbo-matcher` or direct structural checks instead.
- Do not add external NLP dependencies. `turbovec` is allowed; use the workspace's `turbo-unicode`, in-house `sha256`, and `turbo-zstd` implementations for their respective jobs.
- Where a stage must match an upstream tokenizer, its character classes follow that upstream's Unicode version, not `turbo-unicode`'s newer data. The core's BPE split classes are checked against upstream's own engine for every scalar; do not narrow that check to samples.
- Keep the flow minimal: text or model input, normalization, tokenization, subwords, then embeddings. Reuse existing `View`, `Tokens`, `Subwords`, and span data. Do not re-tokenize compatible inputs or allocate synthetic joined strings merely to feed another stage.
- Prefer borrowed slices and zero-copy data flow where the existing interfaces permit them.
- No JSON front end. RPC payloads between a browser and the server are binary protobuf through Connect-ES only: generated clients (`@bufbuild/protoc-gen-es`, `@connectrpc/connect-web`) with the binary codec (`application/proto`, `application/connect+proto`), as `POST /<package>.<Service>/<Method>` through the demo's allowlisted Connect proxy. Raw file uploads are `HTTP PUT` of binary frames to the proxy's upload endpoints (#1377), up to the 2 GiB request cap. JSON is allowed only at the edges where it fits, such as processing a form: it stays small and local, and never carries or mirrors a proto message. It is never a general front-end layer and never in the RPC path: no `/api/*` JSON, NDJSON or SSE routes, no serde derives on generated proto types, and no hand-written JSON or base64 mapping of proto messages. The reason: once JSON is in, agents keep pushing it into the hot path, and that has forced several rewrites. `bench/tests/tree.rs` enforces the RPC-path part (docs/demo.md).

## Hygiene

- Every touched Rust source file must remain under 500 lines.
- No literal em dash characters, workstation paths, or hostnames in code, documentation, commits, or pull request comments.
- AI attribution trailers (such as `Co-Authored-By`) and generated footers are allowed in commits, pull request descriptions, and comments. They are never a reason to block a review.
- Keep documentation and mechanical tree checks in step with moved, added, or removed files.
- Never use destructive worktree cleanup or history rewriting to clear an obstacle. Fix forward with ordinary commits.

## Review and merge

- Any agent may push a fix commit directly to an open pull request branch.
- A merge requires two independent `PASS` verdicts naming the same exact 40-character head SHA and a green hosted `check` gate.
- A reviewer cannot approve a head containing that reviewer's own commit. Pushing a fix voids the pusher's own verdict and requires fresh independent review.
- Exception: a `krick-1` `PASS` counts as one of the two even on a head carrying `krick-1`'s own commits. The other `PASS` must then come from a different agent that has no commit on that head.
- Do not merge with an unresolved `BLOCK`, and do not treat skipped hardware evidence as a passing hardware result.
