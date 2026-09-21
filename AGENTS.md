# Working on TurboEmbed

This guidance applies to the whole repository.

## Scope and orientation

- TurboEmbed is the in-process embedding library. TurboRerank provides reranking, TurboBuffer manages native buffers, and Inferstream is the serving layer. Preserve these boundaries and existing public names unless a task explicitly changes them.
- Start by reading the task, Git status, relevant headers, and the implementation behind the requested feature. Preserve unrelated edits and untracked review notes. Inspect actual remotes before publishing; do not assume a hosting policy from another repository.
- Read `MUSE-REVIEW.md` and `docs/code-review-2026-09-13.md` for orientation. They are dated reviews, not specifications or evidence that a feature works today. Verify their claims against the current checkout.
- Read `ROADMAP.md` and `docs/library-design.md` for product direction: Intel OpenVINO is the confirmed first GPU correctness/performance baseline, followed by NVIDIA and Apple qualification. Prioritize direct execution, reusable native buffers, and thin bindings measured against matched native execution. Inferstream is an optional server consumer.
- The initial work is review and planning. Do not implement Java bindings, new backends, or review fixes until the user requests implementation. Future implementation requests authorize their stated scope.
- Desktop Java targets JDK 25+ through Panama FFM. JNI delivery belongs to the later Android phase, with its own native provider/build path. These are confirmed choices; follow the roadmap's dependencies. OpenNLP integration remains optional and later.

## Code map

- `include/`: canonical C contracts. Keep corresponding headers in `swift/Sources/*C/include/` identical. Rust declarations must match their layouts and semantics.
- `crates/turboembed/` and `native/turboembed/`: Rust wrapper, ORT hooks, C ABI dispatch, and Intel implementation. Despite its filename, `stub.cpp` also dispatches real providers. `native/turboembed/src/hailo.cpp` is the Raspberry Pi AI HAT+ (Hailo) provider behind `--features hailo` (host-side wordpiece + embedding tables + pooling, encoder body on the NPU via HailoRT); see `docs/hailo-embed.md`.
- `native/turbo_buffer/`, `native/wordpiece/`, and `native/turborerank/`: allocation, tokenization, and reranking implementations shared across callers.
- `swift/`: supported Apple implementation and server. The Rust Apple server crate is a Linux compile stub; `native/mlx-engine/` is legacy material.
- `crates/backend*`, `crates/server/`, `crates/arch-*`, and `proto/`: backend contracts, serving, platform construction, and wire contracts. Keep engine-specific behavior behind the backend boundary.
- `config/catalog.toml`, `crates/fetch/`, and `crates/xtask/`: model configuration and fetching. `testdata/` contains fixtures and historical receipts; `docs/` contains contracts and runbooks.

## Correctness requirements

- Treat ownership as executable behavior. Safe wrappers must enforce engine/result lifetimes and prevent concurrent access or callback reentry that violates the native contract. Comments asking callers to avoid unsafe behavior are insufficient.
- Honor pointer-plus-length inputs, including empty strings, embedded NULs, and non-ASCII text. Validate dimensions and arithmetic before allocating or narrowing sizes. Contain failures at language boundaries.
- Preserve the device policy: AUTO selects the host GPU; unavailable accelerators return errors. CPU must be explicitly selected. Mock is for explicit smoke tests and must never impersonate a catalog model.
- Verify separate-engine isolation, including allocator callbacks and error storage. A mutex around one handle does not protect process-wide state used by other engines.
- Preserve model tokenizer, pooling, normalization, truncation, and output-layout semantics. An unsupported option must be reported rather than silently ignored. Compare token IDs as well as output vectors when optimizing tokenization.
- Model tokenization must be correct for supported text inputs. GPU tokenization and richer analysis may remain explicit unsupported capabilities. Initial chunking can be native CPU work with model token budgets and original source offsets.
- The C header declares ABI v1 frozen. Do not silently change layouts or semantics. Flag contract defects and propose a compatibility strategy before changing the ABI. Provider registration currently returns NOT_IMPLEMENTED.
- Keep future FFM and JNI adapters thin. They should share native semantics and enforce resource lifetime, string encoding, error handling, and thread ownership. Android needs its own verified runtime/build support.

## Validation

- Baseline: `cargo test --locked --workspace`. Rust, a C++17 compiler, and `protoc` are needed. Review `.github/workflows/ci.yml` for the current CI commands and feature coverage.
- Formatting check: `cargo fmt --all -- --check`. Default-feature lint: `cargo clippy --locked --workspace --all-targets -- -D warnings`. Do not substitute `--all-features`; it pulls platform-specific dependencies.
- Use targeted existing tests for the change. Inspect Make prerequisites before running targets that fetch models or start hardware work. Bound benchmarks by a stated machine, workload, and finish line.
- Report failed, ignored, conditionally skipped, and unexecuted tests separately. A serial rerun is diagnostic evidence, not proof that the default parallel command passes.
- GPU validation needs the actual provider, device, model revision, and workload. Report local checks, hosted CI, publication, and deployment separately. Do not overwrite reference goldens to make a regression pass.

## Documentation

- Write for a new human contributor. State what an API does, what it requires, and where its limits are. Use repository-relative links and commands that work outside a particular developer's checkout.
- Separate current behavior, measured results, proposals, and unsupported features. Put dated machine results in receipts or runbooks; do not repeat them throughout API documentation.
- Describe allocation and copy claims precisely: which buffers, counters, transfer direction, batch/sequence sizes, and warmup. Zero arena allocations does not mean zero process allocations or zero data movement.
- Avoid slogans, repeated disclaimers, unexplained phase names, and labels such as LIVE or DONE without scope and evidence. Prefer one maintained explanation with links over repeated status prose.
- Describe our requirements and measured behavior professionally. Do not disparage competing libraries or attribute overhead to a language/binding without measurement. Keep comparative benchmarks scoped to equivalent workloads.
- Keep changes focused. Do not rename the product, rewrite all documentation, commit, push, or publish as a side effect of a review.
