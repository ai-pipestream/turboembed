# Core boundary and module-split receipts

Base: `ca4924e9`. Package: `turbo` (the core directory's package name).

The public Rust exports and C layouts remain at their existing entrypoints. Tokenizer and session implementations now live in their respective module directories; C API functions are grouped by lifecycle. All 23 changed Rust files are below 500 lines; the largest is `core/src/lib.rs` at 448 lines.

## Red and green

Each red log records a test failure before its corresponding behavior fix, after the mechanical split. The final suite includes these same tests.

- `red-added.log`: a repeated added-token record hid an omitted special token. The unchecked loader accepted it. Unique content is now required.
- `red-special.log`: upstream retains an added token marked ordinary while skipping special tokens. The loader previously accepted the disagreement with the manifest. It now requires the special flag.
- `red-text.log`: two oversized text views overflowed the byte-total calculation before validation or dispatch. Metadata is now checked first; aggregate estimates saturate and initial token reservations are bounded.
- `red-span.log`: the extracted span-extent arithmetic narrowed text plus prompt length to u32, wrapping at the limit. The test exercises the integer calculation without allocating gigabytes. Checked addition and conversion now reject the extent before encoding.

Additional boundary tests cover valid maximum extents, byte sizes rather than element counts, zero-length views, null pointers with nonzero length, and output-span widths. Shared checks now guard FFI row strides, text and range arrays, and session input/output dimensions before constructing slices or allocating backend state. Existing buffer-allocation out-of-memory behavior is unchanged.

## Final validation

- `cargo fmt --all --check`: passed (`fmt.log`).
- `cargo clippy --locked -p turbo -p turbo-bench --all-targets -- -D warnings`: passed (`clippy.log`).
- `CARGO_TARGET_DIR=target/core cargo test --locked -p turbo`: 300 passed, 10 ignored (`tests.log`).
- `cargo test --locked -p turbo-bench --test tree`: 3 passed (`tree.log`). No stale README or docs paths needed replacement.
- `cargo check --locked -p turbo --no-default-features`: passed (`no-default-features.log`).

Core tests use a separate build directory as the repository README requires. Existing ignored tests require optional reference corpora, hardware, or are fixture-generation helpers; they were not counted as passes. This validation does not claim accelerator execution or performance improvement.
