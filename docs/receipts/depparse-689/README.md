# Feedforward dependency receipts

The red API test was run on the preceding parser implementation before the
neural API was added. It fails with unresolved FeedforwardContext and
FeatureLayout imports. This is a new API red/green receipt, not a claim of
an existing behavior regression. The green suite includes that same test.

Independent reference evidence: the pinned, unmodified Java sources compile
and regenerate all five committed fixtures byte-for-byte. The Rust parity
test compares every weight, feature sequence, score and parse against those
fixtures, including seeded training with dropout and L2. No Java/network is
needed by the normal Rust test suite.

Checks:

- `cargo fmt --all --check`: PASS.
- `cargo clippy --locked -p turbo-depparse -p turbo-train-components --all-targets -- -D warnings`: PASS.
- `cargo test --locked -p turbo-depparse -p turbo-train-components`: 160 passed,
  0 failed, 1 pre-existing ignored test.
- `cargo test --locked -p turbo-bench --test tree`: 3 passed.
- Every touched Rust file is below 500 lines. No added runtime regex dependency,
  literal em dash, private hostname, or private worktree path.

Coverage includes directional first/second children and grandchildren,
Java-compatible legacy slots, raw/ZIP round trips, all truncations of a
2,800-byte model, oversized counts, duplicate IDs, non-finite weights,
missing actions, invalid settings, exact cache parity, cube/ReLU learning,
beam decoding, token/word annotation, provenance, and real UD Spanish input.
The synthetic training fixture is a parity test, not an accuracy benchmark.

The versioned ZIP is a Turbo format. The pinned Java loader accepts the raw
ONLP-FFDP-1 stream, which has explicit read/write methods. Expanded 18-slot
and ReLU models use ZIP; legacy 14-slot cube models support both formats.
