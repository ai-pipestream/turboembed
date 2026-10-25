# Linear/Head main integration (#1183 / #1085)

## Current main sync

Published parent: `a2e791e670c909af287119eccad35b7e30d395d7`.
Live-verified main: `098b6e7b92fabaaf2eca487c045dddb4ab82990c`.
Normal merge: `d5f954b0f849e57856b70d71fc97d1b3c36918b1`.
Clean tested code: `22157c51296e9ff154d1a5913c68cd142e976a6b`.

No competing active integration or newer head was present at the ownership
refresh. This sync uses the preserved adoption checkout and a distinct
build output with two workers; no other chat's checkout or target is used.

The head-test module conflict keeps the adoption's extracted tests and
copies main's new softmax test into that module without changing its
assertions. Main's shared `softmax_f64_in_place` implementation is retained.
The two new conformance MLM layers now use `Linear::new(..).unwrap()` in
place of struct literals, with identical dimensions, weights and biases.

The packed Linear implementation, its cache/concurrency tests, MLM and
SPLADE decoder-cache code, SaT accessors and timing harness are byte-identical
to the published parent. The Head production delta is only main's shared
softmax helper. The entire CPU source subtree, including all NEON paths,
is byte-identical to main (tree `02a190e7014bb97320d12e178a22ae076dc14a83`).
No arithmetic, numeric tolerance, fixture or acceptance threshold is edited.

### Red/Green

**Red** at normal-merge commit
`d5f954b0f849e57856b70d71fc97d1b3c36918b1`:
`cargo check --locked -p turbo --test conformance` fails at
`core/tests/conformance/tasks.rs:295` and `:304` because private Linear
fields cannot be constructed through struct literals. These are the two
reported integration errors, reproduced before the constructor migration.

**Green** at clean code
`22157c51296e9ff154d1a5913c68cd142e976a6b`:

- The same conformance check passes.
- `cargo check --locked --workspace --all-targets`: passed.
- `cargo test --locked -p turbo -p turbo-relation -p turbo-gliner -p turbo-doccat -p turbo-analysis-embed -p turbo-schema-extract -p turbo-sat --lib --tests`:
  1,016 passed, zero failed, 31 existing ignores across 146 targets.
- `cargo clippy --locked -p turbo -p turbo-relation -p turbo-gliner -p turbo-doccat -p turbo-analysis-embed -p turbo-schema-extract -p turbo-sat --all-targets -- -D warnings`:
  passed.
- Workspace formatting and diff checks: passed.
- Tree 6, tool 7 and provenance 7 passed, with no failures or ignores.

Named passing regression controls include:

- `mutations_invalidate_the_lazy_packed_copy`.
- `products_never_hold_the_initialization_lock`.
- `shared_first_use_and_warm_products_agree`.
- `batched_logits_match_one_state_at_a_time` (exact equality retained).
- `mutated_weights_are_the_ones_applied`.
- `the_decoder_is_transposed_once_and_kept`.
- `a_changed_decoder_is_read_not_the_transpose_of_the_old_one`.
- `softmax_f64_in_place_is_stable_and_uniform_on_degenerate_rows`.

Validation uses `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`, `RUST_TEST_THREADS=2`, `TURBO_CPU_THREADS=2`, and
the checkout's own `target/sync-verify`. No standalone workspace shares
that output. These are per-command controls, not persistent settings.

The earlier timing table and its accepted tiny-shape overhead remain tied
to their original source commit and are preserved verbatim in the PR body.
No timing or hardware sweep is added; retaining NEON source is not a new
native ARM runtime claim. This is implementation verification, not an
independent PASS. Fresh external reviews and hosted CI are required for
the new descendant. #1085 remains partial and open; no PR merge occurs.

## Prior integration (historical)

The following records the earlier published adoption at
`a2e791e670c909af287119eccad35b7e30d395d7`.

Adopted published head: `8fbd5e3f2f63befd720cb97ef1b529bc9628cf8d`.
Clean tested code: `58928038b4e7d8e6b56b19a529df9a4c479399ff`.
Main: `524ea6fdac30c6cd4fe69a76a920cf0320ae3195`, including #1265
at `3334d2616cc284bbcee816e71e935bfa868896ee`.

### Integration

Normally merged main at `fe4352879d4a0cce7882ae61165e30595cd11748`.
The conflicts in `core/src/sparse.rs` and `core/tests/mlm.rs` combine
main's cached decoder transpose and MlmHead decoder accessors with this
branch's private Linear weights/biases and cache-invalidating accessors.
The transpose loader and invalidation tests now use weight()/bias() and
weight_mut(); main's transpose reuse and replacement behavior is retained.
No conflict resolution changes arithmetic or numerical assertions.

The initial combined build hit main's existing CRF BorrowMut compile
failure. Under the coordinator's explicit prerequisite authorization,
existing owner repair #1281 at
`e520cde169c721b1a032505cbde21a80b9c04050` was fetched and normally
merged at `b91787eb0938353b01a3eaeb1fe6be62b721881c`. Its two-line repair
is recorded separately from main, not represented as already landed.

The whole-workspace check then identified two new direct Linear field
reads in `analysis/sat/src/head.rs`. The final code commit replaces those
with the same read-only accessors, retaining the row slice, label and bias
arithmetic. SaT was added to the affected test and lint set.

Head, Linear, their existing batch/single and cache-concurrency tests, and
the committed timing harness are byte-identical to the adopted head.
The source API migration to private weight/bias fields remains explicit;
the C ABI is unchanged. Old quantitative timing/allocation evidence is
historical, as recorded in the original
[implementation receipt](https://github.com/ai-pipestream/turboembed/pull/1183#issuecomment-6386574574).
No new timing campaign or universal speedup is claimed.

### Checks from the clean code commit

Environment: `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`, using this checkout's own `target/verify`.

- `cargo check --locked --workspace --all-targets`: passed.
- `cargo test --locked -p turbo -p turbo-relation -p turbo-gliner -p turbo-doccat -p turbo-analysis-embed -p turbo-schema-extract -p turbo-sat --lib --tests`:
  944 passed, zero failed, 30 existing ignored.
- Strict all-target clippy on those seven packages: passed.
- Separate `cargo test --locked -p PACKAGE --doc` invocations for each of
  those seven packages: all succeeded, with zero runnable documentation
  examples. No cross-checkout build artifacts were reused.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `cargo test --locked -p turbo-bench --test tree`: six passed, zero failed
  or ignored.

Passing tests include exact batch/single logits, mutation invalidation,
simultaneous first use, lock release before products, and main's decoder
transpose replacement/reuse controls. The largest Rust file in the
feature/prerequisite diff is 483 lines. No dependency, model, fixture,
acceptance threshold or kernel arithmetic was authored by this adoption.

This receipt is a documentation commit after the clean tested code.
Codex authored the prior redesign and this integration, so this is not an
independent PASS. Earlier resolved findings remain withdrawn; fresh
external current-head reviews and completed hosted CI are still required.
No PR merge, branch deletion, reset, clean, rebase, forced push, training,
hardware campaign or persistent setting change occurred. #1085 remains a
partial parent issue for its other shared-math acceptance criteria.
