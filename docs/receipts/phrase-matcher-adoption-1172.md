# Phrase matcher main integration (#1172 / #1084)

Adopted published head: `740a77e13287d1301b8089a1899a5c6291d847a3`.
Clean tested code: `9c6d99f5035e998fb73f783d5b234596ac1e95ec`.
Main: `524ea6fdac30c6cd4fe69a76a920cf0320ae3195`, including #1265
at `3334d2616cc284bbcee816e71e935bfa868896ee`.

Main was merged normally at `cf0eb1a5c6174683ea57e980d017a3309769d14a`
without conflicts. The PhraseIndex implementation, IdentityTerms consumer
and subgroup regression test files are byte-identical to the adopted
published head. Main's exports are retained alongside PhraseIndex.

The initial combined build stopped at main's existing E0599 in the two
CRF span calls in `analysis/sequence/src/namefind/crf.rs`. Under the
coordinator's explicit prerequisite authorization, existing owner repair
#1281 at `e520cde169c721b1a032505cbde21a80b9c04050` was fetched and
normally merged. Its two BorrowMut calls are preserved exactly; no
competing CRF implementation was written. The prerequisite is recorded
separately from the main SHA, not represented as already merged on main.

## Checks from the clean code commit

Environment: `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`, with this checkout's own `target/verify`.

- `cargo test --locked -p turbo-analysis -p turbo-doccat -p turbo-matcher`:
  510 passed, zero failed or ignored, including documentation tests.
- `cargo clippy --locked -p turbo-analysis -p turbo-doccat -p turbo-matcher --all-targets -- -D warnings`: passed.
- `cargo test --locked -p turbo --test abi program::a_c_program_loads_a_model_tokenizes_and_embeds`:
  one passed, six filtered. This is the named core program case that failed
  on the old pre-#1240 CI base.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `cargo test --locked -p turbo-bench --test tree`: six passed, zero failed
  or ignored.

The unchanged regressions retain all nested phrase memberships and all
payloads of one phrase shared by multiple subgroups. Longest-only callers
keep `matches_at`; IdentityTerms uses `matches_all_at`. The largest Rust
file in the feature/prerequisite diff is 227 lines. No dependency or
numerical contract was changed by this adoption.

This receipt follows the clean tested code as a documentation commit.
Earlier-head reviews are historical evidence; the new published head
requires fresh independent external reviews. This is an adoption receipt,
not a self-PASS. No PR merge, branch deletion, training, timing campaign,
hardware campaign or persistent setting change occurred. #1084 remains
open while the final phrase-matcher slice awaits integration; the token-key
and ABSA helper work retains its separate landed history.
