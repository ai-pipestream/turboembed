# Progressive snapshot repair (#1249)

The original Argon progressive implementation and history from
`c8eda296e41f9487e6d4e19e09e99845c3a2d954` are retained. This adoption
repairs the independently reproduced invalid intermediate graph reported
in the exact-head review. It preserves the progressive wire protocol,
failure reporting, final batch equivalence, cancellation and named views.

## Commits

- Normal main merge: `ed154211e9fb9869cbc5762b76b0341a5d48f272`.
- Incorporated main: `70558814e3171cd7e66441e4ebfed9aa14e5c076`, verified
  against the live remote before merging. The fetch mapping covers all
  branches; the main ref was also fetched explicitly.
- Regression-only commit: `c5e095294734c9735a09a038edcd6f95572f0036`.
- Clean tested implementation: `a3622b614f6ba37a04906c76d4510a9a4e7378bd`.

## Correction

The failure cascade previously stored a skipped descendant while another
required producer was still running. Its provenance named an absent input.
A successful sibling then exported that skipped layer in a graph snapshot
that encoded successfully but failed canonical import.

The cascade still reports blocked descendants immediately and propagates
that knowledge transitively. A descendant whose dependencies are ready is
still stored immediately. Otherwise it remains queued for normal
scheduler readiness; its skipped outputs are stored only once the input
producers have settled. Normal skip completion can therefore produce a
later layer batch with a skipped status. The existing protocol already
carries that status. No inputs, provenance references or layer records
are fabricated or silently discarded.

Both ordinary execution and deferred skips use the same readiness check.
View serialization, cancellation checks and final canonical entry order
are unchanged. No protocol fields, public options or error codes change.
The existing generic optional-producer ordering concern is not expanded
into this repair.

## Permanent red and green regressions

`server/tests/progressive_snapshots.rs` exercises the actual core scheduler
and an actual tonic RPC channel with production normalization, sentence
detection, tokenization, term analysis, keyword and structure annotators.
No mock annotators, fault hooks, sleeps or elapsed-time assertions are used.

The real seed contains NFC-view tokens and named sentences, with no
default sentence layer. Term analysis against the source view fails while
the default sentence producer is still running. Before the repair:

- The scheduler test fails because its failure and sibling callbacks expose
  skipped keywords referencing absent default sentences.
- The RPC test fails because the successful structure layer batch carries
  the same invalid graph.

Both report: `layer turbo:keywords refers to opennlp:sentences, which the
graph does not restore`. Both tests now pass. They import every snapshot,
retain immediate skipped-dependent reporting, check terminal skipped and
failed statuses, and compare final graph bytes with ordinary batch output.

## Validation

All checks ran in an isolated checkout and target with jobs 2,
incremental compilation disabled and development debug info disabled.

- `cargo test --locked -p turbo-analysis -p turbo-kserve --lib --tests`:
  329 passed, 0 failed, 2 existing release-only ignores across 34 targets.
- `cargo clippy --locked -p turbo-analysis -p turbo-kserve --all-targets -- -D warnings`:
  passed.
- `cargo fmt --all -- --check`: passed.
- `cargo test --locked -p turbo-bench --test tree --test tool --test provenance`:
  tree 6, tool 7, provenance 7 passed, with no failures or ignores.
- Original external production probes: 8 passed, 0 failed, 0 ignored. This
  includes the formerly failing snapshot regression, 27 real-channel
  profile/Unicode/offset requests, mixed-document failure isolation,
  named NFC/NFKC instances, real time/date optional-cycle equivalence,
  cancellation, request refusals and subsequent service availability.
- Diff, public prose and Rust size checks passed. The repaired scheduler
  module is 263 lines and the permanent regression file is 130 lines.
  The largest feature Rust file is the pre-existing document module at 494.

The server's private cancellation counters are not exposed; downstream
stopping is asserted at the real core API, with RPC drop and subsequent
service availability exercised separately. Hosted CI and independent
review remain separate from this local proof. No timing or hardware sweep
was run. Unpublished alternatives and probes remain local.

This is an implementer receipt. The resolved finding is withdrawn for the
published descendant, which requires fresh external reviews and hosted CI.
There is no self-PASS or merge; issue #1218 remains open until landing.
