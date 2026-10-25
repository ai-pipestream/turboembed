# Quotation validation adoption (#1244)

The original quotation implementation and history from
`2caeb04bdf4895a127d59dda4607db49bdedc049` are retained. This adoption
repairs the three public-API correctness findings from its independent
review. It does not change the attribution sieves for previously valid
inputs or the documented unchecked raw speaker-chain row.

## Commits and scope

- Normal main merge: `ecc1642b5bbb60528275c62ad30b90248758bd93`, incorporating
  main `debd00a31a62c44e1f9c5244b689e057572152b3` without conflicts.
- Regression-only commit: `0f3b5e99df7dad1d12ae2897d42e2f19114b7ff9`.
- Clean tested implementation: `34dfeb916d7a4581b7af433480f3340939407bde`.

`Quotations::check` rejects reversed or empty secondary spans, and its
existing document-check hook validates speaker/cue bounds and UTF-8
boundaries against the content view. Cue-window ends use saturating
arithmetic before intersecting with the available tokens. Supplied
wrong-view mentions return an explicit annotator error. Annotator version
2 records the changed behavior in provenance and the configuration hash.
No codec schema, chain reference representation or dependency-scan
optimization is introduced.

## Red and green evidence

`analysis/coref/tests/quotation_validation.rs` uses production sentence
detection, tokenization and normalization. Dependency and mention fixtures
are hand-annotated valid layers; this is not a trained-model accuracy
claim. Its helper is `analysis/coref/tests/quotation_support/mod.rs`.

Before production changes, the permanent target had three passing
controls and three failing regressions:

- Both speaker and cue accepted each of an interior UTF-8 boundary,
  out-of-bounds end and reversed span during `Document::add`. Exported
  graph bytes then failed their own import.
- Cue distances `usize::MAX` and `usize::MAX - 1` overflowed on actual
  pre-quote and post-quote paths.
- Correctly anchored named mention/chain inputs in another view were
  silently ignored through the actual Analyzer and direct annotator API.

After repair all six tests pass. Controls retain exact UTF-8 surface and
dependency attribution, confidence and chain 0; graph/provenance
round-trip; malformed codec data rejection; typed group member bounds;
and the explicitly unchecked raw `u32::MAX` speaker chain. The original
external public-API probes also pass all six tests.

## Validation

All commands used an isolated checkout and target with
`CARGO_BUILD_JOBS=2`, `CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=0`.

- `cargo test --locked -p turbo-analysis -p turbo-analysis-layers -p turbo-coref --lib --tests`:
  514 passed, 0 failed, 0 ignored across 48 targets, including the six
  permanent regressions and the original quotation/evaluation tests.
- `cargo clippy --locked -p turbo-analysis -p turbo-analysis-layers -p turbo-coref --all-targets -- -D warnings`:
  passed.
- `cargo fmt --all -- --check`: passed.
- `cargo test --locked -p turbo-bench --test tree --test tool --test provenance`:
  tree 6, tool 7 and provenance 7 passed; no failures or ignores.
- `git diff --check`: passed. Every quotation adoption Rust file has
  fewer than 500 lines. Public prose has no private checkout paths,
  literal em dashes or attribution trailers.

No timing or hardware sweep was run for this adoption. The previously
documented dependency-scan work limit remains a nonblocking follow-up.
Unpublished alternatives and benchmark probes were retained locally.

The three original findings are resolved by the implementation evidence.
This is an implementer receipt, not an independent PASS. The published
descendant head requires fresh external reviews and hosted CI before
merge. Issue #1202 remains open until landing.
