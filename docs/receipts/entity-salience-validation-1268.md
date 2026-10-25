# Entity salience review and validation fixes (#1268 / #1201)

Independent reviewed head: `af91d1cb9aabfcda9eebb12b737586bff64424cd`.
Review scope: the salience commit after
`09de765a87acb5932cf1903efcfa88a25eb22f2f`, with its #1242 prerequisite
inspected for the source references the scorer and headers consume.
The implementation and prerequisite were Cursor work. The review found
no adopted implementation from this reviewer.

Original-head verdict: BLOCK, recorded in
[the review](https://github.com/ai-pipestream/turboembed/pull/1268#issuecomment-6392426128).
An independent production harness had two passing tests and four failing
validation tests: wrong-view sentence inputs were silently discarded,
a partial salience layer imported, an outside consolidated mention
imported, and invalid weights/damping returned valid-looking scores.

Fix code commit: `5f545590f0b0b77833b4370c5eab0aa5b7a2ab9d`.
The checks below ran from that clean code commit. This receipt is a
subsequent documentation commit. The fixes are a separate dependent PR
for external review; no PASS is claimed on this reviewer's own changes.

## Changes and evidence

- Configuration validation checks finite, nonnegative weights with a
  positive total, finite nonnegative decay/bonus, damping in `[0, 1]`, and
  positive iterations. A wider weighted-mean accumulator preserves small
  and large common weight scales. The scorer's producer version is 2;
  `score` accepts source text and returns a `Result`.
- Scoring rejects wrong-view optional inputs. Heading bonuses use actual
  fragments, excluding mentions inside an envelope's gap. Enabled headers
  reject mismatched document-entity views and incomplete scores.
- Salience insertion/import requires exactly one score per consolidated
  entity. Document entities validate the actual view, UTF-8 coordinates
  and source entity envelope/reference, including named dependencies.
  Reversed mention coordinates return a decode error without panicking.

The 12 tests in `salience_validation.rs` use the real newline detector,
tokenizer, dictionary name finder, consolidator, salience scorer, bundle
tokenizer, chunker and salience-aware header step. The pipeline produces
three chunks and the final header `[Entities: Beta | in scope: Acme]`
with carried entities disabled, proving document-salience injection.
Complete binary graph and provenance equality passes, including named
source entities and named salience dependencies. No mock step or provider
was substituted.

Corruption controls cover all five components with NaN, either infinity,
negative and above-one scores, invalid entity/rank fields, missing rows,
outside/UTF-8/reversed mention coordinates, unknown view IDs, wrong source
references and fragment gaps. Configuration controls include zero and
invalid weights, invalid decay/bonus/damping/iterations, and common weight
scales of `1e-30` and `1e30`.

## Checks

Environment: `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`; existing build directories reused.

- `cargo test --locked -p turbo-analysis-layers -p turbo-entity-linker -p turbo-chunker`:
  514 passed, zero failed, one existing ignored timing test
  (`entity::build::cost_tests::timing_crowded_sentence`).
- `cargo clippy --locked -p turbo-analysis-layers -p turbo-entity-linker -p turbo-chunker --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `cargo test --locked -p turbo-bench --test tree --test provenance`:
  six tree and seven provenance tests passed, zero failed or ignored.
- Largest Rust file touched by the validation fixes: 351 lines. Seven
  development dependencies are existing workspace crates; no external
  dependency or regex was added.

No full-workspace run is claimed. The benchmark tool test that measures a
CPU record was excluded under the instruction against timing runs. No
training, inference or hardware campaign ran. Models, bundle manifests,
corpus checksums, acceptance thresholds and persistent settings were not
changed. Per-document allocations remain as documented in
[entity-salience.md](../entity-salience.md); the allocation-free steady-state
goal is still partial.
