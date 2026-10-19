# Dependency parser validation

Base: `ad85be5e42c350bfc4c9633cc9c4e317b28ebab5`.
Reference Java commit: `db39bfe35fb28f47a463a74d13e4721020caec15`.

The cycle regression fails on the base layer implementation and passes with
linear-time forest validation. A 100,000-node chain validates without recursion;
closing the chain into a cycle is rejected. See `red-cycle.log` and `tests.log`.
The initial word resolver panicked on a reversed surface-token range. The
regression in `red-anchor-order.log` passes after validating surface order
before resolving forms. Duplicate byte spans remain supported.

Java parity covers 152 contexts (37 features each), oracle transitions and
14 greedy predictions on 7 inputs, including English EWT and Spanish GSD.
Reproduction and attribution: `analysis/depparse/tests/data/README.md`.
The exhaustive small-tree test enumerates head assignments through 5 nodes.
Additional tests cover training, reload, cross-validation, typed navigation,
MWT/view alignment, token/word pipelines, malformed models and non-finite scores.

Gate results are recorded in the adjacent logs. These tests verify behavior;
they do not measure pretrained-model accuracy or throughput.

Environment: x86_64, rustc 1.99.0 (b940084d7 2026-09-28).
Formatting and required clippy command exited 0. Required crate tests, extra
format-adapter tests and all 3 repository-tree checks exited 0. JVM fixture
regeneration completed with byte-identical output (exit 0).

Counts: 117 layer tests and 21 parser tests (including a compile-fail type
check), 258 format tests, and 3 repository-tree checks. The largest touched
Rust source is 467 lines. No runtime regex dependency was added.
