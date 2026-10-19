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

## Forest validation follow-up

Cycle rejection remains part of `Dependencies::check` and `Document::add`.
CoNLL-U dependency layers with cyclic HEAD columns are rejected on insertion;
the prior base accepted them. Direct dependency codec imports also remain
checked and keep returning `Malformed` for cyclic payloads. The decoder calls
the shared validator instead of maintaining a separate chain walker. Graph
import and Document insertion still validate at their respective boundaries.

Walk tags replace the two-pass traversal. The tags use `usize`, avoiding a
narrowing conversion of the node count or traversal identifier. Scratch space
is 8 bytes per node on x86_64, versus 1 byte in the earlier two-pass version.
An all-root forest needs no scratch allocation. The exhaustive test compares
8,477 head assignments through five nodes with an independent per-node search;
a separate test checks Document insertion of a cycle beside a rooted component.
The existing 100,000-node chain, closed cycle and corrupt-wire tests still pass.

Performance was measured on AMD Ryzen 9 9950X3D, x86_64, rustc 1.99.0,
in release mode pinned to CPU 0. Construction is outside the timed region.
Each process warms up 20 times, then reports median and p10 of 41 validations.
The 1,000,000-token, 25-token-chain case uses five ABBA rounds for base/new
and earlier-head/new. The medians of process medians are:

- Base `ad85be5e` (no cycle guard): 0.455 ms.
- Earlier head `350cdbac` (two-pass guard): 2.428 ms.
- Follow-up walk tags: 1.765 ms, 27.3% less time than the earlier guard.
- Remaining cost versus base: 1.311 ms per million tokens (1.311 ns/token),
  or 3.88 times the bounds-only check. This is the cost of a stronger invariant,
  not performance parity with base.

Three additional ABBA rounds compare the earlier guard with walk tags:
all roots, 0.838 to 0.381 ms; one 1,000,000-node chain, 3.325 to 1.968 ms.
These synthetic validation timings do not measure parser or import throughput.
Raw process results are in `cycle-performance.log`; machine load can affect them.

To reproduce a process run from the repository root:

```sh
CARGO_TARGET_DIR=target cargo build --locked --release -p turbo-analysis-layers --example dependency_check_bench
taskset -c 0 target/release/examples/dependency_check_bench 1000000 25
```

The example source is `analysis/layers/examples/dependency_check_bench.rs`.
For comparisons, copy that example to the same relative location on each clean
revision, build each revision, save the binaries, and alternate the saved
binaries in ABBA order. Keep build artifacts inside the current worktree.
Use widths 1 and 1000000 for the additional forest shapes.

Follow-up gates: fmt, required clippy, 140 parser/layer tests, and 3 repository
file-reference checks pass. Sanitized output is in the `cycle-*.log` files.
