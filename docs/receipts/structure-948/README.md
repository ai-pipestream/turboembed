# Exact-content column detection receipt

The performance correction keeps the block features of #948 and changes
only the rejection check before plain-text column alignment. It checks
tabs and adjacent spaces together, on the original content slices. The
x86-64 implementation uses compile-time-enabled SSE2; other targets use
the portable word scan. It adds no dependency. Neither table alignment
nor list tightness is disabled in the shipped implementation.

## Inputs, builds and output identity

The inputs are the same real documents as the original receipt:

- `docs/analysis.md` at `0edf2f35ce710bc5d04fd3924c61bd992aad017e`,
  324,322 bytes, SHA-256
  `1b3145c00bcde19da0ba88a49cb3b5eb4ef5c84ec74533374f1318c65309b28c`.
- UD English EWT r2.16 training sentences, joined with two newlines and
  one final newline, 1,021,035 bytes, SHA-256
  `f68a568dd94566c9d5b8b207bdd5c9603f281fdea09239136fb3bca76ee14676`.
  The source is CC-BY-SA-4.0, SHA-256
  `3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b`.

No corpus copy is committed. The original conformance document records
the corpus sources and licences.

The measured implementation is
`ba8e6f1e77c2b8bcc5e7745a2bab0c5baa7fdf94`, with main
`6f7a6ea6a16d1fc36972d57a883c67255146887e` and the blocked feature at
`f3b5ac0a6aca3bdeb0ec82b4648c457009b636cc`. The feature baseline is the
original PR merged normally with that same main. All three builds use
the identical benchmark source, locked dependencies and release profile.

A later normal merge synchronizes the implementation to main
`bf52319dfc49486e5109f2c76faccb8fcd6edc02` at
`bd19bfddbaa41739cbdf5a31bbbf794e3eebeeb7`. Both main and implementation
release binaries were rebuilt and remained byte-identical to the measured
binaries. The source tree and artifact hashes are in
[synchronization.json](synchronization.json). The guard campaign uses
these synchronized commits.

Every case first writes the actual span, block kind, parent and depth
columns at `ViewId::SOURCE`. The blocked feature and corrected feature
have identical complete output hashes on all three cases. Pre-feature
main is not claimed to have identical output:

- Analysis as plain text: 833 blocks and zero tables in both. Exactly one
  `List` becomes `LooseList`, with unchanged span, parent and depth.
- Analysis as Markdown: 2,329 blocks and 18 tables in both. One list flag
  differs.
- EWT prose: 12,834 blocks and zero tables in both. Twenty-five list flags
  differ.

The canonical output format is one row per node: kind, byte start, byte
end, optional parent index and depth. Full output hashes and byte counts
are recorded in each campaign's provenance. The correction also produces
identical complete trees to the blocked feature for all 652 CommonMark
inputs in both syntaxes, 1,304 comparisons. Their combined SHA-256 is
`c0cbff34126b38008a81aeca1c15aa4dd1ce4edf5d6444480d0bbb75363e63ec`.

## Paired observations

The machine is an AMD Ryzen 9 9950X3D, Linux x86-64, Rust 1.99.0. Processes
are pinned to logical CPU 4. The governor remains `powersave`; no host
settings or isolation were changed. Other work ran on the host. Each
process performs one untimed output parse, three discarded warmups, then
201 timed parses. File loading, output dumping and printing are outside
the timed region. The timer covers `detect` and its result observation;
destruction is outside elapsed time in every variant. Normalization and
`StructureProjector` are not timed.

All six orders of main, blocked and corrected variants are balanced.
The failing plain case has 24 rounds; the other two cases have 12 rounds
each. These are separate campaigns, not a claim of simultaneous load.
Values below are the median of per-process medians in milliseconds:

- Analysis as plain text: main 0.3972160, corrected 0.3845315.
  Ratio of medians 0.9680665; median within-round ratio 0.9661578.
- Analysis as Markdown: main 0.6045485, corrected 0.5274720.
  Ratio of medians 0.8725057; median within-round ratio 0.8717235.
- EWT prose: main 2.7318150, corrected 2.5604055.
  Ratio of medians 0.9372544; median within-round ratio 0.9407939.

Neither median summary shows a slowdown against main on these inputs.
These are local observations, not an isolated-host significance claim or
a speed guarantee on other architectures. Ambient load changes from
69.49 to 67.69 in the plain campaign and 116.16 to 101.66 in the guard
campaign. Individual paired ratios are variable; their ranges and
quartiles are preserved in [summary.json](summary.json).

[vector/pairs.csv](vector/pairs.csv) and [guards/pairs.csv](guards/pairs.csv)
record every process median and order. Their `samples.json.gz` files keep
all 201 samples per process, with load, clock and UTC observations. Their
`provenance.json` files record input, binary, benchmark, dependency-lock
and runtime-source hashes. The decompressed JSON and CSV hashes are
included. The final campaigns contain 144 processes and 28,944 timed parses.

## Attribution and unsuccessful candidates

The original regression was reproduced before correction. Combining the
two substring checks alone still left a roughly 19% paired slowdown.
Whole-physical-line hints then admitted indentation and padding outside
the original trimmed bounds: 926 actual alignment calls, of which only
12 passed the original predicates. The other 914 pairs describe 117,142
admitted span bytes, not a measured number of bytes traversed. Exact
bound checks restored the original 12 keyed pairs but still missed the
performance gate. These attempts and their raw observations remain under
`history/`; the production parser does not retain their hint machinery.

Hardware counter profiling was unavailable under the host's permissions.
The [attribution](attribution/provenance.json) campaign instead uses
isolated diagnostic builds of the blocked feature on the failing input.
Bypassing column detection preserves its complete output and changes the
median from 0.4450565 to 0.3062190 ms, with median paired ratio 0.6910745.
Bypassing list tightness gives 0.4490940 ms, paired ratio 1.0327655;
bypassing both gives 0.3072655 ms, paired ratio 0.7064522. The latter
builds intentionally omit the one list flag and are diagnostic only.
They are not acceptance candidates and their disabled behavior is not
shipped. This comparison points to the column-detection path; it does
not show a saving from removing list tightness. Its 16 rounds, 64
processes, raw samples, source hashes, output hashes and load are retained.

## Correctness and memory bounds

The synchronized relevant suites pass 374 tests with two existing ignored
release-speed tests. This includes all 60 structure tests, graph/server
codec checks, offset projection, and the unchanged 577/652 CommonMark
expectation with the same 75 named gaps. Strict clippy and formatting pass.
The separator tests compare both vector and portable implementations
against the original predicate on real document lines, all tail lengths,
unaligned slices, tabs, cross-vector space pairs, ASCII controls and UTF-8.
Trimmed headers, container prefixes and Unicode padding retain table
behavior.

The vector load receives an exact 16-byte array and never loads the shorter
tail. The target feature is checked at compile time. The load permits
unaligned addresses as specified by the
[Rust intrinsic documentation](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_loadu_si128.html).
The independent Linux x86-64 [guard_check.rs](guard_check.rs) includes the
actual production scanner and places inputs against an inaccessible page.
All 9,313 cases pass for lengths 0 through 96, every tab/space-pair position
and every address alignment. Its output is [guard-check.txt](guard-check.txt).

From the repository root:

```sh
cargo build --locked --release -p turbo-structure --example block_bench
target/release/examples/block_bench TEXT plain 0 output.tree
taskset -c 4 target/release/examples/block_bench TEXT plain 201
rustc --edition 2024 -O docs/receipts/structure-948/guard_check.rs -o target/structure-guard
target/structure-guard
```

Use separate checkouts and target directories for the recorded main,
blocked and corrected revisions, and overlay the same committed benchmark
source on each. Follow the recorded process order rather than timing all
of one build before another. No hosted CI or external review result is
implied by this local receipt.
