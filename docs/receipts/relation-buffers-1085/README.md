# Relation projection buffer receipt

Baseline: main `3cecfd6d045a09c28a885ca40a46959fb2fc2185`.
The fixed source files and both measured binary SHA256 values are recorded
in `timings.json`. The harnesses read the stored upstream reference fixtures;
fixture hashes are included. They do not time the released model or encoder.

`head_timing.rs` runs the same head operations against baseline and fixed
libraries. The baseline uses the allocating APIs; compiling with
`--cfg reuse` selects the new caller-buffer APIs. GLiREL measures 5,000
iterations after 100 warmups; GLiNER2 measures 500 after 20. GLiNER2 inputs
are head-width vectors from the reference query fixture, used to exercise
consistent shapes, not a new text-quality evaluation.

Both libraries were built with `cargo build --locked --release -p
turbo-relation`, separate targets and two build jobs. The harness uses
`rustc --edition=2024 -O`, linking each build's rlibs. `build_harness.py`
records the commands and target layout used. It selects one serde_json
rlib only for loading JSON inside the harness; JSON values never cross the
library boundary.

Seven interleaved pairs reverse order every round. Both binaries run on the
same CPU selected from process affinity, via `taskset`. This is a shared
host, with one-minute load about 52.25 during the sample, not an isolated
throughput study. No device benchmark or timing threshold is added.

| Head fixture | Main median per iteration | Reused median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| GLiREL | 8.5684904 us | 5.8159608 us | 0.678761 | 0.684389 |
| GLiNER2 | 125.054124 us | 71.539112 us | 0.572065 | 0.568354 |

Small checksum differences from moving single-row products to shared Matrix
arithmetic remain within the unchanged upstream numeric tolerances. The
fixture suites compare individual values, not merely these checksums.
Historical full-model speed figures in the model documents are not updated
by these small-fixture measurements.

`head_alloc.rs` separately counts native allocator calls after warmup over
one iteration, excluding stdout initialization. Its results are in
`allocations.json`: GLiREL 33 to 0, GLiNER2 101 to 88. GLiNER2 query
construction still allocates recurrent and attention temporaries. Returned
scores and encoder requests are not claimed to be allocation-free.

Validation: 113 regular relation tests passed; four existing checkpoint
tests were ignored by that command. A separate release invocation ran both
full GLiNER2 checkpoint tests successfully. Both full GLiREL tests stopped
before inference at the existing CC-BY-NC-SA acceptance gate, so no full
GLiREL checkpoint result is claimed. Stored reduced GLiREL reference parity
and warmed allocation tests passed. No model, label, tolerance or licensing
acceptance was changed.
