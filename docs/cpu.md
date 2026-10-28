# The CPU backend

The host processor, through `turbo_backend.h` like any other backend. It
is always built (the `cpu` feature, on by default) and lists one device.

## Threads

A session computes on a pool of threads it starts when it is created
and joins when it is released; between runs they sleep. A run is split
into tasks the threads take: the embeddings and each LayerNorm by chunks
of tokens, each linear layer by tiles of its output, attention by row,
head and block of queries. The thread that calls `turbo_session_run`
takes tasks too. Nothing is allocated per run, on any thread.

The pool has one thread per processor the process may run on, as
`std::thread::available_parallelism` counts them when the session is
created: the process's CPU affinity mask and its cgroup's CPU quota,
where the system says. `taskset -c 0-15 <program>` gives 16.

`TURBO_CPU_THREADS=<n>`, read when a session is created, sets the count
instead: 1 to 1024. Any other value refuses the session with
`TURBO_E_INVALID_ARGUMENT`. It does not pin the threads; for that, set
the process's affinity (`taskset`, `sched_setaffinity`), which they
inherit. `turbo-bench record --cpus` does both (docs/benchmarks.md).

`turbo_session_desc.cpu_threads` sets the count for one session: 1 to
1024. When it is not 0 it wins over `TURBO_CPU_THREADS`, which is then
not read. A value over 1024 refuses the session with
`TURBO_E_UNSUPPORTED_OPTION` naming field 6.

`TURBO_CPU_STATIC_TABLE`, read when a static model's session is
created, chooses the table it sums (docs/static.md): `stored` the table
as the bundle stores it at every precision, `i8` the I8 copy at every
precision. Unset, `TURBO_PRECISION_FASTEST` sums the I8 copy and the
other precisions the stored table. Any other value refuses the session
with `TURBO_E_INVALID_ARGUMENT`.

`TURBO_CPU_STATIC_PAGES`, read when a static model's first session is
created, chooses where its tables are read from: `mapped` (or unset) the
stored table where the bundle's file is mapped, and the I8 copy on the
heap; `huge` copies of both on 2 MiB boundaries, which Linux is asked to
back with transparent huge pages (`madvise`), so that gathering rows
from anywhere in a large table misses the TLB less. The copy costs the
table's size in memory and the time to copy it, once per model; the
vectors are the same bits either way. Any other value refuses the
session with `TURBO_E_INVALID_ARGUMENT`.

The vectors do not depend on the count: no sum is split between tasks,
and each output is computed in one fixed order whichever thread takes
it, so a batch gives the same bits on any number of threads.

## Late pooling

After `turbo_embed_write_ranges`, a run encodes every row once, as for
`turbo_embed_write_tokens`, and then pools by range instead of by row.
Each range's vector is computed from the final hidden states where the
encoder left them, with nothing copied: the mean of the range's tokens
whose mask is 1, summed in F32 in column order, then cut to
`output_dim` and normalized with the same code a row's mean pooling
uses. A range over a whole row gives that row's mean vector, bit for
bit. The ranges are split into tasks of several ranges each on the
session's threads; each range is summed whole by one task, so the
vectors are the same bits on any number of threads.

For a static model, a range's vector is the mean of the table rows of
its live tokens, times their weights where the model has them: the sum
a row's mean pooling makes, over the range's tokens alone, rounded and
normalized the same way, from the stored table or the I8 copy as the
session chose.

The session's output buffer holds `max(max_batch, max_ranges)` vectors
of the model's width, allocated when the session is created, so a run
of either kind allocates nothing.

## Kernels

The instruction set is chosen once per model, when its first session is
made: AVX-512 (F and VL) where the processor has it, else AVX2 with FMA,
else portable Rust. The AVX-512 and AVX2 kernels give the same bits.
Everything is computed in F32; F16 and BF16 weights are widened once.

The linear layers' weight matrices are copied once per model into the
panel layout the matrix kernel reads: for all-MiniLM-L6-v2, 42.5 MB
beside its 90.9 MB weights file. Sessions share it, and it goes with the
model.

## Benchmark records

`benchmarks/records/x86-64-4d6f19dd.cpu.*` are the CPU backend's
records (docs/benchmarks.md), made at library commit `9048085a` on the
x86_64 host of Machine B, an AMD Ryzen 9 9950X (16 cores, 32 threads),
Linux. Both sides ran on one CCD's 16 threads (`--cpus 8-15,24-31`, so
`TURBO_CPU_THREADS=16`), 20 untimed and 200 timed runs. The reference
is TEI 1.9.4's CPU image
(`text-embeddings-inference@sha256:2538ea1c9640d3763b15af668039d24172d063b42337b0c27796fc2be180c78d`)
in float32, timed per request from sending it to reading the whole
response. p50 in ms, 32 rows:

| Model | Seq | Rows | EXACT | MODEL | TEI at EXACT | TEI at MODEL |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 256 | mixed | 18.5 | 18.7 | 126.5 | 146.7 |
| all-MiniLM-L6-v2 | 256 | dense | 116.1 | 116.7 | 189.0 | 189.7 |
| bge-small-en-v1.5 | 512 | mixed | 66.6 | 66.7 | 766.5 | 901.8 |
| bge-small-en-v1.5 | 512 | dense | 548.4 | 553.3 | 1412.8 | 1420.1 |
| bge-base-en-v1.5 | 512 | mixed | 222.5 | 224.2 | 1929.7 | 1687.0 |
| bge-base-en-v1.5 | 512 | dense | 1800.8 | 1770.1 | 2833.5 | 2857.2 |
| bge-large-en-v1.5 | 512 | mixed | 769.0 | 766.5 | 6849.7 | 5786.5 |
| bge-large-en-v1.5 | 512 | dense | 5980.7 | 5978.2 | 9884.8 | 9812.6 |

The library's lowest cosine against each bundle's reference vectors is
at least 0.99999999998 on every record. The mixed records back
SUPPORTED for EXACT and MODEL on that processor, with `speed_ratio`
0.07 to 0.15. FASTEST has no record, so it stays EXPERIMENTAL. Mixed
rows are mostly padding, which the library does not compute (each mixed
record computes only its live tokens), and their ratios are lower than
the dense records' 0.39 to 0.64.

`benchmarks/records/aarch64-749c9420.cpu.*` are the M2's (Apple M2,
macOS, the NEON kernels), made at library commit `fb5c4217a199` with
`turbo-bench record --bundle <bundle> --device cpu --precision
<exact|model> --rows <mixed|dense> --no-tei` from a release build, 20
untimed and 200 timed runs, 32 rows of 256 tokens, TEI not run. Their
bundle is all-minilm-l6-v2 sealed as manifest `3c305237`, not the
`0f17adc4` seal of the x86_64 and Arm Linux records, so the M2's figures
are not comparable with theirs. The M2 is a desktop and was in use: at
the batch's checks just before it started, the 5-minute load average
read 2.8 to 3.6, with the desktop's apps running, and the records do not
carry that load. min / mean / p50 / p99 in ms:

| Rows | EXACT | MODEL |
|---|---|---|
| mixed | 187.4 / 211.5 / 207.4 / 275.5 | 196.0 / 220.7 / 216.1 / 272.2 |
| dense | 1311.2 / 1392.3 / 1382.6 / 1522.8 | 1279.5 / 1387.5 / 1379.8 / 1572.5 |

The slowest MODEL mixed run took 416.6 ms, twice its median. The
library's lowest cosine is 0.9999999999982 on every record. With no
reference measured they back no cell.
