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
