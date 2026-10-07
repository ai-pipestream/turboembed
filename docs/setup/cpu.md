# CPU

The `cpu` backend runs on the host processor. It is in every build, so
this is also the first thing to try on any machine: it needs no driver,
and it computes the reference every other backend is checked against.
Backend reference: [../cpu.md](../cpu.md).

## Prerequisites

- Rust, through rustup (stable).
- C and C++ compilers (`cc`, `c++`): the test of the C interface
  compiles the header as C11 and C++17 and builds a C program against
  the library.
- Linux, macOS or Windows, any architecture Rust targets.

```
scripts/setup/cpu.sh             # check
scripts/setup/cpu.sh --install   # rustup and the compilers, when missing
```

## Build and test

```
cargo build --release -p turbo
cargo test -p turbo
```

`turbo_version()` returns `0.1.0 cpu`. The CPU is always the last device
the runtime lists; in a CPU-only build it is device 0.

## A bundle

The sealed test bundle in `testdata/tiny-bert-bundle` is in the tree and
needs nothing. For a real model, use a bundle carrying a
`FORMAT_SAFETENSORS` artifact whose `backends` lists `cpu` (every recipe
in `bundle/recipes/` has one, `weights-f32`): a prebuilt one or one you
make ([bundles.md](bundles.md)).

## Conformance

```
# the sealed test bundle
cargo test --release -p turbo --test conformance -- --nocapture

# a real bundle, at each tier
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 TURBO_TEST_DEVICE=cpu \
    cargo test --release -p turbo --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 TURBO_TEST_DEVICE=cpu TURBO_TEST_PRECISION=exact \
    cargo test --release -p turbo --test conformance -- --include-ignored --nocapture
```

Each run prints, per way of writing the rows, one minus the lowest
cosine and the largest absolute difference against the reference.

## Tiers

Every tier computes in F32 on the CPU, so MODEL, FASTEST and EXACT give
the same vectors. A model stored in F16 or BF16 is refused at MODEL and
widened to F32 once at FASTEST and EXACT.

## Environment

| Variable | Meaning |
|---|---|
| `TURBO_CPU_THREADS` | Threads a session computes on, 1 to 1024. Unset: one per processor the process may run on (its affinity mask and cgroup quota). Any other value refuses the session. |

The threads are not pinned; set the process's affinity
(`taskset -c 0-15 <program>`) and they inherit it. The vectors do not
depend on the thread count.

The kernels are chosen per model from what the processor has: AVX-512
(F and VL), else AVX2 with FMA, else portable Rust. The AVX-512 and AVX2
kernels give the same bits. `scripts/setup/cpu.sh` says which set runs.

## Limits

- The linear layers' weights are kept a second time, packed for the
  kernels: about half the weights' size again, shared by the model's
  sessions.
- `turbo_runtime_select` never picks a CPU, so in a CPU-only build it
  returns `TURBO_E_DEVICE_NOT_FOUND`: name the CPU's device index
  instead (with the gRPC server, `device=0`).

## Serving

`cargo build --release -p turbo-kserve`, then [../grpc.md](../grpc.md).
