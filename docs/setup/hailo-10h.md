# Hailo-10H on a Raspberry Pi 5

The `hailo` backend runs on Hailo accelerators through HailoRT. A Hailo
board runs a graph compiled ahead of time for its architecture, a HEF,
in 8-bit integers. This page covers a Hailo-10H on a Raspberry Pi 5. Backend reference:
[../hailo.md](../hailo.md).

## Prerequisites

On the Pi (Raspberry Pi OS, 64-bit):

- `hailo-h10-all` from the Raspberry Pi repository: HailoRT 5.1.1, its
  headers, `hailortcli`, the `hailo1x_pci` driver and firmware. Reboot
  after installing.
- Rust and a C++17 compiler.

```
scripts/setup/hailo.sh --device hailo10h             # check
scripts/setup/hailo.sh --device hailo10h --install   # apt-get install hailo-h10-all, on Raspberry Pi OS
```

The check finds the board on the PCI bus (`1e60:45c4`), the driver, the
`/dev/hailo*` node, HailoRT's header and library, and that HailoRT is
5.x. `hailortcli scan` should list the device.

Making a HEF needs an x86_64 Linux machine with Docker and Hailo's
Dataflow Compiler 5.x, which needs a Hailo Developer Zone account
(below).

## Build

On the Pi:

```
cargo build --release -p turbo --features hailo
```

| Variable | Meaning |
|---|---|
| `TURBO_HAILORT_ROOT` | HailoRT's prefix, with `include/hailo/hailort.h` and `libhailort.so` in `lib/`, `lib/<arch>-linux-gnu/` or `lib64/`. Unset: `/usr`, where the Pi packages put it. |
| `CXX` | The C++ compiler. Unset: `c++`. |

The library links `libhailort.so` with its directory as a run path.
`turbo_version()` names `hailo`. Load the driver before a program
starts: the device list is made once per process.

## A bundle

The backend loads `FORMAT_HEF` artifacts only. The all-MiniLM-L6-v2
recipe's `hef-hailo10h-s128` is one: target `hailo10h`, I8, one row of
128 tokens a frame, from the word-embedding lookup (gathered on the
host, from the bundle's F32 table) to the hidden states. No prebuilt
bundle carries a HEF, because the compiler may not be redistributed.

To make it, on an x86_64 machine:

1. Download the Dataflow Compiler 5.x wheel (Linux x86_64) from Hailo's
   Developer Zone into `bundle/hailo/`.
2. `scripts/setup/bundle-tool.sh --hailo --install` builds the
   `turbo-hailo-dfc` image from `bundle/hailo/Dockerfile` and prints its
   id.
3. Put `turbo-hailo-dfc@<id>` in the HEF artifact's
   `produced_by.container`, and the reference image's id in
   `reference.produced_by.container` ([bundles.md](bundles.md)).
4. `cargo run --release -p turbo-bundle -- make bundle/recipes/all-minilm-l6-v2.json models/upstream/all-minilm-l6-v2 models/bundles/all-minilm-l6-v2`

The compile quantizes on the calibration texts the recipe carries and
runs twice to record whether it is reproducible. Copy the sealed bundle
directory to the Pi.

## Conformance

```
export TURBO_TEST_REQUIRE_HAILO=1

cargo test -p turbo --features hailo
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 \
    cargo test --release -p turbo --features hailo --test hailo -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 TURBO_TEST_DEVICE=hailo \
    cargo test --release -p turbo --features hailo --test conformance -- --include-ignored --nocapture
```

I8 is held to cosine 0.93 against the reference. Reference cases longer
than 128 tokens are checked to be refused and are not compared. HailoRT
writes `hailort.log` into the directory a program runs in.

## Tiers

| Tier | Computes in |
|---|---|
| MODEL | I8, as compiled |
| FASTEST | I8, the same path |
| EXACT | refused: a HEF never computes in F32 |

## Environment

The backend reads no variable of its own.

## Limits

- I8 only; no EXACT.
- One row of `fixed_seq` tokens (128 in the recipe) per frame. A text
  longer than that is refused with `TURBO_E_CAPACITY` unless the caller
  cuts it with `turbo_embed_options.max_tokens = 128`; with
  `max_tokens` 0 the core cuts at the bundle's `max_seq` (256), not the
  HEF's.
- BERT-family models with token type 0 only.
- Host buffers only.
- A frame that does not come back within 10 seconds fails the run, and
  the session then refuses further work.

## Serving

`cargo build --release -p turbo-kserve --features hailo`, then
[../grpc.md](../grpc.md), with each model's `max_seq` set to 128.
