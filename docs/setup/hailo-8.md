# Hailo-8 and Hailo-8L

The `hailo` backend lists and runs Hailo-8 and Hailo-8L boards the same
way it runs a Hailo-10H ([hailo-10h.md](hailo-10h.md)): through HailoRT,
on a HEF compiled for the board's architecture, in 8-bit integers. What
differs is the runtime generation, the driver and the compiler. Backend
reference: [../hailo.md](../hailo.md).

## Prerequisites

On the machine with the board (a Raspberry Pi 5 with Raspberry Pi OS, or
any Linux host with the PCIe board):

- HailoRT 4.x with its headers and the `hailo_pci` driver, of the same
  version. On Raspberry Pi OS: `hailo-all` (HailoRT 4.23.0). Elsewhere:
  HailoRT and its PCIe driver from Hailo's Developer Zone, which needs
  an account.
- Rust and a C++17 compiler.

```
scripts/setup/hailo.sh --device hailo8             # check
scripts/setup/hailo.sh --device hailo8 --install   # apt-get install hailo-all, on Raspberry Pi OS
```

The check finds the board on the PCI bus (`1e60:2864`, both the Hailo-8
and the Hailo-8L), the `hailo_pci` driver, the `/dev/hailo*` node,
HailoRT's header and library, and that HailoRT is 4.x. HailoRT 4.x and
5.x are separate generations: a Hailo-8 does not run on 5.x, and a
Hailo-10H does not run on 4.x.

## Build

```
cargo build --release -p turbo --features hailo
```

`TURBO_HAILORT_ROOT` names HailoRT's prefix when it is not `/usr`
([hailo-10h.md](hailo-10h.md), Build). The device's `arch` is `hailo8`
or `hailo8l`, as HailoRT's identify reports it.

## A bundle

The bundle needs a `FORMAT_HEF` artifact whose `target` is the board's
architecture, compiled by Hailo's Dataflow Compiler 3.x (the generation
that pairs with HailoRT 4.x; 5.x compiles for the Hailo-10H only). No
recipe in `bundle/recipes/` carries one: the MiniLM recipe's HEF is for
the Hailo-10H, and the loader passes over an artifact whose `target` is
not the device's `arch`, so on a Hailo-8 that bundle has no artifact
(`TURBO_E_BUNDLE_NO_ARTIFACT`). The match is exact: a HEF for `hailo8`
does not load on a `hailo8l`, and the other way round.

To compile one, add a HEF artifact to a copy of the recipe, modelled on
`hef-hailo10h-s128`, with `target` `hailo8` (or `hailo8l`) and
`produced_by.container` naming a Dataflow Compiler 3.x image built from
`bundle/hailo/Dockerfile` with the 3.x wheel. The Dockerfile's base
image is Python 3.12: check the Python versions the wheel supports
before building, since the compiler installs its own dependencies with
it. `scripts/setup/bundle-tool.sh --hailo` reports the wheel it finds in
`bundle/hailo/` and its generation.

## Conformance

```
export TURBO_TEST_REQUIRE_HAILO=1

cargo test -p turbo --features hailo     # the listing test checks each board's arch against its PCI id
TURBO_TEST_BUNDLE=<bundle with a hailo8 HEF> TURBO_TEST_DEVICE=hailo \
    cargo test --release -p turbo --features hailo --test conformance -- --include-ignored --nocapture
```

## Tiers, environment and limits

The same as the Hailo-10H: MODEL and FASTEST compute in I8, EXACT is
refused, the backend reads no variable of its own, a frame holds one row
of `fixed_seq` tokens, and longer texts need
`turbo_embed_options.max_tokens` set to cut them.

## Serving

`cargo build --release -p turbo-kserve --features hailo`, then
[../grpc.md](../grpc.md).
