# History

This directory holds documentation and planning material that predates the
Turbo v2 rewrite described in `PLAN.md`.

## `poc/`

`docs/history/poc/` contains every page that lived directly under `docs/`
before the refactor. They describe the proof-of-concept system that was
removed from the working tree at the start of milestone P0 and tagged
`poc-2026-09-21` (see PLAN.md section 9). That system had per-arch gRPC
servers (`inferstream-nvidia`, `inferstream-intel`, the Swift
`inferstream-apple`), the `include/turboembed.h` and
`include/turboembed_prepared.h` headers, and Cargo-feature-selected engine
backends. None of that is the current architecture.

These pages are kept, unedited, because they contain:

- Hardware bring-up runbooks (Intel NPU, Apple Metal, Jetson/CUDA machine
  setup) whose procedural content may still be useful when the corresponding
  Turbo v2 provider (P2 through P6 in `PLAN.md`) is brought up on the same
  machines.
- Design notes and qualification write-ups (`prepared-abi-design.md`,
  `native-tokenizer-contract.md`, `turboembed-architecture.md`, and similar)
  that recorded engineering decisions and measurements from the PoC period.
- Machine-specific validation and benchmark records
  (`*-machine-a.md`, `*-machine-b.md`, `*-machine-c.md`, dated checklists).

None of the pages in `poc/` describe current Turbo v2 behavior. They do not
use the `turbo_` C ABI, the bundle v2 format, or the provider contract in
`PLAN.md` section 4. Do not treat status words in these files (`LIVE`,
`DONE`, capability tables) as claims about the current tree. For current
documentation, see `docs/architecture.md`, `docs/c-api.md`, `docs/bundles.md`,
`docs/providers.md`, `docs/testing.md`, and `PLAN.md`.

## Receipts

The hardware receipts referenced by several of these pages were not moved.
They remain at `testdata/receipts/` (subdirectories `turboembed/`,
`turborerank/`, `turbo_buffer/`, `bench/`), unchanged, as the PoC's dated
qualification records. New Turbo v2 receipts (PLAN.md section 10) are
committed under the same `testdata/receipts/` root as they are produced by
each milestone.
