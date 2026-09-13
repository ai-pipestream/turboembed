# Turbo bench receipts

JSON written by `make bench-turbo MACHINE=…` on the named Machine.
Measured p50/p99, intercepted host↔device bytes, allocs/forward, and
golden-band checks. Not estimates. Not mocks.

| file | Machine | device | command |
|---|---|---|---|
| `machine-a-cuda.json` | **A** (NVIDIA) | CUDA | `make bench-machine-a` |

Do not hand-edit latency or counter fields. Re-run the Make target on
that Machine. `host` is provenance — map it through the root README
chart.
