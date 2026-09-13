# SOLIDIFY final-bench receipts

JSON written by a **live** `make bench-machine-*` on the named Machine.
Not mocks. Not estimated latency. Not a claimed-zero H↔D when the
plugin still host-wraps tensors.

| file | Machine | device | command |
|---|---|---|---|
| `machine-a-cuda.json` | **A** (NVIDIA) | CUDA (TurboEmbed ORT + TurboRerank CE) | `make bench-machine-a` / `make bench-turbo MACHINE=A` |
| `machine-b-ov.json` | **B** | OpenVINO GPU (TurboEmbed GenAI + TurboRerank CompiledModel) | `make bench-machine-b-ov` / `make bench-turbo MACHINE=B` |

Machine C (`machine-c-metal.json`) lands here when that host runs.
Do not invent numbers from another box.

Gates (all three machines): p50/p99, H↔D/USM bytes honesty,
`allocs/forward == 0`, Berlin + MiniLM embed reference vectors in range.

Do not hand-edit latency or counter fields. Re-run the Make target on
that Machine. `host` is provenance — map it through the root README
chart.
