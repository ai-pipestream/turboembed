# SOLIDIFY final-bench receipts

JSON written by a **live** `make bench-machine-*` on the named Machine.
Not mocks. Not estimated latency. Not a claimed-zero H↔D when the
plugin still host-wraps tensors.

| file | Machine | device | command |
|---|---|---|---|
| `machine-b-ov.json` | **B** | OpenVINO GPU (TurboEmbed GenAI + TurboRerank CompiledModel) | `make bench-machine-b-ov` |

Machine A (`machine-a-cuda.json`) and Machine C (`machine-c-metal.json`)
land here when those hosts measure. Do not invent their numbers from
this box.

Gates (all three machines): p50/p99, H↔D/USM bytes honesty,
`allocs/forward == 0`, Berlin + MiniLM embed goldens in band.
