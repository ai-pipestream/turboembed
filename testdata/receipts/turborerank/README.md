# TurboRerank live receipts

| file | Machine | engine | device |
|---|---|---|---|
| `cpu-minilm-l6.json` | any (this Phase 1 host) | first-party FP32 MiniLM CE | **CPU** |

Phase 1 has **no** Machine A/B/C GPU/Metal receipts. CUDA / OpenVINO /
Metal land here when those backends exist — never invent numbers, never
put lab hostnames in docs (Machine A / B / C only).

CPU proof: `make test-turborerank` vs
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (HF
`AutoModelForSequenceClassification` on the pinned checkpoint).
