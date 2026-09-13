# TurboRerank goldens

Frozen scores for `cross-encoder/ms-marco-MiniLM-L6-v2` revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`.

| file | what |
|---|---|
| `ms_marco_minilm_l6_berlin.json` | Berlin population query vs three docs. Identity logits + sigmoid. |

Regenerate after a kernel change (must still match ST/TEI within 2e-3):

```bash
make fetch-rerankers
# after cargo test -p turborerank -- --ignored, copy printed logits
```

Live GPU/Metal receipts belong under
`testdata/receipts/turborerank/` — Machine A/B/C only, never hostnames.
gRPC `Rerank` (Phase 3) compares the same sigmoid vector through
`crates/backend-turborerank`.
