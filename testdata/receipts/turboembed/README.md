# TurboEmbed receipts

Machine-written proof that `turboembed_embed` hit a real engine.

| file | what it proves |
|---|---|
| `apple-minilm.json` | Rust → `turboembed.h` → FP MiniLM mean+L2 on Apple Metal. Cosine vs `testdata/e2e/goldens/nvidia/minilm.json` (≥ 0.97) and apple goldens (≥ 0.99). |

Regenerate on a Mac:

```
make test-turboembed-apple
```

A mock stub, BERT CLS pooler, or missing weights fails the test; do not
hand-edit a passing receipt.
