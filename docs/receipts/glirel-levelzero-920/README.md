# GLiREL on Level Zero, the real checkpoint (#920)

Base: `6618aa7a` (`fable/920-levelzero-glirel`: `89ee8cf4` with origin/main
`29214c70` merged, which carries #1430's fixed
`glirel_on_the_real_checkpoint`), plus the recipe this receipt's commit
adds. Device: Intel Arc Pro B70 through the Level Zero backend, f32 at
both precisions, release builds, with `TURBO_TEST_REQUIRE_LEVELZERO=1`,
so a run that did not reach the device fails. `parity.log` is the run;
the B70 CI runner containers had no cargo, test or nextest process when
it started, and no other client used the B70's compute engine.

## Parity

Twenty runs at each of `TURBO_TEST_PRECISION` `model` and `exact` of

```
GLIREL_PACKAGE=... GLIREL_BUNDLE=... GLIREL_SPM=... DEVICE_REPS=1 \
  TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --features turbo/levelzero -p turbo-relation --test device_real \
  -- --ignored --nocapture glirel
```

40 test binaries, every one passing, and every repeat printing the same
numbers:

| Head | Test | Encoder | Largest difference from the CPU encoder | Decisions the same as the CPU's |
|---|---|---|---|---|
| GLiREL | `analysis/relation/tests/device_real.rs` `glirel_on_the_real_checkpoint` | `jackboyla/glirel-large-v0`, sealed here | 5.7e-6 | 52 pair labels, and every extracted relation |

The difference is `|a - b| / max(1, |a|)` per logit, `a` the CPU
encoder's, against a bound of 1e-4; decisions are held where the CPU's
margin is over twice the bound. The CPU side is the encoder of the
package `turbo-bundle glirel-convert` wrote for the same revision (its
`encoder/` DeBERTa package), to which the bundle is bound tensor for
tensor. The host's one-minute load was between 2.2 and 31.0 over the
run (the log records it before each test); these are correctness runs,
not timings, and the timing lines the test prints are not a receipt.

## The real checkpoint

`jackboyla/glirel-large-v0` at revision
`40a523e12a8432d6da364cf2a195a28755ff04d3`, CC BY-NC-SA 4.0, fetched by
`turbo-zoo fetch` into a throwaway zoo after Kristian accepted the
licence there, and used only for this run. Its weights are not in this
repository, its packages or its bundles. `pytorch_model.bin` and
`glirel_config.json`, and the encoder's base files
(`microsoft/deberta-v3-large` at `64a8c8ea`: `config.json`, `spm.model`),
each match the sha256 the zoo index records; `parity.log` lists them.

`bundle/recipes/glirel-large-v0-encoder.json` seals the encoder:
`bundle/reference/gliner_encoder.py` in the pinned reference container
with the added-tokens argument `glirel`, then `turbo-bundle reference`,
which verified it (docs/glirel-relations.md has the commands). Its
outputs are the ones the #860 thread made independently:
`tokenizer.json` `3588b038`, `model.safetensors` `f0b4dc17`,
`config.json` `8b57329b`, `1_Pooling/config.json` `d6611933`; the recipe
pins all eight.

## Not done

- Timings: `snap-store` holds the B70's render engine about 67 percent
  busy, and this session cannot stop it, so no clean timing is taken.
