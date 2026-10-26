# GLiNER2, GLiREL and GLiNER on Level Zero (#920)

Base: `10f214c8` (`fable/920-gliner2-glirel-cuda`, the device encoder
these heads read). Device: Intel Arc Pro B70 through the Level Zero
backend, f32 at both precisions, release builds, with
`TURBO_TEST_REQUIRE_LEVELZERO=1`, so a run that did not reach the
device fails. `parity.log` is the run; the B70 CI runner containers had
no cargo, test or nextest process when it started.

## Parity

| Head | Test | Encoder | Largest difference from the CPU encoder |
|---|---|---|---|
| GLiNER2 relations | `analysis/relation/tests/device.rs` | `testdata/tiny-deberta-bundle` | 7.5e-9 (bound 1e-4) |
| GLiREL | `analysis/relation/tests/device.rs` | `testdata/tiny-deberta-bundle` | 1.2e-7 (bound 1e-4) |
| GLiNER2 schema extraction | `analysis/schema-extract/tests/device.rs` | `testdata/tiny-deberta-bundle` | 7.5e-9 (bound 1e-4) |
| GLiNER | `analysis/gliner/tests/device.rs` | `testdata/tiny-deberta-bundle` | rows 7.2e-7, logits 5.8e-11 |
| GLiNER2 relations | `analysis/relation/tests/device_real.rs` | `fastino/gliner2-base-v1`, sealed here | 1.6e-6 of the pass's largest logit |
| GLiNER2 schema extraction | `analysis/schema-extract/tests/device_real.rs` | `fastino/gliner2-base-v1`, sealed here | 1.5e-6 of the pass's largest logit |

The tiny-bundle tests ran three times at each of `TURBO_TEST_PRECISION`
`model` and `exact`, and the real ones once at each; every repeat printed
the same numbers. The real checkpoint's counts and extracted relations
are the CPU encoder's.

The real encoder bundle was sealed with
`bundle/recipes/gliner2-base-v1-encoder.json` as `docs/gliner2-relations.md`
describes: `bundle/reference/gliner_encoder.py` in the pinned reference
container over the zoo's `fastino-gliner2-base-v1` files, then
`turbo-bundle reference`, which verified it. The CPU side is the package
`turbo-bundle deberta-convert` wrote for the same checkpoint.

The `device_real` runs here used `DEVICE_REPS=1` and are parity only;
the timing lines they print are not a timing receipt.

## Timings

`timings.log` is the run: three release runs at commit `4e0c2784` of

```
GLINER2_PACKAGE=... GLINER2_BUNDLE=... GLINER2_CHECKPOINT=... DEVICE_REPS=50 \
  TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --locked --features turbo/levelzero -p turbo-relation -p turbo-schema-extract \
  --test device_real -- --ignored --nocapture gliner2 schema
```

each started once no B70 CI runner container had a cargo, test or
nextest process and the host's one-minute load was under 8 (5.2, 6.2 and
4.9 at the starts). Each pass is timed in milliseconds, the CPU encoder
and the device interleaved, 50 times over each of the fixture's three
texts, so each distribution mixes three text lengths: its spread is the
texts as much as the device. Host: AMD Ryzen 9 9950X, 32 threads.

| Pass | CPU encoder p50 (min, mean, p99) | B70 p50 (min, mean, p99) | CPU p50 over B70 p50 |
|---|---|---|---|
| GLiNER2 relations, run 1 | 19.6 (16.1, 20.1, 25.2) | 37.8 (7.4, 35.7, 41.7) | 0.52x |
| GLiNER2 relations, run 2 | 21.2 (17.6, 21.5, 29.4) | 39.4 (26.9, 38.1, 45.7) | 0.54x |
| GLiNER2 relations, run 3 | 18.5 (15.7, 19.0, 23.2) | 32.0 (7.2, 32.4, 39.3) | 0.58x |
| Schema extraction, run 1 | 33.5 (17.1, 32.0, 43.7) | 29.1 (12.5, 28.6, 38.3) | 1.15x |
| Schema extraction, run 2 | 45.8 (23.2, 46.4, 85.6) | 26.7 (8.9, 27.2, 59.8) | 1.72x |
| Schema extraction, run 3 | 33.1 (17.5, 31.4, 42.5) | 25.9 (12.7, 25.4, 36.7) | 1.28x |

The B70 was not quiet. `snap-store` held its render engine 64 to 66
percent busy through every run (xe fdinfo, read before and after each),
on the device the encoder runs on. No other client open at both reads had
more than 0.2 percent. So these are timings under that load, not on an idle device: the
relations pass on the B70 at about half the CPU encoder's speed, schema
extraction a little faster than it, and an idle-device run is still to
be taken. Host load also rose to 15.9 during run 2, which shows in that
run's CPU numbers.

## Not done

- GLiREL on the real checkpoint: there is no core bundle recipe for its
  DeBERTa-v3-large encoder, so GLiREL stands at tiny-bundle parity.
- Timings on an idle B70: `snap-store` held the render engine busy
  through every run above.
