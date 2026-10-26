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

## Not done

- GLiREL on the real checkpoint: there is no core bundle recipe for its
  DeBERTa-v3-large encoder, so GLiREL stands at tiny-bundle parity.
- Timings: release timings on a quiet B70 follow as their own commit.
