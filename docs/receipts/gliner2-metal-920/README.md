# GLiNER2, GLiREL and schema extraction on Metal (#920)

Base: `99a8c64a` (origin/main `f640ab64`, which carries the Metal
DeBERTa-v2 encoder of #883, plus this branch's change: the relation
tests' `device_support` reads `TURBO_TEST_REQUIRE_METAL`). Device: Apple
M2 through the metal backend, macOS 27.0.1, release builds, with
`TURBO_TEST_REQUIRE_METAL=1`, so a run that did not reach the device
fails. `parity.log` is the run, with the commit, command, time and host
load before each test.

## Parity

Twenty runs at each of `TURBO_TEST_PRECISION` `model` and `exact` of

```
GLINER2_PACKAGE=... GLINER2_BUNDLE=... GLINER2_CHECKPOINT=... \
  GLIREL_PACKAGE=... GLIREL_BUNDLE=... GLIREL_SPM=... DEVICE_REPS=1 \
  TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 TURBO_TEST_PRECISION=<p> \
  cargo test --release --features turbo/metal -p turbo-relation -p turbo-schema-extract \
  --test device --test device_real -- --include-ignored --nocapture
```

each the tiny-bundle tests and the real checkpoints': 160 test binaries,
every one passing, every run printing the precision it asked for
(`f32 (model)` or `f32 (exact)`) and the same numbers at both, run after
run.

| Head | Test | Encoder | Largest difference from the CPU encoder | Decisions the same as the CPU's, a run |
|---|---|---|---|---|
| GLiNER2 relations | `analysis/relation/tests/device.rs` | `testdata/tiny-deberta-bundle` | 7.5e-9 | 12 best spans, 12 relations |
| GLiREL | `analysis/relation/tests/device.rs` | `testdata/tiny-deberta-bundle` | 1.8e-7 | 3 best pairs |
| GLiNER2 schema extraction | `analysis/schema-extract/tests/device.rs` | `testdata/tiny-deberta-bundle` | 1.4e-8 | 8 decisions, 46 spans and records |
| GLiNER2 relations | `analysis/relation/tests/device_real.rs` | `fastino/gliner2-base-v1`, sealed here | 8.6e-6 | 18 best spans, 9 relations |
| GLiNER2 schema extraction | `analysis/schema-extract/tests/device_real.rs` | `fastino/gliner2-base-v1`, sealed here | 1.3e-5 | 33 decisions, 24 values |
| GLiREL | `analysis/relation/tests/device_real.rs` | `jackboyla/glirel-large-v0`, sealed here | 3.8e-6 | 52 pair labels |

The bound is 1e-4 throughout. The real checkpoints' differences are
`|a - b| / max(1, |a|)` per logit, `a` the CPU encoder's (#1407's
`device_support`); decisions are held where the CPU's margin is over
twice the bound. The M2 is a desktop in use: the host's one-minute load
was between 3.3 and 19.7 over the run, and the GPU was shared with
desktop applications. These are correctness runs, not timings; the
timing lines the tests print, with `DEVICE_REPS=1`, are not a receipt.

## The real checkpoints

`fastino/gliner2-base-v1` at revision
`f9634218e53580c56edf0de97ca1a7d3f1c2354e`, sealed as
docs/gliner2-relations.md gives it from the zoo's files (the checkpoint's
`model.safetensors` is the zoo index's `845fc4bd`), with the package
`turbo-bundle deberta-convert` wrote for the same revision as the CPU
side. `jackboyla/glirel-large-v0` at revision
`40a523e12a8432d6da364cf2a195a28755ff04d3`, CC BY-NC-SA 4.0, fetched
after the licence was accepted on this machine and sealed as
docs/glirel-relations.md gives it; the bundle and package stay local and
nothing from them is in this commit beyond the figures above.

## Not done

- `fastest`: not run here, as on Level Zero.
- Timings on an idle M2.
