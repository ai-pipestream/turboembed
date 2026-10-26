# GLiNER2, GLiREL and GLiNER on Level Zero (#920)

Base: `28ecaa09` (`fable/920-levelzero`: `fable/920-gliner2-glirel-cuda`,
the device encoder these heads read, with #1407's head `49a708a1` merged
in). Device: Intel Arc Pro B70 through the Level Zero backend, f32 at
both precisions, release builds, with `TURBO_TEST_REQUIRE_LEVELZERO=1`,
so a run that did not reach the device fails. `parity.log` is the run;
the B70 CI runner containers had no cargo, test or nextest process when
it started.

## Parity

Twenty runs at each of `TURBO_TEST_PRECISION` `model` and `exact`, each
run both the tiny-bundle tests and the real checkpoint's: 200 test
binaries, every one passing, and every repeat printing the same
numbers. Each tiny-bundle test also runs its pass twenty times inside
the binary, and prints how many decisions it held.

| Head | Test | Encoder | Largest difference from the CPU encoder | Decisions the same as the CPU's |
|---|---|---|---|---|
| GLiNER2 relations | `analysis/relation/tests/device.rs` | `testdata/tiny-deberta-bundle` | 7.5e-9 (bound 1e-4) | 240 best spans, 240 relations extracted |
| GLiREL | `analysis/relation/tests/device.rs` | `testdata/tiny-deberta-bundle` | 1.2e-7 (bound 1e-4) | 60 best pairs |
| GLiNER2 schema extraction | `analysis/schema-extract/tests/device.rs` | `testdata/tiny-deberta-bundle` | 7.5e-9 (bound 1e-4) | 160 decisions, 920 spans and records decoded |
| GLiNER | `analysis/gliner/tests/device.rs` | `testdata/tiny-deberta-bundle` | rows 7.2e-7, logits 5.8e-11 | the decoded entities |
| GLiNER2 relations | `analysis/relation/tests/device_real.rs` | `fastino/gliner2-base-v1`, sealed here | 4.6e-6 | counts, decisions and extracted relations |
| GLiNER2 schema extraction | `analysis/schema-extract/tests/device_real.rs` | `fastino/gliner2-base-v1`, sealed here | 7.5e-6 | decisions |

The real checkpoint's differences are `|a - b| / max(1, |a|)` per logit,
`a` the CPU encoder's, against a bound of 1e-4 (#1407's
`device_support`); decisions are held where the CPU's margin is over
twice the bound. The host's one-minute load was between 2.1 and 27.5
over the run (the log records it before each test); these are
correctness runs, not timings.

## The real checkpoint

`fastino/gliner2-base-v1` at revision
`f9634218e53580c56edf0de97ca1a7d3f1c2354e`: the files sealed here,
`model.safetensors`, `config.json`, `added_tokens.json`, `spm.model` and
`encoder_config/config.json`, each match the sha256 the zoo index
(`analysis/zoo/index/zoo.json`) records for that revision, and
`bundle/recipes/gliner2-base-v1-encoder.json` pins the same commit. With
`Z` the zoo's `fastino-gliner2-base-v1` artifacts at that revision and
`W` an empty directory:

```
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD/bundle/reference:/scripts:ro -v $Z:/ckpt:ro -v $W:/w --entrypoint python \
    turbo-reference@sha256:3fe2d4ad441927475b97a400364ebcb7fca3f324dae1a0463720eff7fc0a53f4 \
    /scripts/gliner_encoder.py /ckpt/encoder_config/config.json /ckpt/spm.model /ckpt/model.safetensors /w/out \
    encoder. /ckpt/added_tokens.json
cargo run --release --locked -p turbo-bundle -- reference bundle/recipes/gliner2-base-v1-encoder.json $W/out $W/bundle
```

`turbo-bundle reference` verified the bundle. The CPU side is the
package `turbo-bundle deberta-convert` wrote for the same revision. The
`device_real` parity runs used `DEVICE_REPS=1`; the timing lines they
print are not a timing receipt.

## Timings

`timings.log` is the run: three release runs at `fbba0d32` (this
receipt's parent) of

```
GLINER2_PACKAGE=... GLINER2_BUNDLE=... GLINER2_CHECKPOINT=... DEVICE_REPS=50 \
  TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --locked --features turbo/levelzero -p turbo-relation -p turbo-schema-extract \
  --test device_real -- --ignored --nocapture gliner2 schema
```

each started once no B70 CI runner container had a cargo, test or
nextest process and the host's one-minute load was under 8 (1.8, 4.4
and 6.5 at the starts). Each pass is timed in milliseconds, the CPU
encoder and the device interleaved, 50 times over each of the fixture's
three texts, so each distribution mixes three text lengths: its spread
is the texts as much as the device. Every pass is also held to the CPU's
logits and decisions. Host: AMD Ryzen 9 9950X, 32 threads.

| Pass | CPU encoder p50 (min, mean, p99) | B70 p50 (min, mean, p99) | CPU p50 over B70 p50 |
|---|---|---|---|
| GLiNER2 relations, run 1 | 20.1 (16.5, 21.0, 32.1) | 36.9 (28.9, 37.1, 45.9) | 0.54x |
| GLiNER2 relations, run 2 | 18.3 (16.1, 18.4, 23.1) | 38.6 (6.8, 36.4, 40.7) | 0.47x |
| GLiNER2 relations, run 3 | 18.5 (15.6, 18.8, 22.4) | 33.6 (6.6, 33.6, 40.2) | 0.55x |
| Schema extraction, run 1 | 56.9 (20.9, 58.1, 103.4) | 22.9 (7.7, 27.1, 67.8) | 2.49x |
| Schema extraction, run 2 | 32.9 (17.3, 31.7, 42.9) | 25.9 (13.6, 26.1, 37.3) | 1.27x |
| Schema extraction, run 3 | 33.0 (17.8, 31.7, 43.6) | 26.0 (12.6, 25.8, 37.4) | 1.27x |

Neither the host nor the B70 stayed quiet. The host's load rose to 22.3
during run 1, which shows in that run's CPU schema numbers (p50 56.9
against 32.9 and 33.0 in the others); runs 2 and 3 ended at 15.6 and
8.9. `snap-store` held the B70's render engine 61 to 65 percent busy
through every run (xe fdinfo, read before and after each). So these are
timings under that load, not on an idle device: the relations pass on
the B70 at about half the CPU encoder's speed, and schema extraction
about 1.3 times its speed in the two runs the host left alone.

## Not done

- GLiREL on the real checkpoint: there is no core bundle recipe for its
  DeBERTa-v3-large encoder, so GLiREL stands at tiny-bundle parity.
- Timings on an idle B70: `snap-store` held the render engine busy
  through every run above.
