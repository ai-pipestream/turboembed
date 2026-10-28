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

### The exact legs

The runs above labelled `exact` were not: nothing in the relation and
schema-extract tests read `TURBO_TEST_PRECISION` then
(`device_support::options` took `EncoderOptions::default()`, which is
`model`), so for GLiNER2 relations, GLiREL and schema extraction every
run above is a `model` run, twice over. The GLiNER rows read it and are
as labelled. With the variable read and the precision printed,
`parity-exact.log` is the exact leg at `1aebb0da` plus that change:
twenty release runs, each the tiny-bundle tests and the real
checkpoints' (GLiREL's as well, docs/receipts/glirel-levelzero-920),
80 test binaries, all passing, every one printing `f32 (exact)` and the
same numbers each time. With `DEVICE_REPS=1` each tiny-bundle binary
ran its pass once, not twenty times.

| Head | Encoder | Largest difference at exact | Decisions the same as the CPU's, a run |
|---|---|---|---|
| GLiNER2 relations | `testdata/tiny-deberta-bundle` | 7.5e-9 | 12 best spans, 12 relations |
| GLiREL | `testdata/tiny-deberta-bundle` | 1.2e-7 | 3 best pairs |
| GLiNER2 schema extraction | `testdata/tiny-deberta-bundle` | 7.5e-9 | 8 decisions, 46 spans and records |
| GLiNER2 relations | `fastino/gliner2-base-v1` | 4.6e-6 | 18 best spans, 9 relations |
| GLiNER2 schema extraction | `fastino/gliner2-base-v1` | 7.5e-6 | 33 decisions, 24 values |

The figures are the `model` runs' to the digit. Host load was between
19.1 and 31.1; these are correctness runs.

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

Host loaded, B70 idle. `timings.log` is the run: twenty release runs at
`04f89fcb` of

```
GLINER2_PACKAGE=... GLINER2_BUNDLE=... GLINER2_CHECKPOINT=... \
  GLIREL_PACKAGE=... GLIREL_BUNDLE=... GLIREL_SPM=... DEVICE_REPS=50 \
  TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --locked --features turbo/levelzero -p turbo-relation -p turbo-schema-extract \
  --test device_real -- --ignored --nocapture --test-threads 1
```

after `snap-store` was stopped. Before each run the B70 CI runner
containers had no cargo, test or nextest process and no other client
held the card: in the 5 s before each run and over each run, no other
xe client had compute engine time, nor render engine time over 1
percent (gnome-shell at most, the log records each). The host was not
quiet: its one-minute load was between 17.9 and 53.6 over the twenty
runs (before and after each), so the CPU column ran beside other work
and is slower and wider than on an idle host. Each test first runs its
50 parity passes over the fixture (each held to the CPU's logits and
decisions), then times a further 50 passes over the same texts without
comparing them: the timed passes are not themselves checked. A pass is
timed in milliseconds, the CPU encoder and the device interleaved; the
relations fixture has three texts and the schema fixture four, so each
distribution mixes the texts' lengths. Only the encoder moves: the
heads run on the host in both columns, so the B70 column is the encoder
on the B70 plus the heads on the CPU. Host: AMD Ryzen 9 9950X, 32
threads.

Over the twenty runs: the median of each run's p50, the range of the
runs' p50s, the least min, the mean of the means and the largest p99,
and the median and range of each run's CPU p50 over its B70 p50.

| Pass | CPU encoder and heads: p50 (runs), min, mean, p99 | B70 encoder, CPU heads: p50 (runs), min, mean, p99 | CPU p50 over B70 p50 |
|---|---|---|---|
| GLiNER2 relations | 22.5 (16.7 to 37.1), 15.0, 24.9, 69.0 | 7.8 (7.2 to 10.1), 6.7, 8.6, 18.9 | 2.83x (2.30 to 3.86) |
| GLiREL | 47.6 (28.6 to 73.6), 25.2, 54.1, 150.5 | 12.2 (10.8 to 16.4), 10.5, 13.1, 26.7 | 3.89x (2.64 to 5.06) |
| Schema extraction | 39.2 (29.2 to 64.9), 15.8, 39.8, 110.1 | 13.4 (11.1 to 19.7), 6.8, 13.7, 34.5 | 2.83x (2.62 to 3.29) |

Every run's parity passes held: logits within 4.6e-6 (relations),
5.7e-6 (GLiREL) and 7.5e-6 (schema extraction) of the CPU encoder's.

The earlier timings, `timings-desktop-load.log` (three runs at
`fbba0d32`), were taken while `snap-store` held the B70's render engine
61 to 65 percent busy; under that load the relations pass ran at about
half the CPU's speed and schema extraction at about 1.3 times it. They
are kept for the record and are superseded by the runs above.

## Not done

- GLiREL on the real checkpoint: since done, in
  docs/receipts/glirel-levelzero-920 (its recipe is
  `bundle/recipes/glirel-large-v0-encoder.json`).
- Timings on a quiet host: the B70 was idle for the runs above, the
  host was not.
