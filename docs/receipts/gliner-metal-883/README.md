# GLiNER on Metal (#883)

The real `urchade/gliner_small-v2.1` checkpoint (`4e091416cf7c3481db542c2a3d26156916f3a47f`,
Apache-2.0) on the Apple M2 through the Metal backend's DeBERTa-v2 encoder, against the same
converted package on the CPU. The package and the sealed encoder bundle are the pair the CUDA
and Level Zero receipts used (`gliner.json` SHA-256 `fe0d948c...`, encoder `manifest.json`
`5e7f64fd...`); `turbo-bundle verify` and `gliner-verify` pass on them on the M2.

- `parity.log`: `the_real`, twenty release runs at each of `TURBO_TEST_PRECISION=model` and
  `exact`, with `TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1`, each run's line naming
  the commit, the time and the load (1-minute 4.8 to 11.7). Forty of forty passed, with the
  same figures every run:
  the marker and first-subword rows within 3.4e-5 (lowest cosine 1.000000000), every logit
  within 6.1e-5, the packed ids and decoded entities the CPU's.
- `metal_deberta.log`: `TURBO_TEST_REQUIRE_METAL=1 cargo test --release -p turbo --features metal
  --test metal_deberta -- --nocapture`, twenty consecutive runs, each line naming the commit,
  the time and the load. Twenty of twenty passed, 8 tests each, with the same figures every run.
- `timing.log`: its first line names the commit and the command; `real_checkpoint_timing`, `GLINER_REPS=50`, at MODEL, with the load and GPU
  utilization before and the load after. docs/gliner.md gives its table.

Command:

```
GLINER_PACKAGE=<package> GLINER_ENCODER_BUNDLE=<bundle> TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 \
    TURBO_TEST_PRECISION=model cargo test --release -p turbo-gliner --features turbo/metal \
    --test real_device the_real -- --ignored --nocapture
```
