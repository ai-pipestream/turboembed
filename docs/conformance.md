# Conformance

A backend conforms for a bundle when its vectors match the bundle's
reference: the outputs of the upstream pipeline, in fp32 on a CPU, that
the bundle carries (docs/bundle.md, `reference`). The check is one test,
`core/tests/conformance.rs`. It uses only the C interface, so the same
test runs on every backend linked into the build.

## Running it

```
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=<device> \
    cargo test --release -p turbo --test conformance -- --nocapture
```

- `TURBO_TEST_BUNDLE`: the bundle directory, absolute or relative to
  the workspace root (the directory with the top `Cargo.toml`), wherever
  cargo is run from. Unset, the small sealed bundle in
  `testdata/tiny-bert-bundle`.
- `TURBO_TEST_DEVICE`: a runtime device index, or a backend name
  (`cpu`, `cuda`, ...) for the first device that backend lists. Unset,
  the CPU. A backend behind a feature needs it on the command line too:
  `--features cuda` for `cuda` (docs/cuda.md), `--features levelzero`
  for `levelzero` (docs/levelzero.md).

With `--ignored` it also runs `a_real_bundle_matches_its_reference`,
which fails unless `TURBO_TEST_BUNDLE` is set, so a run meant for a real
bundle cannot pass on the small one by accident. `--nocapture` prints,
for each way of writing, the rows compared, one minus the lowest cosine
and the largest absolute difference.

## What it checks

On the device, the bundle is loaded and one session is made at
`TURBO_PRECISION_MODEL` with the model's `max_batch` and `max_seq`.
Then:

1. Ids. Every reference case, encoded by the bundle's tokenizer with
   its prompt role, gives the reference's ids exactly.
2. Capacity. A case whose ids are longer than the session's `max_seq`
   (a fixed-shape artifact's, say) is refused with `TURBO_E_CAPACITY`,
   through `turbo_embed_write_text` and `turbo_embed_write_tokens`
   both, and is not compared. It is never cut differently on one
   device.
3. Vectors, for every other case, four ways: `turbo_embed_write_text`
   one case at a time with the case's prompt role;
   `turbo_embed_write_text` with every case in batches of `max_batch`,
   each text carrying its prefix; `turbo_embed_write_tokens` with the
   reference's ids one at a time; and the same in full batches, padded
   with the pad id and mask 0.

Every vector's cosine against the reference must reach the floor for
the compute dtype `turbo_session_get_info` reports:

| Compute dtype | Lowest cosine |
|---|---|
| F32 | 0.9999 |
| F16, BF16 | 0.999 |

`a_fixed_shape_refuses_the_cases_it_cannot_hold` runs the same check on
the small bundle with its artifact fixed at 32 tokens, so the capacity
path is exercised on every build.
