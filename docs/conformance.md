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
  for `levelzero` (docs/levelzero.md), `--features metal` for `metal`
  (docs/metal.md), `--features hailo` for `hailo` (docs/hailo.md),
  `--features npu` for `npu` (docs/npu.md), whose bundle must carry an
  OpenVINO IR artifact listing `npu` in its `backends`.
- `TURBO_TEST_PRECISION`: the session's precision, `model`, `fastest`
  or `exact`. Unset, `model`. The tolerance is the one for the compute
  dtype the session reports, so `fastest` on a backend that computes it
  in F16 is held to the F16 row below.

With `--ignored` it also runs `a_real_bundle_matches_its_reference`,
which fails unless `TURBO_TEST_BUNDLE` is set, so a run meant for a real
bundle cannot pass on the small one by accident. `--nocapture` prints,
for each way of writing, the rows compared, one minus the lowest cosine
and the largest absolute difference.

## What it checks

On the device, the bundle is loaded and one session is made at
`TURBO_TEST_PRECISION` (`TURBO_PRECISION_MODEL` unless it says
otherwise) with the model's `max_batch` and `max_seq`.
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

Every vector must be within the tolerance for the compute dtype
`turbo_session_get_info` reports: its cosine against the reference at
least the lowest cosine, and, where the table gives one, no value
further from the reference's than the largest absolute difference.

| Compute dtype | Lowest cosine | Largest absolute difference |
|---|---|---|
| F32 | 0.9999 | 1e-4 |
| F16, BF16 | 0.999 | not bounded |
| I8 | 0.93 | not bounded |

The F32, F16 and BF16 rows are `turbo::record::tolerance`, the rule a
benchmark record is held to as well (docs/benchmarks.md). The I8 row is
the test's own (`I8_MIN_COSINE` in `core/tests/common/conformance.rs`): a Hailo
session computes in int8, and a benchmark record of int8 has no
tolerance.

`a_fixed_shape_refuses_the_cases_it_cannot_hold` runs the same check on
the small bundle with its artifact fixed at 32 tokens, so the capacity
path is exercised on every build.

## Other outputs

The same run checks the outputs besides pooled vectors, on the same
device and at the same precision (`core/tests/conformance/tasks.rs`).
Token vectors and sparse weights are held, row by row, to the table
above for the compute dtype the session reports. Rerank scores and QA
logits are held to the score bounds below instead.

| Check | Through | Against |
|---|---|---|
| `rerank_scores_match_upstream` | `turbo_rerank_write_pairs`, one pair a write and `max_batch` pairs a write | the transformers scores in `testdata/tiny-rerank-reference` |
| `extractive_qa_logits_match_upstream` | `turbo::qa::ExtractiveQa` on the backend's first device, one window a run and `max_batch` windows a run | the transformers start and end logits in `testdata/tiny-qa-reference` |
| `token_vectors_match_the_reference_encoder_and_pool_to_upstream` | `turbo_embed_write_ranges` with one range a token, `TURBO_NORMALIZE_NONE`, one row a write and full batches | each token: the f64 encoder over the bundle's weights (`PlainBert` in `core/tests/common`); their mean, normalized: the sentence-transformers vector of `testdata/tiny-bert-bundle` |
| `sparse_weights_match_the_reference_encoder` | the token vectors above, through an MLM head and the max-log pool on the host, as `turbo-embed`'s sparse encoder reads them | the same head and pool over the f64 encoder's states |

The small bundle has no trained MLM head, so the sparse check builds one
on its hidden width with the decoder tied to the word embeddings; the
head's arithmetic is held to numpy in `core/tests/mlm.rs`. The head and
the pool run on the host, so the check covers the device's token states
only, not a sparse path that runs on the device.

### Scores

A cosine over a run of scores does not see the run rescaled, and a
rescale changes which QA answers win over the null score. So every
rerank score and every QA start or end logit is bounded on its own: its
difference from the reference's is at most the bound times the run's
scale, the largest reference value in magnitude
(`core/tests/conformance/scores.rs`).

| Compute dtype | Largest difference, as a fraction of the scale |
|---|---|
| F32 | 1e-4 |
| F16 | 1e-2 |
| BF16 | 4e-2 |
| I8 | 0.15 |

The F32 bound is the one `core/tests/rerank.rs` and
`core/tests/qa_inference.rs` hold the CPU to. The F16, BF16 and I8
bounds are set from each format's precision and have not yet been
measured on a device; a device run that fails one is to be looked at
before the bound is changed.

The decisions are checked too:

- Rerank: every two pairs come out in the reference's order, unless the
  reference's two scores are within twice the bound of each other,
  where the bound cannot tell them apart.
- Extractive QA: each case gives the reference's best span, or no
  answer where the reference gives none, with `turbo::qa::decode` and
  the default null threshold on both, unless the reference's best span
  is within four times the bound of its second span or of the null
  score (each is the sum of two logits).

`--nocapture` prints how many pairs and cases were decided and how many
were too close to say.

A check is skipped, with the reason printed, when the device does not
offer what it needs: `turbo_runtime_capability` reports the task
`TURBO_CAP_UNSUPPORTED` (rerank, extractive QA), or
`turbo_embed_write_ranges` is `TURBO_E_UNSUPPORTED` (token vectors,
sparse). A skip passes; a check the device offers and fails is a
failure.
