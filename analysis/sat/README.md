# turbo-sat

Sentence boundaries from a SaT model (Segment any Text, the `wtpsplit`
package, MIT) on the core's own encoder, written as `opennlp:sentences`.
[docs/neural-annotators.md](../../docs/neural-annotators.md) describes
the design, what it reproduces of wtpsplit and where it differs, the
licences of the checkpoints, and what is not built yet.

## Reference fixtures

No real SaT checkpoint is in the repository. The tests compare the step
with wtpsplit 2.2.2 itself on a small SaT model with random weights,
`testdata/tiny-sat-bundle`, made in the container of `reference/`:

- `reference/Dockerfile`: micromamba's image by digest,
  `reference/conda.lock` (every conda-forge package by URL and MD5:
  Python 3.12.15, PyTorch 2.14.1 for the CPU, transformers 5.17.0,
  tokenizers 0.23.3, safetensors 0.8.0, NumPy 2.5.4) and
  `reference/requirements.txt` (wtpsplit 2.2.2 and the five packages it
  imports that conda-forge does not carry, each by SHA-256, installed
  with `--no-deps`).
- `reference/sat_reference.py`: three commands, `tiny`, `reference` and
  `segment`, described at its top.

The image the fixtures were made with is
`sha256:d283c2464aecaae69c72d1757f7633280ac093dc85eedde67083cbf7a1764319`
(the recipe's `container`). Building it again from the same files and
making the fixtures again gave the same bytes.

To make them again, from the repository root, with `W` an empty
directory:

```
docker build -t turbo-sat-reference analysis/sat/reference
cp testdata/bge-m3/tokenizer.json $W/tokenizer.json
cp analysis/sat/tests/data/texts.jsonl testdata/tiny-sat-recipe/recipe.json $W/
run() { docker run --rm --network none --user 0:0 -v $W:/w turbo-sat-reference "$@"; }

# The model: two layers of width 32 over the cut-down XLM-R vocabulary.
run tiny /w/tokenizer.json /w/out

# The bundle: the reference vectors for the recipe's cases, then sealed.
mkdir -p $W/bundle/reference $W/upstream
run reference /w/out/model /w/out/tokenizer /w/recipe.json \
    /w/bundle/reference/reference.safetensors /w/bundle/reference/report.json \
    "turbo-sat-reference@$(docker image inspect --format '{{.Id}}' turbo-sat-reference)"
cp $W/tokenizer.json $W/out/model/model.safetensors $W/upstream/
cargo run -p turbo-bundle -- seal $W/recipe.json $W/upstream $W/bundle

# The fixtures: threshold, stride and block size.
run segment /w/out/model /w/tokenizer.json /w/texts.jsonl /w/t0.5-s16-b62.jsonl 0.5 16 62
run segment /w/out/model /w/tokenizer.json /w/texts.jsonl /w/t0.25-s62-b62.jsonl 0.25 62 62
run segment /w/out/model /w/tokenizer.json /w/texts.jsonl /w/t0.5-s7-b30.jsonl 0.5 7 30

# Limited lookahead: the same weights, wtpsplit's lookahead 8 (4 a layer).
run tiny /w/tokenizer.json /w/la 8
run segment /w/la/model /w/tokenizer.json /w/texts.jsonl /w/la8-t0.5-s16-b62.jsonl 0.5 16 62
run segment /w/la/model /w/tokenizer.json /w/texts.jsonl /w/la8-t0.025-s7-b30.jsonl 0.025 7 30
```

`tests/parity.rs` holds the `la8-` fixtures to a copy of the tiny bundle
with `architecture.lookahead` 4.

The image runs as a user other than root; `--user 0:0` lets it write
to the mounted directory.

`tests/data/texts.jsonl` holds 90 texts, one JSON string per line: the
54 of `testdata/tokenizer-texts.jsonl`, 22 more for sentence boundaries
(lines 55 to 76: whitespace alone, text with and without punctuation,
abbreviations, line feeds and carriage returns, Unicode line and
paragraph separators and information separators, scripts with and
without spaces, combining marks, emoji sequences, special-token strings,
long runs of one word, and whitespace at either end), and the 14
reference cases of the recipe not already among them. Each fixture's first line names the versions, the SHA-256
of the weights and the settings; then, per text, the ids and offsets
wtpsplit's tokenizer gives, the probability `SaT.predict_proba` gives
every character, and the pieces `SaT.split` returns.

## Tests

```
cargo test -p turbo-sat
TURBO_TEST_BUNDLE=testdata/tiny-sat-bundle cargo test -p turbo --test conformance
cargo test --release -p turbo-sat --test speed -- --ignored --nocapture
```

- `tests/parity.rs`: the scores and pieces against the fixtures, and
  the layer against wtpsplit's pieces with each cut moved to the end of
  its grapheme cluster.
  `TURBO_TEST_DEVICE` and `TURBO_TEST_PRECISION` run it on a device,
  for example `TURBO_TEST_DEVICE=cuda cargo test --release -p turbo-sat
  --features turbo/cuda --test parity`.
- `a_released_checkpoint_matches_wtpsplit` in `tests/parity.rs` holds a
  bundle sealed from a released checkpoint to fixtures the reference's
  `segment` made from its weights, when `TURBO_TEST_SAT_BUNDLE` names
  the bundle and `TURBO_TEST_SAT_FIXTURES` the fixtures, separated by
  commas. `bundle/recipes/sat-1l-sm.json` is the recipe for `sat-1l-sm`
  (refused under the current training-source policy, docs/licences.md).
  Its upstream directory holds the
  checkpoint's `model.safetensors` and, as `tokenizer.json`,
  FacebookAI/xlm-roberta-base's at revision
  `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, the base model its config
  names (the checkpoint has no tokenizer of its own). Its reference
  vectors and fixtures are made as above, with a directory holding the
  checkpoint's `config.json` and `model.safetensors` as the model in
  place of the tiny model's `out/model`, and the xlm-roberta-base file
  as the tokenizer; `config.json` is read by the reference only, not
  sealed. That file's normalizer is the precompiled map alone and its
  pre-tokenizer WhitespaceSplit then Metaspace, so the recipe sets
  `whitespace_split` and not `collapse_spaces`, and its `<mask>` takes
  no `lstrip`. The classifier's one label is the config's `LABEL_0`.
  The recipe's `produced_by` image is the `Dockerfile.classifier` one
  (below), not `reference/Dockerfile`'s. With `M` the directory holding
  `config.json` and `model.safetensors`, the fixtures are

  ```
  run segment /w/M /w/tokenizer.json /w/texts.jsonl /w/t0.25-s64-b510.jsonl 0.25 64 510
  run segment /w/M /w/tokenizer.json /w/texts.jsonl /w/t0.25-s16-b62.jsonl 0.25 16 62
  run segment /w/M /w/tokenizer.json /w/texts.jsonl /w/t0.5-s7-b30.jsonl 0.5 7 30
  ```

  with `/w/tokenizer.json` the xlm-roberta-base file; their SHA-256s are
  in docs/neural-annotators.md.
  `bundle/recipes/sat-3l.json` is the recipe for `sat-3l` (MIT, limited
  lookahead). Its checkpoint ships `pytorch_model.bin` and no
  safetensors: the upstream directory holds it, `config.json` and the
  bge-m3 `tokenizer.json`, and the weights are written by
  `bin_to_safetensors.py` in the `turbo-reference` image, whose report,
  with that image and `reproducible`, sits beside the file as
  `weights/model.safetensors.report.json` before `seal`. The reference
  reads the `.bin` directory; `segment` reads a directory holding
  `config.json` and the written `model.safetensors`.
  Its classifier preserves all 111 columns in the pinned config's
  `id2label` order, using the upstream neutral names `LABEL_0` through
  `LABEL_110`. These names assign no semantics to the other outputs.
  SaT selects row zero, wtpsplit's `Constants.NEWLINE_INDEX`, for the
  boundary probability. The reference writer emits all unpadded token
  logits, including special tokens, so sealing verifies the full head.
  The recipe pins the updated reference image. With the prior pinned
  environment already present, rebuild it without fetching dependencies:

  ```sh
  docker build --network=none --pull=false -t turbo-sat-reference:classifier \
      -f analysis/sat/reference/Dockerfile.classifier analysis/sat/reference
  ```

  This updates only the writer over the same locked dependency image;
  the full environment remains described by `reference/Dockerfile`.
- `tests/annotator.rs`: batching, texts longer than a window, texts
  without words, the threshold and line feeds, the options, the
  configuration hash, the provider and the analyzer.
- The conformance test checks the bundle's encoder against the mean of
  wtpsplit's encoder states for the recipe's cases.
- `tests/speed.rs` times the step; `TURBO_SAT_BUNDLE` names another
  bundle to time.
