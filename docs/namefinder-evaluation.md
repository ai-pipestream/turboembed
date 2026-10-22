# Local NameFinderME evaluation (#177)

`analysis/train-recipes/examples/namefind_eval.rs` runs a caller-provided
OpenNLP name model on a UTF-8 CoNLL-2003 English file. It reports micro
precision, recall and F1 and can compare predictions with the Java producer
in `analysis/train-recipes/reference/NamefinderEval.java`.

This is evaluation tooling, not a trained model release or a CoNLL-2003
accuracy result. No Reuters text is downloaded or distributed. The remaining
#177 corpus evaluation needs a locally licensed corpus or an approved
alternative. The MASC recipes and their accuracy gates (#427) are unchanged;
the FineWeb model publication step (#429) is separate.

## Run the same bytes in both runtimes

Set `MODEL` and `CORPUS` to files the caller is entitled to use. `TYPES` is
`all`, `per`, `org`, `loc` or `misc`. This filters gold labels only, matching
the OpenNLP CoNLL reader; model predictions of other types still count as
false positives. Use a matching label inventory for meaningful accuracy.
No label remapping or merge of independent finders is performed.

Build the Java reference with the pinned OpenNLP 3.x runtime, API, formats
and ML jars on `OPENNLP_CP`. The golden generator image contains these jars.
`REFERENCE_REVISION` identifies the actual reference source commit, not the
Rust checkout.

```sh
mkdir -p target/namefind-eval
javac -cp "$OPENNLP_CP" -d target/namefind-eval \
  analysis/train-recipes/reference/NamefinderEval.java
java -cp "$OPENNLP_CP:target/namefind-eval" NamefinderEval \
  "$MODEL" "$CORPUS" "$TYPES" "$REFERENCE_REVISION" \
  > target/namefind-eval/jvm.json
cargo run --locked --release -p turbo-train-recipes --example namefind_eval -- \
  "$MODEL" "$CORPUS" "$TYPES" target/namefind-eval/jvm.json \
  > target/namefind-eval/turbo.json
```

Without the final JVM record argument, the Rust command only scores and
writes its record. Summaries go to stderr; JSON goes to stdout. Comparison
failure returns a nonzero exit code before writing a result record.

Records include SHA-256 of the complete model and corpus bytes, selected
gold types, token counts, document reset flags, gold spans, predicted spans
and probabilities. They contain no source text or workstation paths. The
record still discloses annotation positions and labels; treat results in
accordance with the input data agreement. Corpus digests are sensitive to
line endings and byte order marks, deliberately preventing comparison of
different inputs under one claimed run.

The comparator checks model and corpus identity, sample order, reset flags,
gold labels and token counts, then requires identical predicted spans and
types and absolute probability error at most `1e-9`. Equal aggregate F1
alone cannot pass. Non-finite scores, invalid probabilities, empty corpora,
empty or out-of-range spans, overlapping predictions and duplicate mentions
are rejected. Adaptive features carry across sentences and reset at document
boundaries as requested by `Conll03NameSampleStream`. OpenNLP's F1 sentinel
`-1` is preserved when precision and recall are both zero.

## Checked-in reference proof

The four sentences in
`analysis/train-recipes/tests/data/namefind-eval.conll` are handwritten test
text, not a sample of Reuters. They include IOB1 and BIO labels, document
resets, non-ASCII names and a sentence without entities. The existing
`analysis/goldens/data/fixtures/ner-bio-fixture.bin` is a synthetic fixture
model, not an accuracy baseline.

`analysis/train-recipes/tests/data/namefind-eval-jvm.json` was produced by
OpenNLP at the revision recorded in its `engine` field, using the existing
`turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1` image with network
access disabled. The fixture model uses the golden generator's stored
cluster serializers: its classpath additionally contains the generator's
classes, with `-DOPENNLP_EXT_ALLOWED_PACKAGES=golden`. This is specific to the
trusted checked-in fixture and is not necessary for ordinary models.

Both runtimes produced 3 correct entities, 4 predictions and 7 gold entities:
precision `0.75`, recall `0.42857142857142855`, F1 `0.5454545454545454`, with
maximum probability error `0`. This demonstrates harness parity on the
fixture; it does not estimate real corpus accuracy. A separate run with
`ner-fixture-rust.bin` also matched spans and probabilities exactly.

The Rust integration test compares the pinned JVM record on ordinary test
runs. Regenerating it requires the Java runtime above; do not replace its
reference values with Rust predictions.
