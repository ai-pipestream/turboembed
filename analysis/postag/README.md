# turbo-postag

OpenNLP 3.x's maxent part-of-speech tagger, `POSTaggerME`, ported to Rust on
the analysis API: `PosModel` opens a `POSModel` file, `PosTagger` tags the
tokens of each sentence by beam search and is the annotator that reads
`opennlp:tokens` and writes `opennlp:pos` (a tag and its probability per
token, aligned to the tokens, with the tag set in the layer). The beam search,
the feature generators, the tag dictionary and the Penn Treebank and Universal
Dependencies tag mapping are `turbo-sequence`'s. The crate documentation
(`src/lib.rs`) describes the modules and where OpenNLP's behaviour is
surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork.

## Models

| File | Origin | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-pos-1.3-2.5.4.bin` | Apache OpenNLP UD models 1.3 (`downloads.apache.org/opennlp/models/ud-models-1.3`, also Maven Central `org.apache.opennlp:opennlp-models-pos-en:1.3.0`) | `510b89087e7ef7ffb8c50c63e3e999122404ac9fa64790dfa93b38d1938c52ce` |
| `opennlp-de-ud-gsd-pos-1.3-2.5.4.bin` | the same release (`opennlp-models-pos-de:1.3.0`) | `073fefbfaff2ed403bf474fccf793752d01c22e69f487cd68b43832d3123742f` |

The two UD models are the OpenNLP 1.3 models trained by OpenNLP 2.5.4 on
Universal Dependencies treebanks (English EWT, German GSD). The models are
under the Apache License 2.0, but the treebanks they were trained on are
CC BY-SA, so the files are not checked in: `cargo run -p turbo-goldens-tool --
fetch-models <dir>` downloads them (with the other models of
`analysis/goldens-tool/src/recipe.rs`, each checked against its published
SHA-512), and the tests read them from the directory `TURBO_GOLDENS_MODELS`
names, or `target/turbo-goldens/models` under the repository root. A test that
needs a model that is not there says so and passes, unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1` makes that a failure, as CI sets it after
fetching the models. Both are maxent models with `BeamSize=3`, a feature
generator descriptor (`generator.featuregen`), Universal Dependencies outcomes
and no tag dictionary. Each file's SHA-512 is in the first `model` record of
every golden made with it (`analysis/goldens/data/postag/<model>/`), and the
tests check it.

The fixture model with a tag dictionary that the tests also run,
`pos-fixture.bin`, is trained for this project on permissive data and checked
in as one of the golden generator's fixtures, in `analysis/goldens/data/fixtures`
(that directory's README says where it comes from).

## Golden files

`tests/golden.rs` compares the tagger with files a JVM running OpenNLP wrote,
`analysis/goldens/data/postag/<run>/<set>/<id>.golden`, made by the golden
generator (`analysis/goldens-tool`, `PosGolden.java`). Each golden holds the
sentences and tokens the JVM's own sentence detector and tokenizer cut and the
tag and probability `POSTaggerME.tag` and `probs` gave each token. The test
loads the JVM's tokens as a golden layer and runs the tagger on them, so what
is compared is the tagging alone: tags exactly and probabilities bit for bit.

| Run | Model | Input sets | What it covers |
|---|---|---|---|
| `opennlp-en-ud-ewt-pos-1.3-2.5.4` | the published English model (see Models) | `en`, `multi`, `edge` | the model as it is, UD tags |
| `opennlp-en-ud-ewt-pos-1.3-2.5.4+penn` | the same | `en` | asked for Penn Treebank tags: UD outcomes converted, the joined ones to `?` |
| `opennlp-en-ud-ewt-pos-1.3-2.5.4+beam1`, `+beam10` | the same, rebuilt with another `BeamSize` | `en` | a beam of one (one evaluation per token) and of ten |
| `opennlp-de-ud-gsd-pos-1.3-2.5.4` | the published German model | `edge` | another language, another outcome set |
| `pos-fixture` | `analysis/goldens/data/fixtures/pos-fixture.bin` | `en`, `edge` | Penn Treebank outcomes converted to UD, and a case-insensitive tag dictionary as the sequence validator (42 of its tokens are restricted by it) |

Beside each golden is a decision point dump, every context the beam search
asked the model about with the probability of every outcome; the test runs
the search with a recording context generator and checks the contexts feature
by feature, in the JVM's order, and the probabilities bit for bit (3,584
points for the English model over its three sets, 496 with a beam of one, 4,474
with a beam of ten, 1,271 for the German model, 2,648 for the fixture).

`en/emoji-split` has a token that is half of a surrogate pair, which the JVM
tagged as such; it cannot be loaded onto the Rust text, so that golden is
counted and skipped in each run over the `en` set.

## Accuracy on the UD English EWT test split

`tests/accuracy.rs` scores the English model on
`en_ewt-ud-test.conllu` of UD English EWT and compares every tag with the
JVM's. The treebank is CC BY-SA 4.0 and is not in the repository: the test is
ignored unless `TURBO_UD_EWT_TEST` names a copy downloaded for the run. The
file is read as OpenNLP's `ConlluStream` and `ConlluPOSSampleStream` read it
for `POSTaggerEvaluator`: a multiword token keeps its form and takes its
parts' UPOS tags joined with `+`, and an empty node stays a token tagged `_`;
neither can be tagged right, so the accuracy is given over every token and
over the plain word lines alone.

```
curl -sSfL -o /tmp/ud/en_ewt-ud-test.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-test.conllu
# The JVM, in the generator's container (analysis/goldens-tool/README.md):
docker run --rm --network none -v <models dir>:/models:ro -v /tmp/ud:/ud \
  --entrypoint java <image> -cp /generator/target/turbo-goldens-java.jar \
  ai.pipestream.turbo.goldens.PosAccuracy /models/opennlp-en-ud-ewt-pos-1.3-2.5.4.bin \
  /ud/en_ewt-ud-test.conllu /ud/jvm-tags.txt
# Rust, checking each tag against the JVM's (the model from the directory
# fetch-models filled):
TURBO_GOLDENS_MODELS=<models dir> \
TURBO_UD_EWT_TEST=/tmp/ud/en_ewt-ud-test.conllu TURBO_UD_EWT_JVM_TAGS=/tmp/ud/jvm-tags.txt \
  cargo test -p turbo-postag --test accuracy -- --include-ignored --nocapture
```

With UD English EWT r2.15 (`en_ewt-ud-test.conllu`, SHA-256
`0612e45914359e4b36d187ef2bfeaf66fc8933210925be985c5fbf6a9e311036`), 2,077
sentences, both gave:

| | Tokens | Correct | Accuracy |
|---|---|---|---|
| every token, as `POSTaggerEvaluator` counts | 24,742 | 22,987 | 92.9068% |
| plain word lines (no multiword tokens, no empty nodes) | 24,386 | 22,676 | 92.9878% |

The Rust tags equal the JVM's on all 24,742 tokens.

## BiLSTM-CRF on the CPU backend

`neural::BilstmPosTagger` runs an `ONLP-BLPT` model (char BiLSTM, word
BiLSTM of one or two layers, optional pretrained vectors and adapter,
optional CRF) and is an annotator from `opennlp:tokens` to `opennlp:pos`. It
has two numerics (`neural::Numerics`):

- `Packed`, the default (`BilstmPosTagger::new`). The weights are packed once
  into `turbo::cpu::Matrix`, the CPU backend's f32 GEMM, and every LSTM runs
  on it: the input projections of all the tokens of a call in one product,
  the recurrent product over the sentences still running at each step
  (longest first), the char LSTM once per distinct word with a
  per-character projection table, and the output layer in one product.
  Each row of a product is computed alone, so a sentence's scores do not
  depend on what it is batched with. Word representations are cached (up to
  100,000 words). The annotator batches every sentence of a document in one
  call. The CRF is decoded in f64 by max-plus Viterbi, and the marginals
  (the probabilities) by forward-backward in log space. Where plain f64
  cannot hold a sentence's scores (a bound on its paths' scores above 1e6,
  as with CRF weights near 1e300), that sentence is decoded by the exact
  scorer instead; a probability that is not finite is an error. A weight
  past f32's range does not fail either: if it cannot be packed, `new`
  makes a `JvmParity` tagger (`numerics()` says so, and
  `with_numerics(.., Numerics::Packed)` returns the error), and if it
  overflows a call's f32 scores, that call's emissions come from the f64
  reference.
- `JvmParity` (`BilstmPosTagger::jvm_parity`), for the golden tests and the
  trainer's evaluation only: the f64 forward pass in the JVM's order and the
  JVM's exact-sum CRF scorer, which give the JVM's emissions to within
  rounding.

The two numerics have different configuration hashes (`config_hash`, FNV-1a
over the numerics and the model's SHA-256), so their outputs are never
cached as each other's. `tests/bilstm_numerics.rs` checks both rules
through the public API, and `tests/bilstm_provenance.rs` checks that each
numerics' tags carry its own hash through an analyzer run, the document
graph export, its bytes and the import, and as the key of a pipeline cache.

`tests/neural_goldens/` runs both on every golden of
`analysis/goldens/data/postag-neural/` (six BiLSTM runs). The parity
numerics match the JVM as before (42,924 of 43,560 emissions bit for bit,
the rest within 8.9e-16). The packed numerics
give the JVM's tag on all 1,980 tokens, with emissions within 7.6e-7 and
probabilities within 1.7e-7 of the JVM's (the test's tolerance is 1e-5; tags
must be equal).

### Accuracy and throughput on UD English EWT

No published BiLSTM-CRF model exists, so `tests/bilstm_ewt.rs` trains one on
the EWT train split (trainer defaults, a CRF, five epochs, the dev split for
model selection), then tags the test split with both numerics. The treebank
is CC BY-SA 4.0 and neither it nor the model is in the repository; the test
is ignored unless the splits are named. Sentences are read as syntactic
words (a multiword token's range line skipped and its parts kept, empty
nodes dropped) with their UPOS tags.

```
for s in train dev test; do curl -sSfL -o /tmp/ud/en_ewt-ud-$s.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-$s.conllu; done
TURBO_UD_EWT_TRAIN=/tmp/ud/en_ewt-ud-train.conllu TURBO_UD_EWT_DEV=/tmp/ud/en_ewt-ud-dev.conllu \
TURBO_UD_EWT_TEST=/tmp/ud/en_ewt-ud-test.conllu TURBO_BILSTM_MODEL=/tmp/ud/bilstm-ewt.bin \
TURBO_BILSTM_REPS=3 TURBO_BILSTM_BATCH=32 \
  taskset -c 6 cargo test --release -p turbo-postag --test bilstm_ewt -- --ignored --nocapture
```

With UD English EWT r2.15 (test split SHA-256 as above), training took
1,224 s on one core and wrote an `ONLP-BLPT-2` model of 10,204,385 bytes,
17 tags, SHA-256
`819328e7b1863ac5247e013547cadbfbf74d81ce09f1a04a1cd2030762adfad2`. On the
test split's 2,077 sentences and 25,094 words:

| Numerics | Correct | Accuracy |
|---|---|---|
| `Packed` (default) | 23,823 | 94.9350% |
| `JvmParity` | 23,823 | 94.9350% |

The two give the same tag on all 25,094 words.

Throughput, one core of an AMD Ryzen 9 9950X3D (pinned with `taskset`),
words per second over the test split, the median of six passes of each
(three rounds of packed, parity, parity, packed). A cold pass loads a fresh
model and tagger, so the representation cache starts empty; a warm pass
reuses one tagger. "Emissions alone" is the BiLSTM without the CRF, one
sentence per call.

| Sentences per call | | Packed | JvmParity | Ratio |
|---|---|---|---|---|
| 1 | cold | 66,623 | 1,405 | 47.4x |
| 1 | warm | 84,526 | 1,466 | 57.7x |
| 32 | cold | 77,675 | 1,423 | 54.6x |
| 32 | warm | 101,107 | 1,465 | 69.0x |
| 128 | cold | 78,893 | 1,431 | 55.1x |
| 128 | warm | 100,456 | 1,461 | 68.8x |
| 1, emissions alone | warm | 107,084 | 7,589 | 14.1x |

Most of the parity numerics' time is the exact-sum CRF scorer; the packed
numerics' gain on the forward pass alone is the last row. The check that
sends a sentence with huge scores to the exact scorer costs nothing
measurable: against the code without it, three runs each on the same core,
32 sentences per call, the medians were 97,568 against 97,655 words per
second warm and 77,083 against 77,078 cold. The parity
numerics are as fast as before this path was added (an A/B over 400 test
sentences on the same core: within 1%).

### The turbo-bench record

`turbo-bench postag` measures the two numerics on a treebank and writes a
record to `benchmarks/records/postag/`: accuracy, agreement, emission and
marginal drift, and words per second and call latency by sentences per
call and by sentence length. The record on this model and the EWT test
split is `benchmarks/records/postag/x86-64.cpu.postag.bilstm-crf.819328e7.90e41ce15c8c.json`; docs/analysis.md ("BiLSTM-CRF
numerics") reads it.

### Not done

- Other backends. The packed numerics use the CPU backend's GEMM directly;
  no GPU or NPU kernel runs the tagger, so no `TURBO_TEST_REQUIRE_*`
  hardware run exists for it.
- A bundle import recipe built in a pinned container: there is no published
  BiLSTM-CRF model to import.
- The JVM on EWT. The EWT comparison is packed against the parity numerics,
  which the goldens tie to the JVM; the JVM was not run on the treebank.
