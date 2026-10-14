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
