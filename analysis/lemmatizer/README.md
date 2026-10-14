# turbo-lemmatizer

OpenNLP 3.x's lemmatizers, ported to Rust on the analysis API: the maxent
`LemmatizerME` (`LemmatizerModel` opens the model, `MaxentLemmatizer` runs the
beam search over each sentence's tagged tokens and applies the predicted lemma
classes, shortest edit scripts, to the lowercased tokens) and the lookup
`DictionaryLemmatizer`. Both read `opennlp:tokens` and `opennlp:pos` and write
`opennlp:lemmas`, a lemma per token; the maxent one also writes
`turbo:confidence<opennlp:lemmas>`. The crate documentation (`src/lib.rs`)
describes the modules and where OpenNLP's behaviour is surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork.

## Models

| File | Origin | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin` | Apache OpenNLP UD models 1.3 (`downloads.apache.org/opennlp/models/ud-models-1.3`, also Maven Central `org.apache.opennlp:opennlp-models-lemmatizer-en:1.3.0`) | `711ad051faba79865c5f9fc5b8c455d6ad8e6d3e8b92f42142af1ef0bc7b3a96` |
| `opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin` | the same release (`opennlp-models-lemmatizer-de:1.3.0`) | `8850986f02dfbe293046bfce93bc0c3f3fa010e89177250abca23d860033d4f6` |

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
fetching the models. Both are maxent models with `BeamSize=3`, the default
factory and so the default context generator, and lemma classes (shortest edit
scripts) as outcomes: 860 in the English model. Their features name Universal
Dependencies tags (`t0=NOUN`), so they run on the tags of the UD
part-of-speech models. Each file's SHA-512 is in the first `model` record of
every golden made with it (`analysis/goldens/data/lemmatize/<model>/`), and the
tests check it.

The dictionary the dictionary lemmatizer's tests run,
`lemmatizer-dictionary.txt`, is made for this project and checked in as one of
the golden generator's fixtures, in `analysis/goldens/data/fixtures` (that
directory's README describes it).

## Golden files

`tests/golden.rs` compares both lemmatizers with files a JVM running OpenNLP
wrote, made by the golden generator (`analysis/goldens-tool`,
`LemmaGolden.java` and `DictionaryLemmaGolden.java`). Each golden holds the
sentences, tokens and part-of-speech tags the JVM's own sentence detector,
tokenizer and tagger made, and the lemma each lemmatizer gave each token. The
test loads the JVM's tokens and tags as golden layers and runs the lemmatizer
on them, so what is compared is the lemmatization alone: lemmas exactly and,
for the maxent lemmatizer, probabilities bit for bit.

| Run | Model | Input sets | What it covers |
|---|---|---|---|
| `lemmatize/opennlp-en-ud-ewt-lemmas-1.3-2.5.4` | the published English model (see Models) | `en`, `multi`, `edge` | 1,304 tokens, 181 of them with a class other than `O` |
| `lemmatize/opennlp-de-ud-gsd-lemmas-1.3-2.5.4` | the published German model | `edge` | 465 tokens, 45 changed |
| `lemmatize-dictionary/lemmatizer-dictionary` | `analysis/goldens/data/fixtures/lemmatizer-dictionary.txt` | `en`, `multi`, `edge` | 1,304 tokens, 130 found in the dictionary, the rest `O`; entries with several lemmas, with a capital letter (never found), listed twice, with an empty lemma field and with fields after the lemma |

Beside each maxent golden is a decision point dump, every context the beam
search asked the model about with the probability of every lemma class; the
test runs the search with a recording context generator and checks the
contexts feature by feature, in the JVM's order, and the probabilities bit for
bit.

`en/emoji-split` has a token that is half of a surrogate pair. The JVM
lemmatizes that token into the same half, so the maxent golden's lemma column
holds a lone surrogate, which the golden reader refuses (a lemma is a string
here); the test counts and skips that golden. The dictionary golden of the same
input reads (the lemma is `O`), and the test skips it because the tokens cannot
be loaded onto the Rust text.
