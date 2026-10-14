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
| `lemmatize/opennlp-en-ud-ewt-lemmas-1.3-2.5.4` | the published English model (`tests/models`) | `en`, `multi`, `edge` | 1,304 tokens, 181 of them with a class other than `O` |
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
