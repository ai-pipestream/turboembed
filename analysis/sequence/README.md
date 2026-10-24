# turbo-sequence

The sequence machinery of OpenNLP 3.x's part-of-speech tagger, chunker and
name finder, ported to Rust: `BeamSearch`, the BIO and BILOU codecs, and the
feature generation framework with the XML descriptors models store. The crate
documentation (`src/lib.rs`) describes the modules and lists where OpenNLP's
behaviour is surprising.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork, and for the code point `StringPattern` the
`ready/string-pattern-unicode` branch at `b2473ca80d65ae2ef10a24b08b1d0ae6a9231109`.

## Golden files

The tests compare the crate with goldens a JVM running OpenNLP wrote: the
runs of `analysis/goldens-tool` listed below, in its container, under
`analysis/goldens/data/<annotator>/<run>/<set>/<input>.golden`. The JVM makes
the files only; nothing calls it at test time. `turbo-goldens generate` makes
them and `turbo-goldens verify` checks that they are what the recipe makes
(see the goldens-tool README).

| Input | Contents |
|---|---|
| `inputs/sequence/sentences.txt` | 16 tokenized sentences, one per line, tokens separated by single spaces: abbreviations, numbers, dates, URLs, accented, Greek, Cyrillic, CJK and kana text, letters and an emoji outside the Basic Multilingual Plane, a titlecase letter, a one-token sentence |
| `inputs/sequence/tokens.txt` | Tokens for `StringPattern`, escaped (`\uXXXX` for one UTF-16 unit, `\\`, `\t`, `\n`, `\r`), lone surrogates included; the empty last line is the empty token |
| `inputs/descriptor-errors/<case>.txt` | 16 malformed feature generator descriptors |
| `inputs/codec/bio.txt`, `bilou.txt` | The label alphabets of the codec runs, one label per line: 10 for BIO, 12 for BILOU |
| `inputs/tag-format/guesses.txt` | Ten lists of outcomes, separated by spaces, ties and the empty list among them |

| Fixture | Contents |
|---|---|
| `fixtures/featuregen/pos-default.xml`, `ner-default.xml` | OpenNLP's default part-of-speech and name finder descriptors (the same files as `src/descriptors/`) |
| `fixtures/featuregen/everything.xml` | Every factory except the window, aggregated and cached ones (those are in `nesting.xml`), with the three resources |
| `fixtures/featuregen/nesting.xml` | `cache="TRUE"` on the root, windows and aggregates inside each other, a cache inside the tree, an aggregate of one, a negative window length |
| `fixtures/featuregen/brown.cluster`, `word.cluster`, `names.dictionary` | A Brown cluster file, a word cluster file and a case-insensitive names dictionary with multi-token entries |
| `fixtures/tags.tagdict`, `tags-cases.tagdict`, `tags-exact.tagdict` | `tags.tagdict` is the dictionary of `pos-fixture.bin`; the other two hold the same 40 entries case-insensitive and case sensitive, with words that lowercase to the same word, a word listed twice, and tags separated by two spaces or ending in one |
| `fixtures/*-fixture.bin` | Small models OpenNLP trained on `train-fixtures/train-*.txt` (below) |

| Run | Read by | Contents |
|---|---|---|
| `string-pattern/utf16-units`, `string-pattern/code-points` | `tests/pattern.rs` | `StringPattern` flags and digit count and `tokenFeature` for about 2,900 tokens: on `tokens`, the inputs, every code point of 18 ranges, lone surrogates, and a capital and a period; on `sentences`, the sentences' tokens. `code-points` runs the branch's classes |
| `features/<descriptor>` | `tests/features.rs` | The features each descriptor's generator gives every token of the sentences, run as one document with `updateAdaptiveData` after each sentence |
| `features/default-pos-context`, `default-pos-context+names-dictionary` | `tests/features.rs` | `DefaultPOSContextGenerator`, without and with the names dictionary as its n-gram dictionary |
| `features/errors` | `tests/features.rs` | The exception `GeneratorFactory.create` throws for each malformed descriptor, as the golden's error record |
| `beam/<model>` | `tests/beam.rs` | Every sequence of the final beam for each sentence (outcomes, probabilities and score as `f64` columns, compared bit for bit), every context the search asked for in sentences 0 and 8, and the name finder's names |
| `tag-dictionary/<name>` | `tests/postag.rs` | A tag dictionary as `POSDictionary` reads it, from a fixture file or a model: whether it is case sensitive, its number of words, the SHA-256 of its words and tags sorted by word, every word and its tags when the dictionary is small, and `getTags` for every token of the sentences and every word of the file, each as written, lowercased and uppercased |
| `tag-format/<model>` | `tests/postag.rs` | `POSTagFormatMapper` for a model's outcomes: the guessed format, `convertTag` of every outcome, every key of both tables and a few other strings, and the tags `POSTaggerME` returns for each sentence when asked for each of the four formats, next to the outcomes it returns unchanged with `CUSTOM` |
| `tag-format/guesses` | `tests/postag.rs` | The format guessed for each list of outcomes |
| `codec/bio`, `codec/bilou` | `tests/codec.rs` | Decoding every outcome sequence of up to three labels, the validator on every prior sequence of up to two labels and every label, encoding 7 span lists, and `areOutcomesCompatible` on every subset of 6 or 7 outcomes |

The beam search runs wire each search as `POSTaggerME`, `ChunkerME` and
`NameFinderME` do (the part-of-speech context cache as large as the beam, the
name finder's additional context window, `updateAdaptiveData` after each
sentence) but call `bestSequences` with the beam size as the count, so the
whole final beam is recorded. Each sentence's best sequence is checked against
what the tagger itself returns. The chunker runs on the tags of the
part-of-speech model its golden names second.

A string that can hold half of a surrogate pair (a token of the string pattern
goldens, a feature) is written as `TrainGolden.lossless` writes it: a backslash
doubled, a lone surrogate as `\uXXXX`.

The code point `StringPattern` comes from the fork's
`ready/string-pattern-unicode` branch at commit b2473ca: the goldens image
compiles its `StringPattern`, `FeatureGeneratorUtil` and `ParamChecks` (which
cdf33c6 does not have) into a jar of their own, which the generator loads
ahead of its class path for that run.

### Fixture models

| Model | Trainer | Descriptor | Codec | SHA-256 |
|---|---|---|---|---|
| `pos-fixture.bin` | GIS, with `tags.tagdict` | `pos-default.xml` | | `db87fc45de70bfe0175bfc402eb5c349a3e7c85abd22a6507c20b69fe50edef8` |
| `chunker-fixture.bin` | perceptron | | | `cc68fdbac1ee51695e232e59ac02754dde51e5fb1dd50d15c565c805204f23b0` |
| `ner-bio-fixture.bin` | perceptron | `everything.xml`, its resources stored in the model | BIO | `da05d773f20be0b8325e6dee3e0a1a103449f694c27275c58cb6f6f29e281081` |
| `ner-bilou-fixture.bin` | GIS | `nesting.xml` | BILOU | `8ebc2bc170c11cfa35eadbe89f35871ff6938ae29c08def57c3d47e95be27ae5` |
| `ner-qn-fixture.bin` | QN | the default one | BIO | `f4b8f125c059ca37b992dc9b787d9db2f84717d43c631bdeece40c2d00f4bab8` |

All are trained with 50 iterations, cutoff 1, on one thread, by
`FixtureModels.java` in `analysis/goldens-tool`, from the training data in
`analysis/goldens/data/train-fixtures/` (15 tagged sentences, the chunks of 8,
names in 14). Retraining them writes a new timestamp into each manifest, so
their hashes change, and the recipe's fixture hashes and every golden made from
them with them; the goldens-tool README says how to run it.

`pos-fixture.bin` is also run with `DefaultPOSContextGenerator` and the names
dictionary as an n-gram dictionary (`beam/pos-fixture+default-context`), as
a model from before OpenNLP 1.8 would be.

At commit cdf33c6, `BrownCluster` and `WordClusterDictionary` close the stream
they serialize to, which is the model's zip stream, so OpenNLP cannot write a
model that stores either. `ner-bio-fixture.bin` stores them through
`StoredClusters.java` in `analysis/goldens-tool`, which writes back the bytes the
resource was read from; its manifest names those serializers, so OpenNLP loads
it only with that class on the class path and its package registered with
`ExtensionLoader`. The crate reads the entries by the kind the descriptor gives
them and does not look at the serializer names.

`QNModel.eval(context, probs)` adds its sums to what `probs` holds, and
`BeamSearch` passes the same array to every evaluation, so inside a beam search
each evaluation of a QN model starts from the previous one's probabilities.
`beam/ner-qn-fixture` is what OpenNLP gives; `beam/ner-qn-fixture+fresh-sums`
runs the same search over a wrapper that
evaluates every context into a new array. Native `BeamSearch` matches the
second. `search.opennlp()` starts a compatibility session matching the first;
retain that session across sentences to carry QN probabilities as Java does.
Both tests run in CI and compare outcome labels, probability bits, score bits,
recorded contexts and decoded name spans for every sentence in the golden.

Each session borrows its search, so its model and outcome order cannot change.
Use one session per independent stream; no Rust global or thread-local state is
used. `reset()` clears its carried probabilities, corresponding to closing the
Java search for the current thread; it does not clear a context generator's
adaptive data. The session matches the JVM's default evaluation cache size of
zero. Native and scored search methods remain stateless between calls.
`MaxentModel::eval_features_seeded` is the explicit caller-buffer overload;
ordinary maxent evaluation continues to start fresh.

### The public models

`opennlp-en-ud-ewt-pos-1.3-2.5.4.bin` is fetched by `turbo-goldens
fetch-models` and read from `TURBO_GOLDENS_MODELS` (or
`target/turbo-goldens/models`), failing when missing if
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. The five OpenNLP 1.5 models are
local-only (their training corpora are LDC corpora or undocumented): CI does
not fetch them, `turbo-goldens fetch-models <dir> --local` does, and
`generate` and `verify` always do, since the generator runs them. The tests
over them run when `TURBO_TEST_OPENNLP_MODELS` names a directory holding them;
without it they say they were skipped and pass, even with
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which CI sets for this crate and which
covers only the fetched model. None are checked in.

| Model | Source | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-pos-1.3-2.5.4.bin` | OpenNLP UD models 1.3 (OpenNLP 2.5.4) | `510b89087e7ef7ffb8c50c63e3e999122404ac9fa64790dfa93b38d1938c52ce` |
| `en-pos-maxent.bin` | OpenNLP 1.5 models | `645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d` |
| `en-pos-perceptron.bin` | OpenNLP 1.5 models | `0b49b7d9bdb9f888aed85e9f41fbcfd6cab607805ba9cd2370e1e5af4e540db8` |
| `en-chunker.bin` | OpenNLP 1.5 models | `7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc` |
| `en-ner-person.bin` | OpenNLP 1.5 models | `687a9263d96b37fced707c9f2ac0560f9edaf54658856395555901924f64dbe4` |
| `en-ner-location.bin` | OpenNLP 1.5 models | `8fe39e48633f4a86c4132d9c54b396a2d8e0460c1d71e3562dacf976984f447b` |

`en-chunker.bin` runs on the tags of `en-pos-maxent.bin`.

The `tag-dictionary` and `tag-format` runs read the first three. The UD model
has no tag dictionary. The two 1.5 models have the same one, case sensitive
with 16,204 words; its goldens hold its hash and lookups rather than every
entry.
