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

The tests compare the crate with files a JVM running OpenNLP wrote. The Java
programs, their inputs and their outputs are under `golden/`. The JVM makes the
files only; nothing calls it at test time.

This generator is standalone for now. Once the `goldens-tool` rework (#316)
lands, it moves onto `analysis/goldens-tool` with the other OpenNLP goldens.

| Path | Made by | Read by | Contents |
|---|---|---|---|
| `inputs/sentences.txt` | by hand | all | 16 tokenized sentences, one per line, tokens separated by single spaces: abbreviations, numbers, dates, URLs, accented, Greek, Cyrillic, CJK and kana text, letters and an emoji outside the Basic Multilingual Plane, a titlecase letter, a one-token sentence |
| `inputs/tokens.txt` | by hand | `PatternGolden` | Tokens for `StringPattern`, escaped, lone surrogates included; the empty last line is the empty token |
| `inputs/train-*.txt` | by hand | `FixtureModels` | Training data of the fixture models: 15 tagged sentences, the chunks of 8, names in 14 |
| `resources/` | by hand | descriptors, `FixtureModels` | A Brown cluster file, a word cluster file, a case-insensitive names dictionary with multi-token entries and a tag dictionary |
| `descriptors/pos-default.xml`, `ner-default.xml` | copied from OpenNLP | `FeatureGolden`, `tests/features.rs` | OpenNLP's default part-of-speech and name finder descriptors (the same files as `src/descriptors/`) |
| `descriptors/everything.xml` | by hand | `FeatureGolden`, `FixtureModels` | Every factory except the window, aggregated and cached ones (those are in `nesting.xml`), with the three resources |
| `descriptors/nesting.xml` | by hand | `FeatureGolden`, `FixtureModels` | `cache="TRUE"` on the root, windows and aggregates inside each other, a cache inside the tree, an aggregate of one, a negative window length |
| `models/*.bin` | `FixtureModels.java` | `tests/beam.rs` | Small models OpenNLP trained on `inputs/train-*.txt` (below) |
| `expected/pattern-units.txt` | `PatternGolden.java` | `tests/pattern.rs` | `StringPattern` flags and digit count and `tokenFeature` for about 2,900 tokens: the inputs, the sentences' tokens, every code point of 18 ranges, lone surrogates, and a capital and a period |
| `expected/pattern-code-points.txt` | `PatternGolden.java` with the branch classes | `tests/pattern.rs` | The same, from the code point version |
| `expected/features-<descriptor>.txt` | `FeatureGolden.java` | `tests/features.rs` | The features each descriptor's generator gives every token of the sentences, run as one document with `updateAdaptiveData` after each sentence |
| `expected/features-default-pos*.txt` | `FeatureGolden.java` | `tests/features.rs` | `DefaultPOSContextGenerator`, without and with the names dictionary as its n-gram dictionary |
| `expected/descriptor-errors.txt` | `FeatureGolden.java` | `tests/features.rs` | 16 malformed descriptors and the exception `GeneratorFactory.create` throws for each |
| `expected/beam-<model>.txt` | `BeamGolden.java` | `tests/beam.rs` | Every sequence of the final beam for each sentence (outcomes, probabilities and score as raw double bits), every context the search asked for in sentences 0 and 8, and the name finder's spans |
| `expected/codec-bio.txt`, `codec-bilou.txt` | `CodecGolden.java` | `tests/codec.rs` | Decoding every outcome sequence of up to three labels from an alphabet of 10 (BIO) or 12 (BILOU), the validator on every prior sequence of up to two labels and every label, encoding 7 span lists, and `areOutcomesCompatible` on every subset of 6 or 7 outcomes |

`BeamGolden` wires each search as `POSTaggerME`, `ChunkerME` and
`NameFinderME` do (the part-of-speech context cache as large as the beam, the
name finder's additional context window, `updateAdaptiveData` after each
sentence) but calls `bestSequences` with the beam size as the count, so the
whole final beam is recorded. It checks each sentence's best sequence against
what the tagger itself returns. The chunker runs on the tags of the
part-of-speech model before it in the list.

Strings are escaped: every UTF-16 code unit outside U+0020..U+007E, and the
backslash, is `\uXXXX`, and tab, line feed and carriage return are `\t`, `\n`
and `\r`. A double is its 16 hex digit bit pattern.

### Fixture models

| Model | Trainer | Descriptor | Codec | SHA-256 |
|---|---|---|---|---|
| `pos-fixture.bin` | GIS, with `resources/tags.tagdict` | `pos-default.xml` | | `db87fc45de70bfe0175bfc402eb5c349a3e7c85abd22a6507c20b69fe50edef8` |
| `chunker-fixture.bin` | perceptron | | | `cc68fdbac1ee51695e232e59ac02754dde51e5fb1dd50d15c565c805204f23b0` |
| `ner-bio-fixture.bin` | perceptron | `everything.xml`, its resources stored in the model | BIO | `da05d773f20be0b8325e6dee3e0a1a103449f694c27275c58cb6f6f29e281081` |
| `ner-bilou-fixture.bin` | GIS | `nesting.xml` | BILOU | `8ebc2bc170c11cfa35eadbe89f35871ff6938ae29c08def57c3d47e95be27ae5` |
| `ner-qn-fixture.bin` | QN | the default one | BIO | `f4b8f125c059ca37b992dc9b787d9db2f84717d43c631bdeece40c2d00f4bab8` |

All are trained with 50 iterations, cutoff 1, on one thread. Retraining them
writes a new timestamp into each manifest, so their SHA-256 changes and every
beam file with it.

`pos-fixture.bin` is also run with `DefaultPOSContextGenerator` and the names
dictionary as an n-gram dictionary (`beam-pos-fixture-default-context.txt`), as
a model from before OpenNLP 1.8 would be.

At commit cdf33c6, `BrownCluster` and `WordClusterDictionary` close the stream
they serialize to, which is the model's zip stream, so OpenNLP cannot write a
model that stores either. `ner-bio-fixture.bin` stores them through
`golden/src/golden/StoredClusters.java`, which writes back the bytes the
resource was read from; its manifest names those serializers, so OpenNLP loads
it only with that class on the class path and its package registered with
`ExtensionLoader`. The crate reads the entries by the kind the descriptor gives
them and does not look at the serializer names.

`QNModel.eval(context, probs)` adds its sums to what `probs` holds, and
`BeamSearch` passes the same array to every evaluation, so inside a beam search
each evaluation of a QN model starts from the previous one's probabilities.
`beam-ner-qn-fixture.txt` is what OpenNLP gives;
`beam-ner-qn-fixture-fresh-sums.txt` runs the same search over a wrapper that
evaluates every context into a new array. The crate matches the second; the
test for the first is ignored until `turbo-maxent` can evaluate from given sums.

### The public models

The beam tests for these run when `TURBO_TEST_OPENNLP_MODELS` names a directory
holding them; without it they say they were skipped and pass, unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. They are not checked in.

| Model | Source | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-pos-1.3-2.5.4.bin` | OpenNLP UD models 1.3 (OpenNLP 2.5.4) | `510b89087e7ef7ffb8c50c63e3e999122404ac9fa64790dfa93b38d1938c52ce` |
| `en-pos-maxent.bin` | OpenNLP 1.5 models | `645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d` |
| `en-pos-perceptron.bin` | OpenNLP 1.5 models | `0b49b7d9bdb9f888aed85e9f41fbcfd6cab607805ba9cd2370e1e5af4e540db8` |
| `en-chunker.bin` | OpenNLP 1.5 models | `7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc` |
| `en-ner-person.bin` | OpenNLP 1.5 models | `687a9263d96b37fced707c9f2ac0560f9edaf54658856395555901924f64dbe4` |
| `en-ner-location.bin` | OpenNLP 1.5 models | `8fe39e48633f4a86c4132d9c54b396a2d8e0460c1d71e3562dacf976984f447b` |

`en-chunker.bin` runs on the tags of `en-pos-maxent.bin`.

### Regenerating

```sh
analysis/sequence/golden/generate.sh <opennlp checkout> [<directory of the public models>]
```

The checkout must be at commit cdf33c6, built and installed into the local
Maven repository (`mvn install -DskipTests`), with commit b2473ca of
`ready/string-pattern-unicode` fetched. The class path is the runtime, API and
ML (commons, maxent, perceptron, bayes) jars of that build and `slf4j-api`
2.0.19. For the code point `StringPattern` the script compiles the branch's
`StringPattern`, `FeatureGeneratorUtil` and `ParamChecks` (which cdf33c6 does
not have) and puts them ahead of those jars. With `TRAIN=1` it retrains the
fixture models first.

The files here were made with OpenJDK 21.0.12.1.
