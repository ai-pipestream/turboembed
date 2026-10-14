# Sentence models

The tests run three published OpenNLP sentence detector models. They are
not checked in: the models are under the Apache License 2.0, but the
Universal Dependencies treebanks they were trained on are CC BY-SA, and
CC BY-SA data is downloaded, not checked in (`docs/licences.md`).

| File | Origin | SHA-256 |
|---|---|---|
| `opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin` | Apache OpenNLP UD models 1.3 (`downloads.apache.org/opennlp/models/ud-models-1.3`, also Maven Central `org.apache.opennlp:opennlp-models-sentdetect-de:1.3.0`) | `b5553223d30a0422e80a28e2ae766a92dd7181e229f0d6c73087951e84142c43` |
| `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin` | the same release (`opennlp-models-sentdetect-en:1.3.0`) | `4de58d12b09421003c8b1ba13e8b58ec533b28e1f22a65272119dc0993d333e8` |
| `opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin` | the same release (`opennlp-models-sentdetect-pt:1.3.0`) | `a711463f146555de26d69d806a9f1a88a66ed0c02f3fe2020b4a475ccea047ff` |

`cargo run -p turbo-goldens-tool -- fetch-models <dir>` downloads them, with
the other models of `analysis/goldens-tool/src/recipe.rs`, and checks each
against its published SHA-512. The tests read them from the directory
`TURBO_GOLDENS_MODELS` names, or `target/turbo-goldens/models` under the
repository root:

```text
cargo run -p turbo-goldens-tool -- fetch-models target/turbo-goldens/models
cargo test -p turbo-sentdetect
```

A test that needs a model that is not there says so and passes, unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1` makes that a failure, as CI sets it
after fetching the models.

The models were trained by OpenNLP 2.5.4 on German GSD, English EWT and
Portuguese GSD. All three are GIS maxent models with `useTokenEnd=true` and
no abbreviation dictionary. The German and English models' end-of-sentence
characters are `. ! ?`; the Portuguese model has the extended set
`. ? ! ; : ( ) « » ' "`.

The goldens in `analysis/goldens/data/sentdetect/<model>/` were made with
these files (each golden records the model's SHA-512, which the tests
check). The two fixture models the tests also run, `en-abbrev.bin` and
`en-plain.bin`, are trained for this project and checked in, in
`analysis/goldens/data/fixtures` (`analysis/sentdetect/tests/fixtures/README.md`
has their training sentences and how they were made).
