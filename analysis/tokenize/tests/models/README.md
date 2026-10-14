# Tokenizer models for the tests

The tests run five published OpenNLP tokenizer models. They are not
checked in: the models are under the Apache License 2.0, but the Universal
Dependencies treebanks they were trained on are CC BY-SA, and CC BY-SA data
is downloaded, not checked in (`docs/licences.md`).

| File | Origin | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin` | Apache OpenNLP UD models 1.3 (`downloads.apache.org/opennlp/models/ud-models-1.3`, also Maven Central `org.apache.opennlp:opennlp-models-tokenizer-en:1.3.0`) | `95139111197ba9f1165fc12be8d4cbac05ea3912f995d167efa1a0e04e3e649a` |
| `opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin` | the same release (`opennlp-models-tokenizer-de:1.3.0`) | `32d0a7ff84fdd50f9e454340bed782d7269845bfa656e0c988cbb823dd628d6e` |
| `opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin` | the same release (`opennlp-models-tokenizer-es:1.3.0`) | `309de611ff7a8c4227c57086838d4783bcede8520026d7ad416c85a513567ce7` |
| `opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin` | the same release (`opennlp-models-tokenizer-fr:1.3.0`) | `a06dbc9b72600ba0886f64a632636e136e5fc225955913c4c1a926a9333bc14a` |
| `opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin` | the same release (`opennlp-models-tokenizer-pt:1.3.0`) | `b85f42a6a1dfe0728b7c7c9df1ca0c98de708c04be3df22da1917b8d8dc4b096` |

`cargo run -p turbo-goldens-tool -- fetch-models <dir>` downloads them, with
the other models of `analysis/goldens-tool/src/recipe.rs`, and checks each
against its published SHA-512. The tests read them from the directory
`TURBO_GOLDENS_MODELS` names, or `target/turbo-goldens/models` under the
repository root:

```text
cargo run -p turbo-goldens-tool -- fetch-models target/turbo-goldens/models
cargo test -p turbo-tokenize
```

A test that needs a model that is not there says so and passes, unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1` makes that a failure, as CI sets it
after fetching the models.

The five models were trained by OpenNLP 2.5.4 on Universal Dependencies
treebanks (English EWT, German, Spanish, French and Portuguese GSD). Each
file's SHA-512 is in the `model` record of every golden made with it
(`analysis/goldens/data/tokenize/<model>/`), and the tests check it. None
of them has an abbreviation dictionary, and all have
`useAlphaNumericOptimization=false`.

The fixture model with an abbreviation dictionary that the tests also
run, `tokenizer-abbreviations.bin`, is trained for this project and checked
in as one of the golden generator's fixtures, in
`analysis/goldens/data/fixtures` (`analysis/goldens/data/README.md` says
where it comes from). The unit tests that do not depend on a model's decisions run
on it, so they need no download.
