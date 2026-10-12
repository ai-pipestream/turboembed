# Tokenizer models for the tests

| File | Origin | Size |
|---|---|---|
| `opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-en:1.3.0` | 326 KB |
| `opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-de:1.3.0` | 524 KB |
| `opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-es:1.3.0` | 539 KB |
| `opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-fr:1.3.0` | 544 KB |
| `opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-pt:1.3.0` | 400 KB |
| `tokenizer-abbreviations.bin` | a copy of `analysis/maxent/tests/fixtures/models/tokenizer.bin` | 3 KB |

The five UD models are the OpenNLP 1.3 models trained by OpenNLP 2.5.4 on
Universal Dependencies treebanks, released under the Apache License 2.0, as
the Maven artifacts ship them. Each file's SHA-256 is in the `model-sha256`
line of every golden file made with it (`tests/golden/tokenizer-me-*`), and
the tests check it. None of them has an abbreviation dictionary,
and all have `useAlphaNumericOptimization=false`.

`tokenizer-abbreviations.bin` is a small model OpenNLP trained on six
sentences with a case-insensitive abbreviation dictionary (`Dr.`, `U.S.`,
`e.g.`). The program that made it is described in
`analysis/maxent/tests/fixtures/README.md`. It is here so that loading a
model's dictionary and the abbreviation branch of the tokenizer are tested
with a file OpenNLP wrote.
