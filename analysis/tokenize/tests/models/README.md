# Tokenizer models for the tests

| File | Origin | Size |
|---|---|---|
| `opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-en:1.3.0` | 326 KB |
| `opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-de:1.3.0` | 524 KB |
| `opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-es:1.3.0` | 539 KB |
| `opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-fr:1.3.0` | 544 KB |
| `opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-tokenizer-pt:1.3.0` | 400 KB |

The five UD models are the OpenNLP 1.3 models trained by OpenNLP 2.5.4 on
Universal Dependencies treebanks, released under the Apache License 2.0, as
the Maven artifacts ship them. Each file's SHA-512 is in the `model`
record of every golden made with it
(`analysis/goldens/data/tokenize/<model>/`), and the tests check it. None
of them has an abbreviation dictionary, and all have
`useAlphaNumericOptimization=false`.

The fixture model with an abbreviation dictionary that the tests also
run, `tokenizer-abbreviations.bin`, is one of the golden generator's
fixtures, in `analysis/goldens/data/fixtures` (that directory's README
says where it comes from).
