# Part-of-speech models for the tests

| File | Origin | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-pos-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-pos-en:1.3.0` | `510b89087e7ef7ffb8c50c63e3e999122404ac9fa64790dfa93b38d1938c52ce` |
| `opennlp-de-ud-gsd-pos-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-pos-de:1.3.0` | `073fefbfaff2ed403bf474fccf793752d01c22e69f487cd68b43832d3123742f` |

The two UD models are the OpenNLP 1.3 models trained by OpenNLP 2.5.4 on
Universal Dependencies treebanks (English EWT, German GSD), released under
the Apache License 2.0, as the Maven artifacts ship them. Both are maxent
models with `BeamSize=3`, a feature generator descriptor
(`generator.featuregen`), Universal Dependencies outcomes and no tag
dictionary. Each file's SHA-512 is in the first `model` record of every
golden made with it (`analysis/goldens/data/postag/<model>/`), and the
tests check it.

The fixture model with a tag dictionary that the tests also run,
`pos-fixture.bin`, is one of the golden generator's fixtures, in
`analysis/goldens/data/fixtures` (that directory's README says where it
comes from).
