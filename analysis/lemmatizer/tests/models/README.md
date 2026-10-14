# Lemmatizer models for the tests

| File | Origin | SHA-256 |
|---|---|---|
| `opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-lemmatizer-en:1.3.0` | `711ad051faba79865c5f9fc5b8c455d6ad8e6d3e8b92f42142af1ef0bc7b3a96` |
| `opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin` | Maven Central `org.apache.opennlp:opennlp-models-lemmatizer-de:1.3.0` | `8850986f02dfbe293046bfce93bc0c3f3fa010e89177250abca23d860033d4f6` |

The two UD models are the OpenNLP 1.3 models trained by OpenNLP 2.5.4 on
Universal Dependencies treebanks (English EWT, German GSD), released under
the Apache License 2.0, as the Maven artifacts ship them. Both are maxent
models with `BeamSize=3`, the default factory and so the default context
generator, and lemma classes (shortest edit scripts) as outcomes: 860 in
the English model. Their features name Universal Dependencies tags
(`t0=NOUN`), so they run on the tags of the UD part-of-speech models. Each
file's SHA-512 is in the first `model` record of every golden made with it
(`analysis/goldens/data/lemmatize/<model>/`), and the tests check it.

The dictionary the dictionary lemmatizer's tests run,
`lemmatizer-dictionary.txt`, is one of the golden generator's fixtures, in
`analysis/goldens/data/fixtures` (that directory's README describes it).
