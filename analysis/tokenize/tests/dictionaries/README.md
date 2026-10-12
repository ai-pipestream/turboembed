# Detokenizer dictionaries

Copied unchanged from Apache OpenNLP 3.x, which is licensed under the
Apache License, Version 2.0 (each file carries the license header):

- `latin-detokenizer.xml`: the test resource of `DictionaryDetokenizerTest`,
  `opennlp-core/opennlp-runtime/src/test/resources/opennlp/tools/tokenize/`.
- `en-detokenizer.xml`, `fr-detokenizer.xml`, `pt-detokenizer.xml`: the
  English, French and Portuguese detokenizer dictionaries,
  `opennlp-tools/lang/{en,fr,pt}/tokenizer/`.

The expected outputs in `tests/detok.rs` that are not from the Java
test itself are what OpenNLP's `DictionaryDetokenizer` printed for the
same tokens with these dictionaries, with no split marker and with the
marker `<SPLIT>`.
