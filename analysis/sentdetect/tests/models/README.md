# Sentence models

Three OpenNLP sentence detector models, checked in so that the golden tests
run on any machine:

| File | Maven artifact | SHA-256 |
|---|---|---|
| `opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin` | `org.apache.opennlp:opennlp-models-sentdetect-de:1.3.0` | `b5553223d30a0422e80a28e2ae766a92dd7181e229f0d6c73087951e84142c43` |
| `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin` | `org.apache.opennlp:opennlp-models-sentdetect-en:1.3.0` | `4de58d12b09421003c8b1ba13e8b58ec533b28e1f22a65272119dc0993d333e8` |
| `opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin` | `org.apache.opennlp:opennlp-models-sentdetect-pt:1.3.0` | `a711463f146555de26d69d806a9f1a88a66ed0c02f3fe2020b4a475ccea047ff` |

They come from Maven Central, are licensed under the Apache License 2.0,
and were trained by OpenNLP 2.5.4 on Universal Dependencies treebanks
(German GSD, English EWT, Portuguese GSD). All three are GIS maxent models
with `useTokenEnd=true` and no abbreviation dictionary. The German and
English models' end-of-sentence characters are `. ! ?`; the Portuguese
model has the extended set `. ? ! ; : ( ) « » ' "`.

The goldens in `analysis/goldens/data/sentdetect/<model>/` were made with
these files (each golden records the model's SHA-512, which the tests
check).
