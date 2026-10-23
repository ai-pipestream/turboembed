# Fixture sentence models

The published OpenNLP sentence models have no abbreviation dictionary, so
two small English models are trained here to test the dictionary path,
`useTokenEnd=false` and the context features one by one.

## Files

| Path | What it is |
|---|---|
| `sentences.txt` | About 210 hand-written English sentences, one per line, an empty line between documents, with abbreviations (Mr., Dr., U.S., e.g., p.m.) and numbers |
| `src/TrainSentenceModel.java` | Trains the two models |

The models it trains, `en-abbrev.bin` (GIS maxent, cutoff 1, 100
iterations; `useTokenEnd=true`; a case-insensitive abbreviation
dictionary of 13 entries, one of them of two tokens, `et al.`) and
`en-plain.bin` (the same training without a dictionary,
`useTokenEnd=false`), are the golden generator's fixtures, in
`analysis/goldens/data/fixtures`. Their 17 inputs are
`analysis/goldens/data/inputs/sentence-fixture`, and their goldens and
dumps, in both whitespace modes, are
`analysis/goldens/data/sentdetect/en-abbrev[+legacy]` and
`en-plain[+legacy]`, made by `turbo-goldens generate`
(`analysis/goldens-tool/README.md`).

Both models declare the English end characters `. ! ?` in their manifest,
as the factory writes them.

## Training them again

From the repository root, with `$CP` set to the class path of the OpenNLP
3.x build the goldens name (its runtime, API and ML jars and `slf4j-api`):

```
F=analysis/sentdetect/tests/fixtures
javac -d /path/to/classes -cp "$CP" $F/src/TrainSentenceModel.java
java -cp "/path/to/classes:$CP" TrainSentenceModel $F/sentences.txt analysis/goldens/data/fixtures
```

A model trained again has new bytes, so its SHA-512 in
`analysis/goldens-tool/src/recipe/fixtures.rs` changes with it and the goldens are
generated again.
