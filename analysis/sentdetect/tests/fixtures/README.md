# Fixture sentence models

The published OpenNLP sentence models have no abbreviation dictionary, so
two small English models are trained here to test the dictionary path,
`useTokenEnd=false` and the context features one by one.

## Files

| Path | What it is |
|---|---|
| `sentences.txt` | About 210 hand-written English sentences, one per line, an empty line between documents, with abbreviations (Mr., Dr., U.S., e.g., p.m.) and numbers |
| `models/en-abbrev.bin` | GIS maxent, cutoff 1, 100 iterations; `useTokenEnd=true`; a case-insensitive abbreviation dictionary of 13 entries, one of them of two tokens (`et al.`) |
| `models/en-plain.bin` | The same training without a dictionary, `useTokenEnd=false` |
| `inputs/*.txt` | 17 inputs: abbreviations at the start, in upper case, after a bracket or an apostrophe; numbers with dots; quotes after the period; `z.B.` style runs and an upper case letter after a period; a split whose token runs on past it (`hard.Bekannt.`), so that with `useTokenEnd` the next end character is before the sentence start and is passed over; a lone `.`; a text without a final period; whitespace only; CR LF and lone CR; leading and trailing whitespace; `U+0085`, `U+2028`, `U+2029`; `U+001C` and `U+001F`; emoji inside a sentence; a character outside the BMP right after a period, also at the very end |
| `golden/sentdetect-me-fixture-<model>/` | `SentenceDetectorME.sentPosDetect` and `probs()` per input, in the format of `analysis/golden/jvm/README.md` |
| `golden/sentdetect-me-fixture-<model>-legacy/` | The same in the legacy whitespace mode |
| `golden/contexts/sentdetect-me-fixture-<model>[-legacy].txt` | Every candidate the detector asked the model about, per input |
| `src/TrainSentenceModel.java` | Trains the two models |
| `src/GoldenSentences.java` | Writes the golden and context files |

Both models declare the English end characters `. ! ?` in their manifest,
as the factory writes them. Each golden file records the model's SHA-256.

## Context files

The header is that of a golden file (component name
`sentdetect-me-fixture-<model>-contexts`), then `outcomes: s n`. For each
input there is an `input:` line, then for every candidate the model was
asked about:

```
at: <UTF-16 index of the end character>
features: <feature>	<feature>	...      (tab separated, each escaped like the input)
probs: <P(s)> <P(n)>
decision: split | no-split | vetoed
```

`no-split` means the model's best outcome is not a split; `vetoed` means
it is, and the abbreviation dictionary refused it. The loop that writes
them is a copy of `sentPosDetect`; the program rebuilds the spans from its
own decisions and reports a problem when they differ from what
`sentPosDetect` returns.

## Running it

From the repository root, with `$CP` set to the OpenNLP class path of
`analysis/golden/jvm/README.md` and `$OPENNLP_COMMIT` to the commit the
jars were built from (the files here were made at
`cdf33c653d5d61d5af7d37ffee189a83bd8392ea`, with OpenJDK 21.0.12.1):

```
F=analysis/sentdetect/tests/fixtures
G=analysis/golden/jvm/src
javac -d /path/to/classes -cp "$CP" $F/src/TrainSentenceModel.java
java -cp "/path/to/classes:$CP" TrainSentenceModel $F/sentences.txt $F/models

javac -d /path/to/classes -cp "$CP" $G/GoldenFile.java $G/TextEscape.java $G/Utf8Offsets.java \
  $G/RunInfo.java $G/Inputs.java $G/ModelFile.java $G/ModelHeader.java $F/src/GoldenSentences.java
java -cp "/path/to/classes:$CP" GoldenSentences "$OPENNLP_COMMIT" $F/inputs $F/models $F/golden
```

The last line `GoldenSentences` prints is `<n> inputs, <n> models, <n>
problems`, each problem listed after it. Training writes the time into the
manifest, so retrained models differ byte for byte; the golden files must
then be written again, since they record each model's SHA-256.
