# SymSpell test data

| File | What it is |
|---|---|
| `frequency_dictionary_tiny.txt`, `frequency_bigramdictionary_tiny.txt` | OpenNLP's test fixtures (`src/test/resources/opennlp/spellcheck/` of the spellcheck add-on), copied unchanged. `TinyDictionary` builds its engine from them, and so do the tests here. |
| `model-tiny.bin` | A model written by OpenNLP's `SymSpellModelSerializer`: the default configuration, the two tiny dictionaries, and four more words that exercise the encodings (`café`, a word with U+0000, a word with the supplementary character U+10428, and the count `Long.MAX_VALUE`). |
| `model-levenshtein.bin` | A model written the same way with the Levenshtein metric, maximum edit distance 1, prefix length 4, count threshold 3 (two of its eight words stay below it) and a fixed corpus size of 10^9. |
| `expected.txt` | What OpenNLP returns for 2511 lookups (every verbosity and distance), 2015 compound lookups and 4218 runs of `SpellCheckingCharSequenceNormalizer` in six configurations, over the two models above and phrases generated from their words. |
| `GenerateSpellcheckData.java` | The program that wrote the three files above. |

Every line of `expected.txt` is tab-separated: `lookup`, the model
(`tiny` or `levenshtein`), the verbosity, the maximum distance, the term,
then the suggestions as `term,distance,count` joined by `|`; `compound`,
the model, the maximum distance, the input, then `term,distance,count`;
`normalize`, the configuration, the input, then the output. In the
strings a backslash, `|`, `,`, every control character, every White_Space
character except U+0020, U+FEFF and every supplementary character are
written `\u{hex}`.

## How the files were made

The OpenNLP classes are those of the spellcheck add-on (branch
`ready/spellcheck-recase` of the OpenNLP fork, commit `275c81bb`) and of
OpenNLP itself (branch `kristian-3.x-features`, commit `cdf33c65`),
compiled from source with OpenJDK 21.0.12 and `slf4j-api` 2.0.19 on the
class path. The add-on classes the program uses call three methods that
OpenNLP at that commit does not have yet, so three small classes on the
source path ahead of OpenNLP's supply them; none of them is on the path
that writes the files (the dictionary loader, which needs two more
missing `StringUtil` methods, is not compiled; the program reads the two
dictionaries itself):

- `opennlp.tools.util.ArgumentChecks.requireNonNullArg(Object, String)`,
  which throws `IllegalArgumentException` for `null`;
- `opennlp.tools.util.ResourceLimits.requireWithinMaxEntries(int, String,
  Function)`, a copy of OpenNLP's `ResourceLimits` with that method added
  (it throws the given exception when the count is below 0 or above
  `MAX_ENTRIES`), used only when a model is read;
- `opennlp.tools.util.normalizer.MailAddressScan`, a copy of OpenNLP's
  class made public with `isAddress(text)` added, true when an address
  found by `matchEnd(text, 0)` covers the whole text. The unit tests in
  `analysis/normalize/src/spellcheck/guards.rs` hold this definition to
  the reference pattern of OpenNLP's own normalizer test.

With `SP` the add-on's `src/main/java` directory, `API` and `RUNTIME` the
`src/main/java` directories of `opennlp-api` and `opennlp-runtime`, and
`shim` the directory with the three classes:

```
javac -cp slf4j-api-2.0.19.jar -d out \
  -sourcepath "shim:$SP:$API:$RUNTIME" GenerateSpellcheckData.java
java -cp out:slf4j-api-2.0.19.jar GenerateSpellcheckData <this directory>
```

The program reads the two dictionaries from the directory it is given and
writes the other three files there. Its inputs come from
`java.util.Random` with seed 189, so a second run writes the same bytes.
The generated text keeps to characters of Unicode 15.0 (the JDK's
version), and the phrases and the edited terms to the Basic Multilingual
Plane (the one supplementary word is looked up only as it is), so neither
the JDK's Unicode tables nor OpenNLP's UTF-16 lengths are the reason
for a difference.

The fixtures and the OpenNLP classes are under the Apache License 2.0.
