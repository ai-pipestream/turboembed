# Bundled stopword lists

These are the eleven stopword lists OpenNLP bundles, copied unchanged from
`opennlp-core/opennlp-runtime/src/main/resources/opennlp/tools/stopword/` at
commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the OpenNLP fork.
`SHA256SUMS` lists each file's digest, and `golden/generate.sh` fails if a file
here differs from the resource in the OpenNLP jar.

They are used under the BSD license reproduced in `LICENSE` in this directory,
which is the text OpenNLP's own `LICENSE` file gives for them. OpenNLP's
`NOTICE` says the lists are derived from Apache Lucene, which distributes them
under the Snowball project's BSD license, and that the Bulgarian list is
Lucene's, created by Jacques Savoy and also under the BSD license. Each file's
header, kept as OpenNLP ships it, names its origin.

| File | Language | Origin, as the file's header gives it | Changes the header records | Words |
|---|---|---|---|---|
| `bg.txt` | Bulgarian | Created by Jacques Savoy (his CLEF resources page), BSD license | Comments changed from `\|` to `#` | 190 |
| `da.txt` | Danish | Snowball, `algorithms/danish/stop.txt`, BSD license | Converted to UTF-8, notice added, comments changed from `\|` to `#` | 94 |
| `de.txt` | German | Snowball, `algorithms/german/stop.txt`, BSD license | As for Danish | 231 |
| `en.txt` | English | Snowball, `algorithms/english/stop.txt`, BSD license | As for Danish | 174 |
| `es.txt` | Spanish | Snowball, `algorithms/spanish/stop.txt`, BSD license | As for Danish | 308 |
| `fi.txt` | Finnish | Snowball, `algorithms/finnish/stop.txt`, BSD license | As for Danish, and the pronoun and determiner paradigm rows, which Snowball writes as columns on one line, expanded to one word per line | 229 (232 lines; three words appear twice) |
| `fr.txt` | French | Snowball, `algorithms/french/stop.txt`, BSD license | As for Danish | 154 |
| `it.txt` | Italian | Snowball, `algorithms/italian/stop.txt`, BSD license | As for Danish | 279 |
| `nl.txt` | Dutch | Snowball, `algorithms/dutch/stop.txt`, BSD license | As for Danish | 101 |
| `pt.txt` | Portuguese | Snowball, `algorithms/portuguese/stop.txt`, BSD license | As for Danish | 203 |
| `ru.txt` | Russian | Snowball, `algorithms/russian/stop.txt`, BSD license | As for Danish | 159 |

The word counts are the distinct words OpenNLP loads (`StopwordFilter.stopwords()`),
which `golden/expected/lists.txt` records.
