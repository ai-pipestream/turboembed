# Golden file generator

A plain Java program that runs the OpenNLP 3.x components the Rust ports
replace and writes what they returned to text files. The Rust tests read those
files and compare spans exactly and probabilities within 1e-9. The JVM is used
to make the files only; nothing calls it at test time. The design is in
`docs/segmentation.md`, section "Golden files".

## What it writes

For every input in `inputs/` (UTF-8 text files, one input each):

| Component | Directory under `analysis/tokenize/tests/golden` |
|---|---|
| `SimpleTokenizer` | `simple`, `simple-newlines` (keep new lines on), plus `-legacy` of both |
| `WhitespaceTokenizer` | `whitespace`, `whitespace-newlines`, plus `-legacy` of both |
| UAX #29 `WordTokenizer` (default maximum token length) | `uax29` (unicode mode only) |
| `TokenizerME` over the whole input | `tokenizer-me-<language>` |

| Component | Directory under `analysis/sentdetect/tests/golden` |
|---|---|
| `NewlineSentenceDetector` | `newline`, `newline-legacy` |
| `SentenceDetectorME` over the whole input | `sentdetect-me-<language>` |

Each file in these directories is named after its input (`<input>.txt`).
`analysis/tokenize/tests/golden/contexts/tokenizer-me-<language>.txt` holds, for five
inputs (`spanish`, `french`, `numbers`, `urls-emails`, `supplementary`), every context
`DefaultTokenContextGenerator` built and the two probabilities the model returned.

`analysis/tokenize/tests/golden/uax29-wordbreaktest.txt` is a plain copy of OpenNLP's
`WordBreakTest.txt` test resource (Unicode 17.0, license header kept). The program
does not write it.

Models are found by their manifest: every `*.bin` file in the models directory whose
`Component-Name` is `TokenizerME` gets tokenizer and context files, and one whose
`Component-Name` is `SentenceDetectorME` gets sentence files. The language in the
directory name is the manifest `Language`. To add a model, put it in the models directory
and run the program again; files of the other models are rewritten with the same content.

The whitespace mode is OpenNLP's `WhitespaceMode`. `unicode` (the default) is run for
every component. `legacy` is run for `simple`, `whitespace` and `newline` (and the two
`-newlines` variants).

## File format

Plain text, UTF-8, every line ends with a line feed.

```
component: simple
opennlp: <commit>
jdk: <java.version>
whitespace: unicode
model: <file name>                      (model runs only)
model-sha256: <hex>                     (model runs only)
alpha-numeric-pattern: <escaped>        (TokenizerME only; empty if the manifest has none)
use-alpha-numeric-optimization: <bool>  (TokenizerME only)
input: <escaped text>
span: <utf8 start> <utf8 end> <utf16 start> <utf16 end> [<probability>] [<type>]
```

- `input` escapes a backslash as `\\`, line feed, carriage return and tab as `\n`, `\r`,
  `\t`, and every other UTF-16 code unit outside U+0020..U+007E as `\uXXXX` (four upper
  case hex digits). A character outside the BMP is two surrogate escapes. An empty input
  is an empty value after `input: `.
- Offsets are half-open. UTF-16 offsets are what Java returned. UTF-8 offsets are the byte
  length of `text.substring(0, offset)` encoded as UTF-8.
- A span with an offset between the two halves of a surrogate pair has no UTF-8 offset. It
  is written as `span-invalid: <utf16 start> <utf16 end> [<probability>] [<type>]`.
- The probability is `Double.toString`, present for `TokenizerME` and `SentenceDetectorME`
  (one per span, from `probs()`). The type is the `WordType` name, present for `uax29`.
- If a component throws on an input, the file has an `error: <escaped class and message>`
  line instead of span lines.

Context files have the same header (component name `tokenizer-me-<language>-contexts`),
then `outcomes: T F`, and for each input an `input:` line followed by, for every
whitespace unit the model was consulted on, `unit: <utf16 start> <utf16 end>` and for
every position `at: <index>` (the index passed to `getContext`, relative to the unit),
`features: <feature>` separated by tabs (each escaped like the input) and
`probs: <P(T)> <P(F)>`. Units of fewer than two code units, and units the alphanumeric
check accepts, are not consulted and have no `unit:` line. The loop is a copy of
`TokenizerME.tokenizePos` (including the jump over the dots of a known abbreviation), and
the program checks that the spans it rebuilds equal `TokenizerME.tokenizePos` for each
input.

## Provenance

- OpenNLP: the 3.0.0-SNAPSHOT jars built from the OpenNLP checkout at commit
  `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` (recorded in each golden file's `opennlp:`
  line). The class path is the runtime, API and ML (commons, maxent, perceptron, bayes)
  jars of that build plus `slf4j-api` 2.0.19.
- JDK: OpenJDK 21.0.12.1 (the `jdk:` line of each file).
- Models: OpenNLP 2.5.4 UD models (Spanish, French, German, English, Portuguese
  tokenizers; German, English, Portuguese sentence models). Each file records the model
  name and SHA-256. The models are not in the repository.

## Running it

From the repository root, with `$CP` set to the class path above (jar files separated by
`:`), `$OPENNLP_COMMIT` set to the checkout's `git rev-parse HEAD`, and `$MODELS` set to
the directory of model files:

```
javac -d /path/to/classes -cp "$CP" analysis/golden/jvm/src/*.java
java -cp "/path/to/classes:$CP" Golden "$OPENNLP_COMMIT" analysis/golden/jvm/inputs "$MODELS" analysis
```

The last line printed is `<n> inputs, <n> models, <n> problems`; each problem (an
exception from a component, a mismatch in the context loop) is listed after it.

## Source files

| File | Role |
|---|---|
| `Golden.java` | Entry point; runs unicode mode, then legacy mode |
| `TokenizerRuns.java` | Simple, whitespace, UAX #29 and `TokenizerME` files |
| `SentenceRuns.java` | Newline and `SentenceDetectorME` files |
| `ContextRuns.java` | Feature and probability files |
| `GoldenFile.java` | Header, input and span lines |
| `TextEscape.java`, `Utf8Offsets.java` | Escaping and offset conversion |
| `ModelFile.java`, `ModelHeader.java` | Model discovery and header lines |
| `Inputs.java`, `RunInfo.java` | Input loading and per-run header values |

## Inputs

`inputs/*.txt` are the input texts. They cover abbreviations, numbers, quotes and
brackets, dashes and hyphens, ellipses, URLs and e-mail addresses, apostrophes, CR LF and
lone CR, tabs, no-break and ideographic spaces, `U+0085`, `U+2028`, `U+2029`, control
separators `U+001C..U+001F`, leading and trailing whitespace, an empty input and a
whitespace-only input, Spanish, French and German, Greek, Cyrillic, Hebrew, Arabic,
Japanese, Chinese, Korean, Thai, emoji (single, with skin tone, ZWJ sequence, flags,
keycap), combining marks, characters outside the BMP, and two long words (300 letters,
and 254 letters followed by an emoji so that the 255 code unit cut falls inside a pair).

## Behaviour of OpenNLP seen in the files

- `SimpleTokenizer` with new lines kept returns `Span(start, start + 1)` for a line break
  using the `start` of the last token, so a line break after a space, or the second of two
  line breaks separated by a space, gets a span that covers the wrong character
  (`simple-newlines/leading-trailing.txt`). If no token has started yet, `start` is -1 and
  the call throws (`simple-newlines/whitespace-only.txt`, which therefore has an `error:`
  line). `WhitespaceTokenizer` with new lines kept is correct.
- `SimpleTokenizer` classifies code units, so a character outside the BMP becomes two
  spans of half a character each (`span-invalid:` lines in `simple`).
