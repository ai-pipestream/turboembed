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

`analysis/tokenize/tests/golden/uax29-wordbreaktest.txt` is an unchanged copy of
Unicode's `auxiliary/WordBreakTest.txt` for Unicode 18.0.0 (license header kept). The
program does not write it.

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

## Variant runs

`Variants` writes the files below, which the plain run cannot, because no UD model has an
abbreviation dictionary or the alphanumeric optimization, and the detokenizer has no model.
It reads `variants/` (committed data, below) and the same `inputs/` and models directory, and
leaves the files above untouched.

| Run | Directory under `analysis/tokenize/tests/golden` | Inputs |
|---|---|---|
| `TokenizerME` of the fixture model `variants/abbreviations/tokenizer-abbreviations.bin` (OpenNLP-trained on six sentences, case-insensitive dictionary `Dr.`, `U.S.`, `e.g.` inside the model file) | `variants/abbreviations-fixture` | `variants/abbreviations/*.txt` and `inputs/` |
| `TokenizerME` of the English UD model with `variants/abbreviations/english-dictionary.xml` added through `TokenizerFactory` | `variants/abbreviations-en` | same |
| `TokenizerME` of each UD model on inputs where it splits between the halves of a surrogate pair | `variants/surrogate-<language>` | `variants/surrogate/*.txt` (the file name starts with the language it splits in; every model runs every file) |
| `TokenizerME` of each UD model rebuilt through `TokenizerFactory` with `useAlphaNumericOptimization` on and the model's own pattern | `variants/alphanumeric-<language>` | `inputs/` and `variants/alphanumeric/*.txt` |
| Feature and probability files for the abbreviation and alphanumeric runs | `variants/contexts/tokenizer-me-<name>.txt` | the abbreviation inputs; the five context inputs and `mixed` |
| `DictionaryDetokenizer` | `detokenizer/<case>.txt` | `variants/detokenizer/cases/*.txt` with `variants/detokenizer/dictionaries/` |

Tokenizer files have the format above plus a line after the model lines: `abbreviations:` (the
dictionary file and its SHA-256) for the English run and `effective-alpha-numeric-pattern:` for the
alphanumeric runs. The `alpha-numeric-pattern` line of a rebuilt model is the pattern of the loaded
model, which for every UD model is its manifest's pattern, the pattern of the model's language.

`variants/abbreviations/tokenizer-abbreviations.bin` is a copy of
`analysis/maxent/tests/fixtures/models/tokenizer.bin` (SHA-256
`bd5334327d3c4ffab19765bec7755b71a9eaea50e4c23ce10712f48bdb14c7e1`), which `MakeModelFixtures.java`
trains as described in `analysis/maxent/tests/fixtures/README.md` (added by PR #277). This
program does not write it.

A detokenizer case file has one token per line (the part of the name before the first `-` names
the dictionary `<name>-detokenizer.xml`; an empty file is an empty token list). The golden file is:

```
component: detokenizer
opennlp: <commit>
jdk: <java.version>
whitespace: unicode
dictionary: <file name>
dictionary-sha256: <hex>
split-marker: <SPLIT>
tokens: <count>
token: <escaped token> <DetokenizationOperation name>     (one per token)
text: <escaped detokenize(tokens, null)>
text-marked: <escaped detokenize(tokens, "<SPLIT>")>
```

`variants/detokenizer/dictionaries/{en,fr,pt,latin}-detokenizer.xml` are Apache OpenNLP's files
(license header kept); `quotes-detokenizer.xml` is two `RIGHT_LEFT_MATCHING` entries.

Run it the same way as `Golden`, with the five UD tokenizer models in `$MODELS`:

```
java -cp "/path/to/classes:$CP" Variants "$OPENNLP_COMMIT" analysis/golden/jvm/variants \
    analysis/golden/jvm/inputs "$MODELS" analysis
```

## Alphanumeric pattern

`TokenizerFactory.getAlphaNumericPattern` returns the manifest's `alphaNumericPattern`. Without
one it asks `lang.Factory.getAlphanumeric` with the factory's `languageCode` field, and a factory
made for a loaded model leaves that field null (only `getLanguageCode()` reads the language from
the model). A loaded model without the property therefore always gets `^[A-Za-z0-9]+$`, never its
language's pattern; this was checked on the JVM with a German model whose pattern was removed.
`ContextRuns` takes the factory's pattern with `TokenizerME`'s fallback to
`Factory.DEFAULT_ALPHANUMERIC`. Each UD model's manifest holds the pattern of its language, so with
the optimization on, French accepts `caf\u00E9` as one alphanumeric token and English does not.

## Provenance

- OpenNLP: the 3.0.0-SNAPSHOT jars built from the OpenNLP checkout at commit
  `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` (recorded in each golden file's `opennlp:`
  line). The class path is the runtime, API and ML (commons, maxent, perceptron, bayes)
  jars of that build plus `slf4j-api` 2.0.19.
- JDK: OpenJDK 21.0.12.1 (the `jdk:` line of each file).
- Models: Apache OpenNLP's Universal Dependencies 1.3 models, trained with OpenNLP 2.5.4
  (Spanish, French, German, English and Portuguese tokenizers; German, English and
  Portuguese sentence models). Each golden file records the model name and SHA-256. The
  models are not in the repository. They are downloaded from
  `https://downloads.apache.org/opennlp/models/ud-models-1.3/<file>`, and their SHA-512
  is the one published beside each file as `<file>.sha512`:

  | File | SHA-512 |
  |---|---|
  | `opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin` | `c811604c24cdbe490ab42dcffdbae63f9a74b39622f21b7b31f63eea566ffe35ef871f0da905e61180656b877cf0be4d626588c6312fcc18df5ab9a35b4183fa` |
  | `opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin` | `343c9fe93ff301b736f30fa2e0d954b270994a1535ec2bd424553cbd82b13eb0d86aa2042946db62322430a8b5a8f29a8a2d882f107cb26c6d4f6d494d97debf` |
  | `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin` | `fed5d53ea87bf66afeb08c084f89ca9b95a96de0920dca40487b2ab9972ee6b7ef211e477db480b7253431de079fd66dfd14329cd7d19ad1f565155eb0cf5191` |
  | `opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin` | `147d1cdcb89ca5d9bd0ead8fddf2cc14ae543a9151803fe56b3c1fa152b986605e34fb6dda4819c68aa22fc9bffa6bfab14f916b7d0bf4892d452686af34778a` |
  | `opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin` | `ba5280ca74b190515f630ed2b2257a758e8baa876a8da3c02735f81a5543c37f7bc57614fd1ecb2850b4a2df98b2d2e452718de43f676f9b63b064206f86c467` |
  | `opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin` | `18e56e96eae9741935c5f54f16d3c6933e151ec550b882d2e5e1194ac02c6b1e56063843ff3805a35cf44810091674ef6bf2f080d1bc761770d8b09ee9a01277` |
  | `opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin` | `0c0390b36530ce9807e9f3d69cfd09524d965506dbb6e2aab2f34556c8858903d538433b828e1f6ed7eaca92f336a04be83461a45b28b177d01ba372fafa35f8` |
  | `opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin` | `6ff66aec4835e9d5caeeb50a2e70bd673840ef6219fe1dc4018c8f9999b21b959c52fdac68844a0934bbcd77346743627f728fb9c99b7ca468e795fc5291c3bb` |

  The English entries are the ones `analysis/goldens-tool/src/recipe.rs` pins.

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
| `Variants.java` | Entry point of the variant runs: abbreviations, surrogate halves, alphanumeric optimization, detokenizer |
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
