# Sentence detection and tokenization

The segmentation crates write the first two layers of a document: the
sentences (`opennlp:sentences`) and the word tokens (`opennlp:tokens`).
They are ports of the OpenNLP 3.x sentence detectors and tokenizers onto
the analysis API (docs/analysis.md), reading OpenNLP's own models. A
model-free tokenizer gives tokens before any model is loaded, the
UAX #29 tokenizer handles text in any script, the dictionary word
tokenizer adds the words of Thai, Lao, Khmer and Burmese (which are
written without spaces), and a rule sentence splitter gives sentences in
languages that have no sentence model.

OpenNLP is the reference. Where the Rust side matches it, a golden file
made by the JVM proves it. Where it differs on purpose, this page says
where and why.

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-unicode` | `analysis/unicode/` | Character data: Java's character classes, the Unicode `White_Space` set, `Word_Break`, `Sentence_Break` and `Extended_Pictographic` properties, extended grapheme cluster boundaries, `Line_Break=SA`, a UTF-16 code unit cursor over UTF-8 text |
| `turbo-jvm` | `analysis/jvm/` | JDK routines ported bit for bit; `turbo-maxent` uses `strict_math::exp` and `strict_math::log`, the fdlibm functions behind `StrictMath.exp` and `StrictMath.log`; `math::log` is a correctly rounded logarithm, which matched `Math.log` of the JVM the goldens are made with on the integers compared (its documentation says which) |
| `turbo-maxent` | `analysis/maxent/` | OpenNLP model files: the zip container, the manifest, maxent and perceptron models and their evaluation, the dictionary XML |
| `turbo-sentdetect` | `analysis/sentdetect/` | The newline detector, the paragraph detector, the maxent sentence detector and the rule sentence splitter (UAX #29 with per-language tailorings) |
| `turbo-tokenize` | `analysis/tokenize/` | The simple, whitespace and UAX #29 word tokenizers, the dictionary word tokenizer for Thai, Lao, Khmer and Burmese, the maxent tokenizer, the unigram frequency segmenter for Chinese, the MeCab dictionary lattice tokenizer for Japanese and Korean, and the dictionary detokenizer |
| `turbo-tokenize-layers` | `analysis/tokenize-layers/` | The tokenizers' own layer: the word type of each token |

They depend on the standard library, `turbo-analysis` and
`turbo-analysis-layers`, and on each other in that order. Nothing here
depends on the `turbo` core crate: the embedding model's subword
tokenizer is a separate annotator (`SubwordAnnotator` in
`turbo-analysis-embed`), so a program that only segments text does not
link the engine.

## Character classes and Java parity

OpenNLP's algorithms run on Java `char` values, which are UTF-16 code
units, and classify them with `java.lang.Character`. To produce the same
features and the same splits, the ports walk the text as code units too.
`turbo-unicode` provides:

- `Utf16Cursor`: steps over a `&str` one UTF-16 code unit at a time
  without copying the text. A character outside the Basic Multilingual
  Plane is two steps, a high surrogate then a low one; each step knows
  the byte offset of the character it belongs to. The cursor is the
  only place where "the character before position p" is defined, so
  every port agrees on it.
- Java's classes on a code unit or a code point: `is_letter` (general
  category `L*`), `is_digit` (`Nd` only), `is_upper_case` (`Lu` and
  `Other_Uppercase`), `is_alphabetic`, `is_whitespace` (Java's own
  definition), `is_space_separator` (`Zs`), and `to_lower_case` (the
  simple one-to-one mapping). A surrogate half is in no class, as in
  Java. Rust's `char::is_numeric` and `char::is_alphabetic` are wider
  than Java's `isDigit` and `isLetter`, so they are not used.
- The ports call these through `java::code_point`, which takes a `u32`
  so that a lone surrogate code unit is classified the way Java
  classifies a `char`. The Unicode version of the tables and the code
  points on which they answer differently from JDK 21, the JDK the
  golden files are made with, are in docs/normalizers.md ("Differences
  from the JDK").

### Whitespace

OpenNLP has two definitions of whitespace, chosen by a global
`WhitespaceMode`: `Unicode`, the default since 3.0, is the Unicode
`White_Space` set (25 code points, `U+0085` included, `U+001C` to
`U+001F` excluded); `Legacy` is `Character.isWhitespace` or general
category `Zs`. The mode changes tokens, trimmed spans and model
features, so it is a parameter of every annotator here, never a global,
and a golden file records the mode it was made with. The default is
`Unicode`.

### Model features

A maxent model's predicates are Java strings read from the model file
in Java's modified UTF-8, which can hold a lone surrogate. The ports
generate features as modified UTF-8 bytes into one reusable buffer, and
the model's predicate map is keyed by those bytes, so a feature built
around a character outside the BMP matches exactly what the JVM looked
up. For text inside the BMP, modified UTF-8 is UTF-8.

## Model files

An OpenNLP `.bin` model is a zip holding `manifest.properties` (a Java
properties file), one `.model` entry (the classifier), and optional
artifacts such as `abbreviations.dictionary` (the dictionary XML).
`turbo-maxent` reads the zip (stored and deflated entries, with its own
inflate), the properties file as `java.util.Properties` does (escapes,
continuation lines, `\uXXXX`), and the model:

| Type string | Trainer | Evaluation |
|---|---|---|
| `GIS` | GIS maxent | sum the parameters of the context's predicates onto a uniform prior (`StrictMath.log`), `StrictMath.exp`, normalize |
| `QN` | L-BFGS maxent | sum the parameters, then `StrictMath.exp(x - logsumexp)` |
| `Perceptron` | perceptron | sum the parameters, divide by the largest magnitude when it is above 1, `StrictMath.exp`, normalize |

A model's parameters are stored by predicate as (outcome, weight)
pairs. `MaxentModel::eval_modified_utf8` takes the context's features as
modified UTF-8 bytes and writes one probability per outcome into a
caller's `[f64]`; it allocates nothing. `MaxentModel::eval` takes `&str`
features and encodes the few that differ in modified UTF-8.
Probabilities are `f64` end to end, as they are in Java, and `exp` and
`log` are `turbo-jvm`'s ports of fdlibm, which `StrictMath` specifies, so
the fixture models' probabilities equal the JVM's bit for bit.
`NaiveBayes` models are refused with an error that names the type;
nothing here needs them.

The dictionary XML (`<dictionary case_sensitive="..."> <entry
attr="..."> <token>...</token> </entry>`) is read by a small reader
written for exactly that shape. A case-insensitive dictionary lowercases
each entry per code point with Java's `toLowerCase`, as `StringList`
does, and `contains` lowercases the probe the same way.

## Sentence detection

### `NewlineDetector`

Reads a `View`, writes `Sentences`. A sentence is each run between `\n`
or `\r` characters (not `U+2028`, `U+2029` or `U+0085`), trimmed of
whitespace in the configured mode, dropped if nothing remains. This is
`NewlineSentenceDetector` exactly, including `\r\n`, which it treats as
two separators with an empty piece between them that is dropped;
`str::lines` is not used because it does not split on a lone `\r`. The
separators are ASCII, so the scan is over bytes, eight at a time, and
whitespace is decoded only where a byte is not ASCII. The function
`newline_spans` gives the same spans outside a pipeline. The goldens of
the runs `newline` and `newline+legacy` are matched exactly, in UTF-8
and UTF-16 offsets.

### `ParagraphDetector`

Reads a `View`, writes `Structure` blocks of kind `Paragraph`, each
top level (no parent, depth 0). A line ends at `\n`, `\r\n` or a lone
`\r`, the line breaks `NewlineDetector` sees, so the two detectors agree
on where lines are and a CR LF file is not one paragraph per line. A
line is blank when it holds only whitespace in the configured mode. A
paragraph is a maximal run of non-blank lines; its span starts at the
first non-whitespace character of its first line and ends at the last
non-whitespace character of its last line, so a paragraph holds its
inner line breaks but no leading or trailing ones. The function
`paragraph_spans` gives the same spans outside a pipeline. OpenNLP has
three slightly different paragraph definitions (an exactly empty line in
`ParagraphStream`, a line that is empty after `String.trim` in
`EmptyLinePreprocessorStream`, Unicode line breaks in the whitespace
normalizer); this is the one `turbo:structure` uses.

### `SentenceDetector` (maxent)

Reads a `View`, writes `Sentences` and `Confidence<Sentences>`. It is
`SentenceDetectorME`, reading a `SentenceModel` (`SentenceModel::open`
checks the component, the factory, the `useTokenEnd` property and the
outcomes `s` and `n`, as OpenNLP does):

1. The end-of-sentence scanner lists every position holding one of the
   model's end characters, the manifest's `eosCharacters`. A model
   without the property (or with an empty one) gets `. ! ?` whatever its
   language: `SentenceDetectorFactory` looks up the language default of
   `lang/Factory` with a language code that is never set on a factory
   loaded from a file, so the Portuguese default never applies to a
   loaded model, and a Thai model gets the default context generator, not
   the Thai one. Both were checked on the JVM with manifests stripped of
   the property. Every model OpenNLP's factory writes has it; the
   Portuguese UD model lists `. ? ! ; : ( ) « » ' "`. A set with a
   character outside the BMP is refused.
2. Leading parts of a run of end characters are skipped (`z.B.` is one
   candidate), unless the character after one is upper case.
3. For each candidate, `DefaultSDContextGenerator` builds the features:
   `sp` and `sn` for whitespace on either side, `eos=` the character,
   the prefix, previous, suffix and next words with the prefix length,
   capitalization and abbreviation flags. Word boundaries here are
   whitespace in the configured mode, and `String.trim` semantics
   (every code unit at or below `U+0020`) apply to the words. The words
   are written per code unit as modified UTF-8, so a suffix that ends
   between the halves of a character outside the BMP is looked up with
   its lone surrogate, as the JVM looks it up. A text that is all ASCII
   (its UTF-16 length equals its byte length, which the scanner counts
   anyway) is walked by byte; any other text through `Utf16Cursor`.
   The searches for the whitespace before and after a candidate remember
   what earlier candidates of the same text found, and a word feature
   longer than every predicate of the model is not built (it could not
   change the probabilities), nor is a word longer than every dictionary
   entry looked up. Without both, a text with no whitespace takes time
   quadratic in its length, since every word reaches to an end of the
   text. `detect_traced` builds every feature, as the Java does.
4. The model decides `s` or `n`. A split is then checked against the
   abbreviation dictionary (each entry's first token, lowercased per code
   point unless the dictionary is case-sensitive). An entry counts when it
   holds the candidate, ends on the code unit just before it, or starts
   on it, and starts no earlier than the start of the current segment.
   The split is refused when such an entry starts at the segment start
   and ends exactly at the candidate, or starts right after whitespace
   (`Character.isWhitespace`), an apostrophe or `(`; an entry starting at
   the segment start but not ending at the candidate is judged by its own
   first character instead. So with an entry `et al.`, indexed as `et`,
   "We x et. Then" is not split at the `.`.
5. Sentence starts are placed after the end character, or after the
   following whitespace run when `useTokenEnd` is set. Spans are
   trimmed of Unicode `White_Space` (this step and the skip in step 2
   always use the Unicode set, as OpenNLP does since OPENNLP-205) and
   empty ones are dropped with their probability. The text after the last
   start is a last sentence with probability 1.

The sentence detector works on code units for the features and on code
points for the spans, as the Java does. Its spans are UTF-8 byte
offsets on character boundaries by construction: an end character is
always a whole character. `SentenceDetector::detect` gives the spans and
probabilities outside a pipeline, and `detect_traced` also hands every
candidate's features, probabilities and decision to a callback. The
probabilities are `f64`, and the `Confidence` layer stores them as they
are.

The annotator's confidence output is aligned to the sentences it writes
in the same step, which the analyzer accepts, so the detector is added
with a plain `Analyzer::builder().add(..)`.

The model's `ModelRef` is `opennlp:SentenceDetectorME:<language>`, the
OpenNLP version in the manifest as the revision, and the SHA-256 of the
model file.

The goldens are the runs of the three published UD models
(`analysis/goldens/data/sentdetect/opennlp-<language>-...`). The models
are trained on CC BY-SA treebanks and not checked in;
`turbo-goldens fetch-models` downloads them and the tests read them from
its directory (`analysis/sentdetect/tests/models/README.md`). The
published models have no abbreviation dictionary, so two small English
models trained on the JVM (`en-abbrev.bin` and `en-plain.bin` in
`analysis/goldens/data/fixtures`; the training sentences, the Java
source and the commands are in `analysis/sentdetect/tests/fixtures`)
cover the dictionary, `useTokenEnd=false` and both whitespace modes,
with every context of their dumps compared feature by feature.

### `RuleSentenceDetector` (rules)

Reads a `View`, writes `Sentences`. A sentence splitter that needs no
model, for the many languages OpenNLP has no sentence model for, and for
text the published models were not trained on. Its annotator id is
`turbo-sentdetect:rules`; the configuration hash covers the language
whose rules it resolved to, the tailorings switched on and the line
break mode, so `de-AT` and `de` share one. `RuleSplitter` gives the
same spans outside a pipeline (`split`, and `split_with` with a
`RuleScratch` that a caller reuses so that repeated calls allocate
nothing once its buffers have grown).

It is the UAX #29 sentence algorithm (Unicode 18.0) with tailorings in
the manner of Wikimedia's sentencex: conservative, aware of quotations
and brackets, with list items and numbered references kept whole. Its
steps, each but the first of which `Rules` can switch off:

1. Every character gets its `Sentence_Break` value
   (`turbo_unicode::sentence_break`, generated from
   `SentenceBreakProperty.txt`). The language can make more characters
   end a sentence or fewer (below), and U+2026 HORIZONTAL ELLIPSIS ends
   one as a full stop does. The full stops of an ellipsis in brackets
   (`[...]`, `[…]`, `(...)`), which marks words left out of a quotation,
   end none (`terminators`).
2. Line breaks (`LineBreaks`). With `Paragraph`, the default, a line
   break inside a paragraph is a space, and only a blank line (a line
   holding only spaces) or U+2029 separates paragraphs; text wrapped at
   a fixed width then keeps its sentences whole. With `Sentence` every
   line break ends a sentence, as UAX #29 rule SB4 has it.
3. The rules SB1 to SB998 place the boundaries, in one pass: the
   `SATerm Close* Sp*` pattern that SB8 to SB11 share is tracked as a
   state, and the look-ahead of SB8 remembers where its last scan
   stopped. `RuleSplitter::uax29()` (no tailoring, `LineBreaks::Sentence`)
   places every break of Unicode's `SentenceBreakTest.txt` exactly.
4. Numbered references after a terminator stay with their sentence:
   `It is big.[1][2] Next` and `It is big. [3] Next` break before `Next`
   (`references`).
5. A lower case word after `!` or `?` continues the sentence (`Yahoo! in
   the department`, `"Why?" she asked`), as one after a full stop already
   does by SB8 (`continuation`).
6. A full stop after one of the language's abbreviations ends no
   sentence (`abbreviations`), nor does one after a run of initials
   before a capitalized word (`J. Smith`, `Л.Н. Толстой`, `U.S.
   Government`; `initials`). The CLDR suppressions match case-sensitively
   with their full stop, as ICU matches them; the sentencex lists match
   the word lowercased. Only the 64 characters before the full stop are
   looked at, so a text without spaces stays linear; no abbreviation is
   that long. If the next word is one of the language's sentence
   starters (English only: `The`, `It`, `He`, ...), the full stop ends
   the sentence after all (`Pitt, Briggs & Co. It closed`), unless the
   abbreviation is a single lower case letter (`p. The`). In German a
   number of one to three digits with a full stop before a capital
   letter, a digit or a dash is an ordinal (`am 12. Juni`, `1. -3.
   Jahrhundert`) and ends no sentence.
7. List item markers start sentences (`lists`): numbers and single ASCII
   letters with `.`, `)`, `.)` or in parentheses, and bullets. A bullet
   counts at the start of a line, or inside a line when its paragraph
   holds another of the same bullet; a number or letter right after a
   counted bullet counts with it. A marker with `)`, `.)` or
   parentheses counts at the start of a line, or inside a line next to
   the previous or next marker of its sequence. A number or letter with
   a bare full stop is also prose (`plan A. Then plan B.`, `I was 7.
   Then I turned 8.`, `Der 1. und 2. Platz`, `John A. Smith and Mary B.
   Jones`), so it counts only as part of a list: at the start of a line
   unless a single letter with a full stop or the end of the text
   follows (`J. K. Rowling`, `x. A`); inside a line when it continues
   by one a counted marker of its sequence in the same paragraph (`1. The
   first item 2. The second item`), or when it is a `1.` or `a.` with
   its `2.` or `b.` later in the paragraph and it follows a sentence end,
   a colon or an earlier counted marker of the paragraph (`Finally,
   this. 1. The first item 2. The second item`). A full stop inside a
   marker ends nothing.
   `-`, `*`, `+` and `–` are bullets only at the start of a line.
8. No boundary falls between an opening bracket or quotation mark and
   the mark that closes it (`He said "Stop. Go." Then` is two
   sentences; `brackets`). A mark that also serves as an apostrophe
   (`'`, `’`) opens only before a word and closes only after one, and
   counts as neither inside a word. A closer that matches none of the
   eight innermost open marks is ignored, and a paragraph break closes
   everything, so a stray quotation mark cannot join paragraphs.

The spans are then trimmed of Unicode `White_Space` and empty ones
dropped, as the other detectors' spans are, and are UTF-8 byte offsets
on character boundaries.

**Languages.** `rules::LANGUAGES` lists 304 languages by BCP 47 code,
with their English names. Seventeen have rules of their own: `am ar bg
da de el en es fr hy it kk my pl pt ru uk`. Every other language names
a fallback chain, the languages whose rules suit its text best, in
order: a language of the same script and region whose abbreviations
appear in its texts (Russian for the Cyrillic-script languages of
Russia, French for the languages of francophone Africa), or one it is a
variety of (Swiss German, Western Armenian). Chains are at most three
steps; 186 languages reach rules that way or have their own, and the
rest get the root rules: the UAX #29 rules with every tailoring but no
abbreviations. A tag is matched by its longest known prefix (`de-AT`
uses `de`, `zh-Hant-TW` uses `zh`), case and `_` do not matter, and a
tag the table does not know gets the root rules. The table and the
chains were written for this project.

| Language | Terminators | Abbreviations | Other |
|---|---|---|---|
| de, en, es, fr, it, pt, ru | Unicode | CLDR 48.2 suppressions | en: sentencex sentence starters; de: ordinal full stops |
| am, ar, bg, da, kk, pl, uk | Unicode | sentencex | |
| el | also `;` and U+037E GREEK QUESTION MARK | none | |
| hy | also `:` (Armenian full stop); `.` and U+2024 end none | none | |
| my | also U+104F | none | |

The data is under `analysis/sentdetect/data/rules` with its licences
and a README naming the sources. It is compiled into Rust tables
(`analysis/sentdetect/src/rules/tables.rs`, generated by the test `analysis/sentdetect/tests/rules_tables.rs`,
which fails when the committed tables differ from the data); nothing is
read at run time. sentencex's abbreviation lists for `bn de el en es fi
fr gu hi it kn ml nl pa pt ru sk ta te` are not used: they share much
of their content with the Moses nonbreaking prefix lists, which are
LGPL. LanguageTool's `segment.srx` (LGPL) and MediaWiki's language
fallbacks (GPL) are not used either.

**Where it differs from its references.** Each difference has a test
in `analysis/sentdetect/src/rules/tests.rs` or the module that makes it, and shows in the
conformance results below.

- From UAX #29: `LineBreaks::Paragraph` is the default, so a single line
  break is not SB4's break (`line_breaks`; six lines of
  `SentenceBreakTest.txt` break at a single LF and fail for the rules
  detector). A lower case word after `!` or `?` continues the sentence,
  where SB11 breaks (`continuation_after_exclamation_and_question`; two
  lines of the test). The other tailorings remove UAX breaks, move them
  past references, or add them at list markers and after a language's
  own terminators; on the other 504 lines of the test the rules detector
  agrees with UAX #29.
- U+2026 ends a sentence (`language_terminators`); in UAX #29 it is
  `Other`. A bracketed ellipsis ends none (`bracketed_ellipsis`).
- From sentencex: the abbreviation data is CLDR's for the seven
  languages above instead of sentencex's own, so an abbreviation CLDR
  does not list (`p.`, `St.` in English) ends a sentence here.
  Brackets protect what they enclose up to a paragraph break, with no
  length limit. List markers need a line start, a bullet or a sequence
  that reaches back to one, or a sentence end before a `1.` or `a.`.
  A single newline is a space by default.
- From ICU's suppressions: they are applied to the word before any full
  stop that UAX #29 breaks after, and a sentence starter after the
  abbreviation restores the break in English; ICU has no starters.
- Without a fallback model, Thai, Lao, Khmer and other scripts written
  without sentence-ending punctuation get one sentence per paragraph:
  UAX #29 has nothing to break at, and there is no character rule that
  finds sentence ends in unpunctuated text. When an
  `UnpunctuatedSplitter` fallback (`analysis/sentdetect/src/rules/fallback.rs`)
  such as `turbo_sat::SatAnnotator` (`analysis/sat/src/fallback.rs`,
  `rule_splitter_with_sat`, `rule_detector_with_sat`) is attached via
  `RuleSplitter::with_unpunctuated_fallback` or
  `RuleSentenceDetector::with_unpunctuated_fallback`, the rule splitter
  first runs UAX #29 and language tailorings, then routes any candidate
  sentence span that contains `Thai`, `Lao` or `Khmer` characters and is
  not already terminated by a Khmer full stop (`U+17D4` KHAN or `U+17D5`
  BARIYOOSAN) through the SaT fallback in a single batch across the
  document or batch of documents, validating, trimming and shifting the
  resulting sub-spans back into document byte offsets.

**Unpunctuated script evaluation (`analysis/sat/tests/unpunctuated_eval.rs`).**
Evaluated on 200 five-sentence paragraphs (1,000 gold sentences, 800
internal sentence boundaries per corpus) joined by single spaces with
`split_on_newlines: false` against `segment-any-text/sat-1l-sm` (commit
`00a783439c201b20ee91ee1624dc817f1bf0e578`): UD Thai PUD (`th_pud`,
commit `d3735d6b7cfb2fb2bf9aa2ca5fb9ffd5fb09b011`), Khmer `khPOS` 1.0
(commit `87a6dc83c2beee822dd8ff33ac1c0b9fd7c53eb6`, 1,000 `KAN`-terminated
sentences evaluated both unpunctuated with `U+17D4`/`U+17D5` stripped and
as punctuated in `khPOS`), and Lao POS (`SeqLabeling` `Lao_train.txt`,
commit `ac2e2aa1dd6cf2eb2a3eb9d560cdbe9da8449989`):

| Corpus | Detector | Gold / Pred / Match Spans | Span P | Span R | Span F1 | Gold / Pred / Match Boundaries | Boundary P | Boundary R | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Thai (`th_pud`) | `rules` | 1000 / 209 / 3 | 0.014 | 0.003 | 0.005 | 800 / 9 / 4 | 0.444 | 0.005 | 0.010 |
| Thai (`th_pud`) | `rules + sat-1l-sm` (`t = 0.25`) | 1000 / 1134 / 437 | 0.385 | 0.437 | **0.410** | 800 / 934 / 590 | 0.632 | 0.738 | **0.681** |
| Lao (`lao_pos`) | `rules` | 1000 / 536 / 272 | 0.507 | 0.272 | 0.354 | 800 / 336 / 333 | 0.991 | 0.416 | 0.586 |
| Lao (`lao_pos`) | `rules + sat-1l-sm` (`t = 0.25`) | 1000 / 957 / 655 | 0.684 | 0.655 | **0.669** | 800 / 757 / 652 | 0.861 | 0.815 | **0.838** |
| Khmer unpunctuated (`khpos_unpunc`) | `rules` | 1000 / 265 / 2 | 0.008 | 0.002 | 0.003 | 800 / 65 / 3 | 0.046 | 0.004 | 0.007 |
| Khmer unpunctuated (`khpos_unpunc`) | `rules + sat-1l-sm` (`t = 0.25`) | 1000 / 430 / 8 | 0.019 | 0.008 | **0.011** | 800 / 230 / 16 | 0.070 | 0.020 | **0.031** |
| Khmer unpunctuated (`khpos_unpunc`) | `rules + sat-1l-sm` (`t = 0.025`) | 1000 / 1195 / 27 | 0.023 | 0.027 | **0.025** | 800 / 995 / 119 | 0.120 | 0.149 | **0.133** |
| Khmer punctuated (`khpos_raw`) | `rules` | 1000 / 1097 / 907 | 0.827 | 0.907 | 0.865 | 800 / 897 / 777 | 0.866 | 0.971 | 0.916 |
| Khmer punctuated (`khpos_raw`) | `rules + sat-1l-sm` (`t = 0.25`) | 1000 / 1105 / 907 | 0.821 | 0.907 | **0.862** | 800 / 905 / 777 | 0.859 | 0.971 | **0.911** |

**Conformance.** `analysis/sentdetect/tests/conformance.rs` runs three detectors over three
suites stored as typed test data in
`analysis/goldens/data/conformance/sentences` (its README names the
sources, commits and licences): the Pragmatic Segmenter specs (MIT; 585
cases in 23 languages, the first 52 English cases being its Golden
Rules), the pySBD tests (MIT; 471 cases) and `SentenceBreakTest.txt`
(Unicode 18.0; 512 lines). A text case passes when the detector's
sentences equal the expected ones once runs of whitespace are made one
space; a break test line passes when the spans equal the UAX segments
trimmed of whitespace. The detectors are `rules` (every tailoring, the
case's language, with the bundled abbreviation inventory), `uax29` (no
tailoring) and the maxent detector with the published UD model of each
language that has one, in JVM parity mode, on the cases in that
language; `maxent-en-inventory` is the English model with its default
inventory. The CI step "sentence conformance and agreement, models
required" prints the pass counts. Every failing case is listed with what differs in
`analysis/sentdetect/tests/conformance/<detector>.<suite>.failures`, and for `rules` with
the tailoring without which it would pass; the test fails when the
lists change. 34 text cases expect sentences that the reference's
cleaning step rewrote, and fail for every detector that runs them.

| Detector | Pragmatic Segmenter | Golden Rules (English) | pySBD | `SentenceBreakTest.txt` |
|---|---|---|---|---|
| `rules` | 437 of 585 | 46 of 52 | 377 of 471 | 504 of 512 |
| `uax29` | 305 of 585 | 29 of 52 | 264 of 471 | 512 of 512 |
| maxent, English UD model, parity | 133 of 289 (English cases) | 31 of 52 | 98 of 193 (English) | 467 of 512 |
| maxent, English UD model, default | 155 of 289 (English cases) | 36 of 52 | 112 of 193 (English) | |
| maxent, German UD model | 22 of 37 (German cases) | | 20 of 35 (German) | |

The Portuguese model has no case in its language. Among the rules
detector's failures are abbreviations it has no data for (there is no
list for Dutch or Slovak, and CLDR's lists are short), a sentence that
starts with a lower case letter after a full stop (the Greek cases,
which SB8 keeps together), `:` as a sentence end in Arabic, and the
references' document cleaning. By language, the rules detector passes
all cases of `am bg fa fr hi hy mr my pl ur zh`, and 342 of 482 English,
61 of 72 German, 86 of 109 Danish, 78 of 84 Russian, 60 of 72 Italian
and 54 of 72 Spanish cases, against 234, 29, 63, 72, 56 and 41 for
plain UAX #29.

**Agreement with the maxent detector.** `analysis/sentdetect/tests/agreement.rs` runs the
rule splitter and the maxent detector with the published model, both
as they run by default (with the bundled abbreviation inventory), on the
same texts, for the three languages with a model: the golden inputs in
that language and the texts of the conformance cases in it (not their
expected sentences). Boundaries are the starts of the sentences after
the first; precision is the share of the rule splitter's boundaries
that the model also places, recall the share of the model's boundaries
that the rule splitter places.

| Language, texts | Texts | Split alike | Sentences (maxent, rules, same span) | Boundaries (maxent, rules, same) | Precision | Recall |
|---|---|---|---|---|---|---|
| English, golden inputs | 45 | 33 | 121, 119, 99 | 78, 76, 69 | 90.8% | 88.5% |
| English, conformance texts | 482 | 292 | 3180, 2649, 1803 | 2698, 2167, 1934 | 89.2% | 71.7% |
| German, golden inputs | 2 | 1 | 7, 6, 5 | 5, 4, 4 | 100% | 80.0% |
| German, conformance texts | 72 | 42 | 194, 179, 100 | 122, 107, 76 | 71.0% | 62.3% |
| Portuguese, golden inputs | 1 | 1 | 5, 5, 5 | 4, 4, 4 | 100% | 100% |

The maxent detector places more boundaries than the rule splitter on
every set but the Portuguese one. The numbers are asserted by the
test.

Both tests need the models for the maxent rows and skip those rows
without them; the CI job fetches the models and runs both with
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, so that there a missing model
fails.

### Abbreviation inventories

`AbbreviationInventory` (`analysis/sentdetect/src/inventory/`) is a
classified list of abbreviations that the maxent detector and the rule
splitter consult at a full stop, on top of a model's dictionary or a
language's CLDR suppressions. It is on by default:
`SentenceDetector::new` takes the bundled inventory of the model's
language and `RuleSplitter::new` (and so `RuleSentenceDetector::new`)
that of its tag, through `AbbreviationInventories::for_language`. Only
English has a bundled inventory; every other language gets
`AbbreviationInventory::EMPTY`, and splits as before. `RuleSplitter::uax29`
has none. `with_abbreviations` sets another inventory.

**JVM parity mode.** `SentenceDetector::with_jvm_parity()` drops the
inventory (it is `with_abbreviations(AbbreviationInventory::EMPTY)`), and
the detector's spans, probabilities and `config_hash` are then those of
OpenNLP's `SentenceDetectorME` bit for bit, as before inventories. The
JVM golden comparisons (`analysis/sentdetect/tests/golden.rs`, the fuzz
runs against the goldens, the sentence counts of
`analysis/train/tests/roundtrip.rs`), the `maxent-<language>`
conformance runs and the `turbo-train-recipes` model gate, which
compares models and not the inventory, use it. `RuleSplitter::without_inventory()` is the same
for the rule splitter.

The bundled English inventory
(`analysis/sentdetect/src/inventory/english.rs`) has four classes,
composed with `|` or named by an `AbbreviationPreset`
(`Minimal`, `Standard`, `Scholarly`, `Full`):

| Class | Examples |
|---|---|
| `General` | `etc.`, `vs.`, `e.g.`, `i.e.`, `fig.`, `a.m.`, months |
| `Titles` | `Mr.`, `Mrs.`, `Dr.`, `Prof.`, `Sen.`, `Rep.`, `Gen.`, `Jr.` |
| `Academic` | `et al.`, `op. cit.`, `ib.`, `cf.`, `Ph.D.`, `vol.`, `no.`, `pp.` |
| `LegalTechnical` | `Inc.`, `Corp.`, `Ltd.`, `Co.`, `v.`, `Sec.`, `Art.`, `No.`, `U.S.C.`, `Fed. Reg.` |

Each entry has a context rule. A `Prepositive` entry (titles, `e.g.`,
`v.`) ends no sentence before a word, an initial or a number. A
`Numeric` entry (`fig.`, `vol.`, `No.`, `pp.`, `Sec.`) ends none before
a number or a canonical Roman numeral (not a lone `I`). A `Trailing`
entry (`etc.`, `Inc.`, `et al.`) ends none before a lower case word or a
number, and has no opinion before a capitalized word, so a sentence
that ends with it still ends. Any entry ends no sentence before `,`, `;`
or `:`, and a full stop inside an entry (`U.` of `U.S.C.`, `op.` of
`op. cit.`) ends none. An entry matches as written or with its first
letter in the other case (`Fig.` for `fig.`). A short list of words that
are never abbreviations (`no`, `ok`, `it`) allows a split that a
dictionary or a CLDR suppression would refuse. Users add entries with a
rule (`with_addition`) and remove entries or words (`with_exclusion`).

In the maxent detector the inventory acts where the model says split:
a suppression vetoes the split, an allowance takes it whatever the
model's dictionary says, and otherwise the dictionary decides. The
model's features and probabilities are not changed, since they must be
those it was trained with. In the rule splitter the inventory is part of
the abbreviation tailoring and is asked before the language's own
abbreviations, initials and sentence starters.

`config_hash` of both annotators folds in the inventory's hash: the
language, the classes, every active entry with its rule, the
never-abbreviation words, the additions and the exclusions. So the
default hash of an English detector changed with inventories, and the
hash in parity mode (or of a language without an inventory) is the hash
from before them; `analysis/sentdetect/tests/abbreviations.rs` checks
both. `RuleSentenceDetector::VERSION` is `3` since the default changed.
`turbo-eval` names a maxent sentencer with an inventory by the model and
the inventory's hash (`... abbreviations <hash>`), so its records tell
the two modes apart.

On the conformance suites the inventory adds 8 Pragmatic Segmenter and
5 pySBD passes to the rule splitter and 22 and 14 to the English maxent
model, and fails no case that passed without it. The fixed cases are
abbreviations before a name, a number or a lower case word (`St.`, `Mt.`,
`e.g.`, `p.`, `no.`, `co.`, `Ph.D.`, `a.m.`, `Jan. 20, former Sen.`).

## Tokenization

Every tokenizer reads a `View` and an optional `Sentences` layer. With
sentences, each sentence is tokenized on its own and the `Tokens` layer
carries the sentence of every token; without, the whole view is one
unit. All write `Tokens`; the maxent tokenizer also writes
`Confidence<Tokens>`, and the UAX #29 tokenizer also writes `WordTypes`.
One function, `tokenize_units_with`, turns the view and the optional
sentences into units, collects the tokens and hands each token's value
(the maxent tokenizer's probability) to the caller in token order, so
every tokenizer handles sentences the same way. It is generic over what
a tokenizer reports per token: a rule tokenizer reports a bare `Span`,
whose value is `()`, and calls it through `tokenize_units`, so its loop
moves nothing but spans. The simple, whitespace and maxent tokenizers
take the same `TokenizerOptions` (the whitespace mode and
`keepNewLines`), with one configuration hash. Each rule tokenizer is
also a plain function over a `&str` that appends spans to a caller's
buffer (`simple_spans`, `whitespace_spans`, `word_tokens`), and
`TokenizerMe::tokenize` appends spans with their probabilities.

### `SimpleTokenizer`

`SimpleTokenizer`: a token is a run of one character class (whitespace,
alphabetic, digit, other), and a run of "other" characters splits
wherever the character changes, so `...` is one token and `.,` is two.
Deviation: Java classifies each surrogate half on its own, so a
character outside the BMP (an emoji, a rare CJK character) becomes two
"other" tokens that each cover half a character. Half a character is
not a valid UTF-8 span, so here such a character is one "other" token
covering the whole character. It is never merged with a neighbour, not
even an identical character, because in Java no half ever joins the
token before or after it; so the token's edges are exactly the outer
edges of the Java halves. This is noted as an OpenNLP finding.

The `keepNewLines` option emits each `\n` and `\r` as a token of its
own. (The Java compares the character with the general category
constants `LINE_SEPARATOR` and `LETTER_NUMBER`, whose values are 13 and
10, so it tests for CR and LF by accident. The result is the same.)
Two more Java defects in this option are fixed rather than copied: the
Java throws when a line break comes before the first token, and it
reuses a stale start offset for a line break that follows a space, so
the span of that line break points at the wrong place. Here every `\n`
and `\r` is a token at its real position. The goldens record what Java
produced, and the tests for the affected inputs check against the
corrected positions and say so.

### `WhitespaceTokenizer`

A token is a run of non-whitespace in the configured mode, with the same
`keepNewLines` option. The scan steps over bytes through a 256-entry
table built for the mode: an ASCII byte is answered by the table alone,
and since every whitespace character above ASCII, in either mode,
starts with the byte 0xC2, 0xE1, 0xE2 or 0xE3, only those four bytes
make it decode a character and ask the mode.

### `WordTokenizer` (UAX #29)

Word boundaries by UAX #29 rules WB1 to WB999 over the Unicode 18.0
`Word_Break` property, with `Extended_Pictographic` for WB3c. OpenNLP
bundles the 17.0 property; docs/normalizers.md says which characters
differ. The segmenter is a state machine over the last two significant
classes, the current class and, where a rule needs it, the next
significant class and the regional indicator parity, so a document is
one pass with no backtracking. As in the Java, the decision for rules
WB5 to WB999 comes from a table over (last, current) pairs derived at
compile time from the rule cascade, and the cascade runs only for the
pairs whose answer depends on the look-ahead or the parity. Segments
that hold no letter, digit or script character (punctuation, spaces) are
not tokens. Each token's type is one of alphanumeric, numeric,
ideographic, hiragana, katakana, hangul, southeast asian or emoji, by
the script and classes of its characters, and goes to the `WordTypes`
layer. Tokens longer than `max_token_length` UTF-16 code units (255 by
default) are cut into pieces of at most that length, each with the type
of the whole word, never inside a character. The boundaries are checked
against Unicode's `WordBreakTest.txt`, every case.

`WordTokenizer` writes the `Tokens` first and then the `WordTypes`
aligned to them, and runs in a pipeline like any other step: the
analyzer accepts an aligned output whose anchor is an output the same
step writes before it, and the document checks that the types are as
long as the tokens when it stores them. `UntypedWordTokenizer` writes
the same tokens without the types, for a pipeline that has no use for
them.

### `DictionaryWordTokenizer` (Thai, Lao, Khmer, Burmese)

Thai, Lao, Khmer and Burmese are written without spaces between words,
and UAX #29 does not find the words inside a run of their letters. `DictionaryWordTokenizer` is the
UAX #29 word tokenizer with those runs handed to a per-script segmenter
(`Segmenter::Lstm` for Thai and `Segmenter::Dictionary` for Lao, Khmer
and Burmese), a port of ICU4X's complex-script word segmenters
(`icu_segmenter`, the `DictionaryBreakIterator` with its
`ZeroTrieSimpleAscii` word lists and the `LstmSegmenter` with its
bidirectional LSTM weights) with no dependency on the ICU4X crates. It
writes `Tokens` and `WordTypes` (a word of a run has the type southeast
asian), cuts long words as `WordTokenizer` does, and has the same
untyped variant, `UntypedDictionaryWordTokenizer`. `ComplexSegmenter` is
the segmenter under it, over a `&str`, with one `Segmenter` (`Dictionary`
or `Lstm`) per script (`ComplexSegmenter::with_segmenters` lets a caller
pick either segmenter per script, while `ComplexSegmenter::bundled` picks
the default for each script).

A complex run is a maximal run of characters of one script among Thai,
Lao, Khmer and Myanmar that have `Line_Break=SA`: the sets
`[:sc=Thai:]&[:lb=SA:]` and so on that ICU4X's data generator gives its
segmenter. Digits, punctuation and the other characters of those scripts
are not `SA` and stay with the UAX #29 rules. The text between runs is
segmented by the UAX #29 rules on its own, so the state machine starts
afresh after each run, as ICU4X's does. Inside a dictionary run the
dictionary is walked from each position, one character at a time: a word
that no longer word continues is taken at once; otherwise the longest
word seen that ends on a grapheme cluster boundary is taken when the
walk can go no further; where no word starts, the piece read until the
walk failed is one segment, or the rest of the run when the run ended
first. Inside an LSTM run the text is split into dictionary entries by
greedy longest-match over the model's `ZeroTrieSimpleAscii` vocabulary
(one Unicode scalar when no entry starts at that position, mapped to
the out-of-vocabulary row `V`); a forward and backward 1-layer LSTM
(`hun = 27`, `x = 40`, `c = 4`, IEEE-754 `f32` arithmetic in the same
order as ICU4X's 8-way unrolled dot product) scores each cluster as
`B`/`I`/`E`/`S`, and boundaries are emitted after each cluster where `E`
(`weights[2]`) strictly exceeds `B`, `I` and `S` (and at the end of the
run), snapped forward to the end of the containing extended grapheme
cluster if an entry ends mid-cluster. Because each cluster's embedding is
picked from a small table of `V + 1` vectors, `fw_b + fw_w * x[id]` and
`bw_b + bw_w * x[id]` are precomputed for every vocabulary ID when the
`.lstm` model is loaded, leaving only the `[4 * hun, hun]` recurrent
products per character step.

Data: the four word lists and four LSTM weight tables of ICU4X's baked
segmenter data at commit `bb3e1f9a0c966f7334aa400066f520f64f3dc3d6`,
generated with CLDR 49.0.0-BETA4 and Unicode 18.0.0 from ICU's export
data `release-79.1rc`. The CLDR release is a beta, but CLDR does not
supply these files: the dictionaries are ICU4C's
`brkitr/dictionaries/*dict.txt` (equal to ICU4C's lists at
`release-79.1rc` exactly, checked by an ignored test when given the
files), and the LSTM models are ICU4X's
`Thai_codepoints_exclusive_model4_heavy`,
`Lao_codepoints_exclusive_model4_heavy`,
`Khmer_codepoints_exclusive_model4_heavy` and
`Burmese_codepoints_exclusive_model4_heavy` (`CodePoints` models with
`hun = 27`). The published `icu_segmenter_data` 2.3.0 is older (CLDR
48.2.1) and is not used. Each `.dict` file in
`analysis/tokenize/data/icu4x-segmenter/` is the entry's alphabet (an
inversion list of at most 128 characters; a character's position in it
is the byte the trie steps on) and its trie bytes, unchanged, behind a
small header; the dictionaries are read in place from the
`include_bytes!` data, with no decoding and no allocation. Thai has
26383 words, Lao 30550, Khmer 81028 and Burmese 41120. Each `.lstm` file
(`TZL1` header) stores the model type, `V`, `x`, `hun`, the
`ZeroTrieSimpleAscii` bytes, and the nine little-endian `f32` arrays
(`mat1`..`mat9`, ~170 KB per script). The data are under the Unicode
License v3, with ICU's notices for the Thai and Khmer lists and BSD
style licences for the Lao and Burmese lists; `LICENSE` beside the data
has all of them, and the data README says how to rebuild the files from
ICU4X's source (`regenerate_from_icu4x` and `regenerate_lstm_from_icu4x`,
ignored, rebuild them byte for byte).

The proofs: every line of ICU4X's `WordBreakDictionary.txt` without CJK
(81 of 88; ICU4X segments Chinese and Japanese with a different
dictionary, which this tokenizer leaves to `UnigramSegmenter` and
`LatticeTokenizer`) gives the same boundaries byte for byte under
`bundled_dictionaries()`, and every non-CJK line of ICU4X's
`WordBreakLstm.txt` (77 of 84) gives the same boundaries byte for byte
under `bundled_lstms()` (`analysis/tokenize/tests/complex_icu4x.rs`);
every boundary of both segmenters is an extended grapheme cluster
boundary by the Unicode 18.0 rules, including GB9c (except where one
cluster spans runs of two scripts, deviation 3), in random texts of the
four scripts' characters and runs of dictionary words with letters
dropped, doubled or inserted
(`analysis/tokenize/tests/complex_graphemes.rs`). The grapheme cluster
boundaries are those of `turbo_unicode::grapheme`, checked against every
case of `GraphemeBreakTest.txt`.

Deviations from ICU4X, each with a test:

1. ICU4X takes a word that no longer word continues, the piece read
   where no word starts, and an LSTM `E` step without looking at
   grapheme clusters, so a boundary can fall between a letter and its
   marks (in Burmese and Khmer, also inside a conjunct joined by the
   virama, which GB9c now keeps together). Here such a boundary moves
   forward to the end of its cluster (in the dictionary, the next word
   is looked for from there; in the LSTM, any `E` inside the cluster
   snaps to the cluster end)
   (`a_boundary_never_splits_a_grapheme_cluster` and
   `a_word_inside_a_grapheme_cluster_is_not_taken` in
   `analysis/tokenize/src/complex/dictionary.rs`, and
   `lstm_boundary_inside_grapheme_cluster_snaps_to_cluster_end` in
   `analysis/tokenize/src/complex/lstm.rs`). Where ICU4X's boundary is
   on a cluster boundary, the two agree.
2. A run that starts with marks belonging to the cluster before it (a
   Thai vowel sign after a Latin letter) starts at the next cluster
   boundary, and the marks join the segment before: `a\u{0E31}ภาษา` gives
   `a\u{0E31}`, `ภาษา`, where ICU4X starts the run at the mark. A run
   inside one cluster gives no word of its own.
3. Marks or joiners of no script after a run's last letter are in its
   cluster and join the last word: `ภาษา\u{0300} b` gives `ภาษา\u{0300}`,
   ` `, `b`, where ICU4X makes the mark a segment of its own. The
   segmenter still sees the run alone, so the words do not change
   (`runs_are_trimmed_to_grapheme_clusters` in `analysis/tokenize/src/complex/mod.rs`).
   The one place a boundary still falls inside a cluster: GB9c joins a
   consonant to the linker before it whatever their scripts, so a Khmer
   COENG followed by a Myanmar letter is one cluster. Extending the
   Khmer run to the cluster's end would give its last word the Myanmar
   letter, so the run stops where the Myanmar run starts and the Burmese
   segmenter sees its whole run: `ក្កខ` (U+1780 U+17D2 U+1000 U+1001)
   gives `ក្`, `က`, `ခ`
   (`a_cluster_across_two_runs_is_cut_where_the_second_starts`).
4. ICU4X's older (1.x) segmenter hardcoded the script ranges and left out
   U+0EA3 LAO LETTER LO LING; the sets here come from the Unicode 18.0
   `Script` and `Line_Break` properties, and a test counts them (73 Thai,
   73 Lao, 87 Khmer, 187 Myanmar characters).
5. The other `SA` scripts (Tai Tham, Tai Viet, New Tai Lue, Ahom and
   others) have no segmenter here, and ICU4X has none for them either.
   Here their characters keep their `Word_Break` values and the UAX #29
   rules segment them like any other text
   (`a_script_without_a_dictionary_keeps_the_uax29_rules`). ICU4X's
   rule data treats `Line_Break=SA` characters apart (it takes them out
   of `Extend`), so its boundaries in these scripts can differ from
   these.

Accuracy and speed across all four scripts. Both segmenters were
evaluated on gold segmentations, scoring the words inside complex runs
only (a word is correct when the gold has a word over the same
characters):

- **Thai**: UD Thai PUD (commit
  `670083cd185a2ae215ce7aef1127c2b63640217d`, `th_pud-ud-test.conllu`,
  1000 sentences, 287631 B, 21640 gold words in complex runs).
- **Burmese**: myPOS 3.0 (commit
  `f1a620ba271cae3d8658441eb626450fca09cdfa`, `mypos-ver.3.0.txt`,
  43196 sentences, 6954379 B, 468105 gold words in complex runs).
- **Khmer**: khPOS 1.0 (commit
  `0211405c145b0a340ad558569bd06c946b2296b4`, `train.all`, 12000
  sentences, 1801664 B, 121294 gold words in complex runs).
- **Lao**: SeqLabeling Lao POS (commit
  `8025a122e95422901e0907f0fc8ca4b5c1c34235`, `Lao_train.txt`, 4613
  sentences, 798438 B, 64882 gold words in complex runs, 63644 in Lao
  runs and 1238 in embedded Thai runs).

All rows below come from `analysis/tokenize/tests/complex_eval.rs`
(ignored, since the corpora are not bundled), which asserts exact word
counts and verifies that `ComplexSegmenter::bundled()` chooses the
higher-F1 segmenter for every script (`Lstm` for Thai, `Dictionary` for
Lao, Khmer and Burmese). Throughput was measured on an `Intel(R)
Core(TM) Ultra 9 285K` (`cargo test --release`, best of 5 passes after a
warm pass, zero heap allocations per run via `LstmScratch`):

| Script | Corpus | Segmenter | Default | Precision | Recall | F1 | Throughput (MB/s) | Throughput (segs/s) |
|---|---|---|---|---|---|---|---|---|
| Thai | Thai PUD | Dictionary | | 0.690 | 0.790 | 0.737 | 72.52 | 6,967,793 |
| Thai | Thai PUD | LSTM | yes | 0.764 | 0.825 | 0.793 | 2.17 | 198,170 |
| Burmese | myPOS 3.0 | Dictionary | yes | 0.641 | 0.765 | 0.698 | 70.02 | 6,324,508 |
| Burmese | myPOS 3.0 | LSTM | | 0.532 | 0.635 | 0.579 | 2.22 | 200,597 |
| Khmer | khPOS 1.0 | Dictionary | yes | 0.737 | 0.761 | 0.748 | 68.16 | 5,037,626 |
| Khmer | khPOS 1.0 | LSTM | | 0.693 | 0.702 | 0.697 | 2.22 | 160,900 |
| Lao | SeqLabeling Lao | Dictionary | yes | 0.622 | 0.701 | 0.659 | 70.71 | 6,813,118 |
| Lao | SeqLabeling Lao | LSTM | | 0.591 | 0.678 | 0.632 | 2.17 | 212,892 |

The LSTM gains +5.6 F1 points on Thai (`0.793` vs `0.737`), whereas the
dictionary beats the LSTM on Burmese (`0.698` vs `0.579`), Khmer (`0.748`
vs `0.697`) and Lao (`0.659` vs `0.632`; on the mixed Lao+Thai runs in
`Lao_train.txt`, `bundled()` using the Thai LSTM on Thai runs and the
Lao dictionary on Lao runs reaches F1 `0.666`). Accordingly,
`ComplexSegmenter::bundled()` uses the LSTM for Thai and the dictionary
for Lao, Khmer and Burmese.

For reference, PyThaiNLP's newmm (nlpo3 1.4.0 with the PyThaiNLP 5.4.0
word list; neither is bundled) scores
precision 0.698, recall 0.645, F1 0.671 on the same Thai PUD runs.
Scored against newmm as if it were the gold, the dictionary segmenter
has precision 0.583, recall 0.722 and F1 0.645; the two give the same
words for 24 of the 1000 sentences.

### `TokenizerMe` (maxent)

`TokenizerME`: whitespace tokens are the units; a unit of one code unit,
or one the alphanumeric check accepts, is a token as it stands. In any
other unit, the model decides at every code unit position from the
second whether a token boundary falls there (`T` or `F`), with
`DefaultTokenContextGenerator`'s features: the prefix and suffix
strings, the two characters on each side with their classes (letter,
caps, digit, whitespace, end of sentence, quote, brackets), the
character pairs across the position, a flag for `&...;` entities and
one for a known abbreviation. A token's confidence is the product of the
decisions inside it. When the model splits a unit that the abbreviation
dictionary lists, the unit from the current token start to its end is a
token and the loop jumps ahead by as many positions as the unit has
dots, as the Java does; the tokens after the jump can overlap that one
(`(c)Co.ltd` listed gives `(c)Co.ltd`, `Co.ltd` and `.ltd`), and that is
kept.

The prefix and suffix features are the whole text before and after the
position, so building them at every position of a unit with no
whitespace is quadratic in the unit's length. A feature the model has
no predicate for adds nothing to the sums, so when the model loads, the
length of its longest `p=` and longest `s=` predicate is recorded, and a
prefix or suffix whose text is longer than that already in UTF-8 is not
built. The probabilities are bit for bit the same (a test compares them
with every feature built, on a 6000-byte unit), and the bytes built per
position are bounded by the model: 71 bytes for each of the two for the
English UD model.

The model file is checked as OpenNLP checks it: the component is
`TokenizerME`, `useAlphaNumericOptimization` is present, and the
classifier's outcomes are `T` and `F`. The alphanumeric optimization is
on when that property is `true` in any case. Its pattern is the
manifest's `alphaNumericPattern`, or `^[A-Za-z0-9]+$` when the manifest
has none, whatever the model's language: OpenNLP's `lang.Factory` has
per-language patterns, but the factory OpenNLP builds for a loaded model
asks for them with a language it never set, so it always gets the
default (checked on the JVM with a German model whose pattern was
removed). The per-language patterns are ported too, for a caller that
wants them. Every pattern OpenNLP ships has the form `^[characters]+$`;
the check reads that form (single characters and ranges) into a set and
tests the unit by code point as `java.util.regex` would. A pattern of
any other form, or with `[`, `]`, `\`, `^` or `&` in the class, is
refused when the model loads with the optimization on, with the pattern
in the error, rather than approximated (OpenNLP hands such a pattern to
`java.util.regex`). With the optimization off the pattern is never used
and is not read.

The whitespace units come from `TokenizerOptions`: the whitespace mode
decides which characters separate units, and also the `_ws` feature,
though a unit never holds whitespace; `keep_newlines` makes every line
feed and carriage return a unit, and so a token of probability 1, of its
own.

The annotator reads `View` and `Option<Sentences>` and writes `Tokens`
and `Confidence<Tokens>` (`f64`, the probabilities as computed). It goes
through `tokenize_units_with` like every tokenizer, so each sentence is
tokenized on its own, the tokens carry their sentence, and sentences in
another view are refused. The confidence is aligned to the tokens the
same step writes, which `Analyzer::build` accepts, so the annotator is
added to a pipeline with `Builder::add` like any other.

Deviation: Java indexes a unit by code unit, so a split can fall between
the two halves of a character outside the BMP. Such a split is not a
valid UTF-8 boundary and is not taken here; the model is still evaluated
at that position and the token goes on, its probability multiplied by
that decision's like any other. The two Java tokens that meet between
the halves are therefore one token here, with the product of their
probabilities, and each of them can hold more than half a character:
for `'😀.` the English model gives (1,2) and (2,3) in Java (UTF-16
offsets) and the one token (1,5) here (UTF-8 offsets). A token start
that the abbreviation jump would put between two halves moves to the
end of that character, so a half-character token Java emits after the
jump has nothing in its place here: with `😀.😀` in the abbreviation
dictionary of the English model Java gives (0,5) and (4,5), and this
gives (0,9). The golden files agree with the port once the Java tokens
that meet between two halves are joined this way; the variant runs
include inputs on which every model splits inside U+1F600.

### `UnigramSegmenter` (Chinese)

`opennlp.tools.tokenize.lattice.UnigramSegmenter`
(`turbo_tokenize::unigram`): dictionary and frequency segmentation for
Chinese and other text written without spaces. It reads a `View` and an
optional `Sentences` layer and writes `Tokens`, through
`tokenize_units` like the other tokenizers. `UnigramSegmenter::segment`
appends byte spans for a `&str`, and `segment_with` reuses a `Scratch`
so repeated calls allocate nothing once its buffers have grown.

The lexicon (`UnigramLexicon`) is a UTF-8 file with one word, its count
and an optional tag per line, which is the format of jieba's `dict.txt`.
It is read as `UnigramSegmenter.load` reads it: lines end at `\n`, `\r`
or `\r\n`; a line is trimmed of Unicode `White_Space` in both whitespace
modes; the word ends at the first whitespace character in the configured
mode; the count is read as `Long.parseLong` reads it (an optional sign,
then decimal digits of any script in the Basic Multilingual Plane, such
as fullwidth digits) and must be positive; a word listed twice gets the
sum of its counts; the total must fit in a Java `long`; at most ten
million distinct words (`ResourceLimits.MAX_ENTRIES`) unless the caller
sets another limit. The errors carry the Java's messages and line
numbers. A word's score is `log(count) - log(total)`, and an unlisted
character scores `log(0.5) - log(total)`, below every listed word. The
file's SHA-256 is the model's hash in the provenance.

The text is split at whitespace in the configured mode, and each
whitespace-free stretch is cut by dynamic programming: from each
character in order, the character alone is tried at the unknown score,
then every lexicon word that starts there, shortest first, and a
position takes a new best path only when its sum is strictly higher. So
of two paths with equal sums, the one whose last word starts first is
kept, as in the Java. This is jieba's `get_DAG` and `calc` without its
hidden Markov model for unlisted words.

The trie is laid out for that walk rather than copied from the Java,
whose nodes keep sorted child arrays searched by binary search: the
root's edges on characters of the Basic Multilingual Plane are a table
indexed by the character, every other edge is a slot of one open
addressing hash table keyed by (parent, character), and each step
carries a 32-bit filter of the characters the child's own edges are on,
so a walk usually ends without a probe. The Java's trie has an edge per
UTF-16 code unit and this one an edge per character; both find the same
words, since a word is whole characters and the walk starts on a
character boundary.

The scores need the logarithm the JVM computes. OpenNLP calls
`Math.log`, not `StrictMath.log`, and on the JVM the goldens are made
with (Temurin 21.0.12, x86-64) `Math.log` differs from fdlibm on 112599
of the integers 1 to 2^24. The segmenter uses `turbo_jvm::math::log`, a
correctly rounded logarithm in double-double arithmetic that uses no
platform function: it equals that JVM's `Math.log` on every integer from
1 to 2^24 but one (14237211, where the true value lies within 1e-22 of a
midpoint and the JVM returns the farther double) and on 4 million
random integers up to 2^63. The segmenter takes logarithms only of
integers (the counts and their total) and of 0.5. The parity is with
that JVM build: `Math.log` is the JVM's own routine, which differs
between ports and turns back into fdlibm with `-XX:-UseLibmIntrinsic`.
A lexicon with a count or a total of 14237211 gets a score one unit in
the last place from the JVM's. The choice matters only where two paths have
equal exact sums, such as counts with equal products: the fixture
lexicon has such a pair (`人民` 1 `大会` 148 against `人民大` 2 `会` 74)
that the JVM and the port both cut `人民大|会`, and that fdlibm's
logarithm would cut `人民|大会`.

Deviation: none in the tokens. The Java iterates over code unit
positions and the port over character boundaries; the positions inside a
surrogate pair are never reached in the Java (the step over an unlisted
character takes the whole pair), so skipping them changes nothing. The
lexicon file must be UTF-8 (the Java takes any charset), and the entry
limit is a parameter rather than the `OPENNLP_MAX_ENTRIES` system
property. A line that starts with a character that is whitespace in the
legacy mode but not Unicode `White_Space` (`U+001C` to `U+001F`) lists the
empty word in legacy mode; the Java keeps it on the trie's root, where
nothing scores it, but counts it in the total and the entries, and so
does the port.

The goldens are the runs of three lexicons, each in both whitespace
modes: `zh-unigram-fixture.txt` (OpenNLP's test lexicon, words with
equal count products, CR LF and lone CR line ends, fullwidth digits and
a `+` sign in counts, no-break and ideographic spaces between word and
count, a word outside the Basic Multilingual Plane, a word listed twice)
on the `zh`, `multi`, `edge` and `surrogate` inputs;
`jieba-dict-subset.txt` (the lines of jieba's `dict.txt` whose word
occurs in the `zh`, `multi` or `edge` inputs) on those sets; and
`zh-unigram-modes.txt`, whose words differ between the modes, on the
`zh` inputs. Every token matches, on 264 goldens. In a run outside the
repository at commit `0260665`, with the whole of jieba's `dict.txt`
(fxsjy/jieba commit `67fa2e36e72f69d9134b8a1037b83fbb070b9775`, 349046
lines), the port's tokens equalled the JVM's on every line of the PKU
and MSR test sets of the SIGHAN 2005 bakeoff (1945 and 3985 lines,
102198 and 108642 tokens). That comparison is not committed and cannot
be repeated from the repository, since neither the dictionary nor the
bakeoff data is in it.

Against the bakeoff's gold segmentation, scored like its `score` script
(`analysis/tokenize/tests/unigram_sighan.rs`, ignored by default, with the dictionary and
the data supplied by the caller; neither is in the repository, and the
bakeoff data is for research use only):

| Test set | Segmenter | P | R | F1 | OOV recall |
|---|---|---|---|---|---|
| PKU | this port (equal to OpenNLP) | 0.824 | 0.807 | 0.816 | 0.419 |
| PKU | jieba-rs 0.11.0, no HMM | 0.859 | 0.814 | 0.836 | 0.543 |
| PKU | jieba-rs 0.11.0, HMM | 0.853 | 0.787 | 0.818 | 0.583 |
| MSR | this port (equal to OpenNLP) | 0.820 | 0.834 | 0.827 | 0.338 |
| MSR | jieba-rs 0.11.0, no HMM | 0.820 | 0.834 | 0.827 | 0.338 |
| MSR | jieba-rs 0.11.0, HMM | 0.816 | 0.810 | 0.813 | 0.449 |

All use the same `dict.txt`. The rows for this port are what
`unigram_sighan.rs` asserts; the jieba-rs rows were made at commit
`0260665` by a program outside the repository (jieba-rs is not a
dependency of the workspace). On MSR the port's output and jieba-rs's
without its HMM are the same file. On PKU, of the 1851 tokens jieba-rs
without its HMM finds and the port does not, 1849 are runs of ASCII
digits or letters (`12`, `2000`, `1999`): jieba keeps a run of unlisted
ASCII alphanumerics together, and OpenNLP's segmenter makes each
unlisted character a word of its own. The PKU text writes numbers in
ASCII digits and the MSR text in fullwidth ones, which jieba does not
group. The other two are one tie broken the other way: `手牵着` 3 `手`
28466 and `手` 28466 `牵着手` 3 have equal sums, OpenNLP keeps
`手|牵着手`, whose last word starts first, and jieba, whose route is
computed from the end of the sentence, keeps `手牵着|手`. The port keeps
OpenNLP's behaviour in both; grouping such runs would be a change to
OpenNLP.

Speed is not measured here: no matched benchmark against jieba-rs is
committed, so this section makes no speed claim (AGENTS.md, rule 7).

### `LatticeTokenizer` (Japanese, Korean)

`opennlp.tools.tokenize.lattice.LatticeTokenizer` and its dictionary
reader `MecabDictionary` (`turbo_tokenize::lattice`): morphological
segmentation with a MeCab dictionary. It reads a `View` and an optional
`Sentences` layer and writes `Tokens` and `PosTags`, through
`tokenize_units` like the other tokenizers. The tag of a token is the
first feature column of its dictionary entry, in a tag set named
`mecab` built from the dictionary. `LatticeTokenizer::analyze` appends
`Morpheme`s for a `&str` (a byte span, the dictionary entry and whether
unknown word handling made it), `MecabDictionary::features` gives an
entry's feature columns, and `analyze_with` reuses a `Scratch` so
repeated calls allocate nothing once its buffers and the output have
grown (`analysis/tokenize/tests/lattice_allocations.rs` counts the
allocations). The annotator makes a new `Scratch` per call. The model
in the provenance is `opennlp:MecabDictionary` with the SHA-256 of a
manifest of the files read: for each, in the order read, its name, a
tab, the SHA-256 of its bytes in hex and a line feed.

The dictionary is a MeCab source directory, read as
`MecabDictionary.load` reads it; the compiled `sys.dic` is not read,
by OpenNLP or here. The files are read in this order:

- `matrix.def`, first, because its dimensions bound every context id.
  The header holds two positive numbers, each at most the entry limit
  (ten million by default), whose product is at most 134217728 cells.
  Each line after it is a right id, a left id and a cost, which must fit
  in 16 bits. Every pair must be listed exactly once.
- Every `*.csv` file, sorted by name. Each line is `surface,left
  id,right id,cost,features...`, split with MeCab's quoting (a field in
  double quotes may hold commas, and `""` is a quote). The left id must
  be below the matrix's second dimension and the right id below its
  first. At most the entry limit of entries across the files.
- `char.def`. Category lines are `NAME INVOKE GROUP LENGTH`, with at
  most 17 categories and a `LENGTH` from 0 to 15; `DEFAULT` must be
  defined. Mapping lines are `0xXXXX` or `0xXXXX..0xYYYY` followed by
  one or more category names, and a later mapping wins where two
  overlap. Text after `#` is a comment.
- `unk.def`, in the lexicon format. Its surfaces are category names,
  which `char.def` must define.

Lines end at `\n`, `\r` or `\r\n`, are trimmed of Unicode `White_Space`
in both whitespace modes, and are split into fields at whitespace in the
configured mode. Numbers are read as `Integer.parseInt` reads them:
decimal digits of any script in the Basic Multilingual Plane, and
`char.def`'s hexadecimal also accepts fullwidth letters. The checks and
their order are the Java's, and a malformed file is an error that names
the file, the line and the problem (`DictionaryError`), never a panic.
The entry and cell limits are a parameter (`DictionaryLimits`) rather
than the `opennlp.max.entries` and `opennlp.max.matrix.cells` system
properties. The surfaces go in a double-array trie with one transition
per UTF-8 byte, and each surface's entries stay in file order. The
Java's trie has one transition per UTF-16 code unit. Both find the same
surfaces, since a surface is whole characters and a walk starts on a
character boundary.

The text is split at whitespace in the configured mode. Each
whitespace-free stretch is cut on the path through the word lattice
with the lowest sum of word costs and connection costs, the start and
end of the stretch taking context id 0. The candidates are visited from
each character boundary that a node ends at, in order:

1. The lexicon entries of every surface that starts there, shortest
   surface first and each surface's entries in file order.
2. Unknown word candidates, when nothing matched or the character's
   first category has `INVOKE` set. The candidates use that category's
   `unk.def` templates. The run of characters is extended while each
   character shares a category with the next, as MeCab's
   `seekToOtherType` does. `GROUP` offers the whole run, and `LENGTH`
   offers its first 1 to `LENGTH` characters, without repeating the
   whole run.
3. When neither offered anything, one character with the category's
   templates, or with `DEFAULT`'s where the category has none.

Each candidate takes the predecessor whose total is strictly lowest,
looking at the nodes that end at its start, latest added first. At the
end of the stretch, the first node with the strictly lowest total wins.
These rules decide ties, and the port keeps them; the fixture
dictionaries have ties of each kind. A dictionary whose `unk.def` has no
`DEFAULT` template can leave a character without a candidate. Then the
Java throws, naming the UTF-16 position, and the port returns
`NoCandidate` with the byte offset.

Deviations, each with a test in `analysis/tokenize/src/lattice`:

- The files must be UTF-8. The Java takes any charset. IPADIC, which is
  EUC-JP, is read from a UTF-8 copy that `turbo-goldens fetch-models`
  writes so that it is the text Java's decoder reads (see the last of
  the OpenNLP findings).
- The port checks that a whole file is UTF-8 before reading its lines.
  The Java decodes as it reads, so where a malformed line comes before
  bad bytes, the two report different errors.
- A `matrix.def` that declares more than 65536 cells but has fewer bytes
  after its header than its lines need (six per cell, less one) is
  refused before its table is allocated. The Java allocates the table from the header
  first, then finds that pairs are missing.
- In the legacy whitespace mode, a `char.def` line made only of
  characters that are whitespace in that mode but not Unicode
  `White_Space` (`U+001C` to `U+001F`) is a malformed category. In the
  Java it throws an `ArrayIndexOutOfBoundsException`.
- Where `unk.def` names several undefined categories, the port reports
  the first in file order, and the Java the first in its `HashMap`'s
  order.
- More than 65536 distinct first feature columns are refused, since a
  `Tag` numbers at most that many. The Java has no tags.

The goldens are the runs of six dictionaries
(`analysis/goldens-tool/README.md`, "Lattice goldens"). Three are
fixtures written for this project. `mecab-fixture-ja`, run in both
whitespace modes on the `ja`, `multi`, `edge` and `surrogate` inputs,
has costs that make ties, a quoted surface with a comma, overlapping
categories and supplementary characters. `mecab-fixture-ko` runs on the
`ko`, `multi` and `edge` inputs. `mecab-fixture-partial`, whose
`unk.def` has no `DEFAULT` template, runs on `ja` and `edge`, and throws
on most of them. The other three are published dictionaries:
`mecab-ipadic-2.7.0-20070801` and `unidic-mecab-2.1.2` on `ja`, `multi`,
`edge` and `surrogate`, and `mecab-ko-dic-2.1.1-20180720` on `ko`,
`multi`, `edge` and `surrogate`. For every golden, the port's spans,
feature columns and unknown flags equal the JVM's, and on every golden
where the JVM threw, the port fails at the same character:

| Dictionary | Goldens | Morphemes | Unknown | JVM errors |
|---|---|---|---|---|
| `mecab-fixture-ja` | 65 | 1497 | 1244 | 0 |
| `mecab-fixture-ja+legacy` | 65 | 1494 | 1241 | 0 |
| `mecab-fixture-ko` | 55 | 1310 | 1155 | 0 |
| `mecab-fixture-partial` | 48 | 17 | 11 | 44 |
| `mecab-ipadic-2.7.0-20070801` | 65 | 1512 | 1068 | 0 |
| `mecab-ko-dic-2.1.1-20180720` | 61 | 1360 | 971 | 0 |
| `unidic-mecab-2.1.2` | 65 | 2016 | 519 | 0 |

`analysis/tokenize/tests/lattice_golden.rs` asserts these numbers. The
fixture runs always run. The IPADIC, mecab-ko-dic and UniDic tests are
ignored by default and read the dictionaries `turbo-goldens fetch-models`
fetched, from `TURBO_MECAB_DICTIONARIES` or
`target/turbo-goldens/models`. CI runs them after `fetch-models`, with
the dictionaries required. The UniDic run uses the 17-column source
distribution (`unidic-mecab-2.1.2.tar.gz`), which both the JVM and Rust
readers parse directly; unidic-lite 1.0.8 holds only a compiled
dictionary, which neither reader reads (see "Compiled dictionaries"
below).

#### Readings

OpenNLP leaves an entry's features as one string. `LatticeAnalyzer` is
the lattice tokenizer together with the dictionary's format
(`DictionaryFormat`), and writes a third layer beside `Tokens` and
`PosTags`: `Readings` (`turbo:readings`, in `turbo-tokenize-layers`),
aligned to the tokens, so its offsets are the tokens' own. It has three
columns, each `None` where the format has no such column:

| Format | Reading | Pronunciation | Lemma | Columns |
|---|---|---|---|---|
| `Ipadic` | 読み | 発音 | 原形 | 9 |
| `MecabKoDic` | 읽기 | none | none | 8 |
| `Unidic(Source17)` | none | `pron` | `orthBase` | 17 |
| `Unidic(Full26)` | `kana` (column 17) | `pron` | `orthBase` | 26 |
| `Unidic(Full29)` | `kana` (column 20) | `pron` | `orthBase` | 29 |

Each format's columns are an enum in file order (`IpadicColumn`,
`KoDicColumn`, and `UnidicColumn` for the 17 columns every UniDic
layout starts with), and `DictionaryFormat::map` names the three columns
from them, so no column is looked up by a string. The UniDic layouts are
the 2.1.2 source distribution (17 columns), the 2.1.2 binary
distribution and unidic-lite (as the 26 fields of unidic-lite's `bos-feature`),
and 2.2.0 and later (29). The format is given, not guessed:
`LatticeAnalyzer::new` checks that every lexicon entry has the format's
number of columns and refuses the dictionary otherwise (`FormatError`).
The `unk.def` templates are not checked, since they usually have fewer
columns (IPADIC's have seven). For a dictionary known to be UniDic,
`UnidicLayout::of` picks the layout whose column count a lexicon entry
has, and returns it only if every lexicon entry passes that check.

An item is an empty string where the token's entry leaves the column
out: an unknown word whose template lacks it, or a value that is empty
or `*`, which the published dictionaries write for none (mecab-ko-dic's
symbols have `*` for 읽기). `FeatureMap::reading`, `pronunciation` and
`lemma` give the same values for one entry, as `Option<&str>`, outside a
pipeline. For UniDic the lemma is `orthBase`, the base form as written,
which is what IPADIC's 原形 holds. UniDic's own `lemma` column is the
lexeme in a standard spelling with an English gloss on some loanwords
(為る for する, `テスト-test`), and stays in the features.

`analysis/tokenize/tests/lattice_readings.rs` runs the analyzer on every
golden of `mecab-fixture-ko` (in mecab-ko-dic's layout),
`mecab-ipadic-2.7.0-20070801`, `mecab-ko-dic-2.1.1-20180720` and
`unidic-mecab-2.1.2` (`Unidic(Source17)`), and checks each token's three
items against the columns of the feature string the JVM printed for it.
The fetched dictionaries' tests run in CI with the lattice goldens. The
26- and 29-column UniDic layouts (`Full26` and `Full29`) are also tested
on entries written in each layout
(`analysis/tokenize/src/lattice/format.rs`):

| Dictionary | Tokens | With a reading | With a pronunciation | With a lemma |
|---|---|---|---|---|
| `mecab-fixture-ko` | 1310 | 46 | none | none |
| `mecab-ipadic-2.7.0-20070801` | 1512 | 444 | 444 | 444 |
| `mecab-ko-dic-2.1.1-20180720` | 1360 | 211 | none | none |
| `unidic-mecab-2.1.2` | 2016 | none | 1219 | 1496 |

On IPADIC the 444 are the tokens that are not unknown words. On UniDic
2.1.2 (`Source17`), 1496 are the tokens that are not unknown words (and
carry `orthBase`), while 277 of those are symbols and punctuation whose
`pron` column is empty (`""`), leaving 1219 with a pronunciation. On the
Korean dictionaries the other known tokens are mostly punctuation.

#### Compiled dictionaries, lindera and vibrato

Two Rust MeCab-style tokenizers were read for this comparison: lindera
6.2.0 with lindera-dictionary 6.2.0 (MIT) and vibrato 0.5.2 (MIT or
Apache-2.0). Neither is a dependency, and no code was taken from either.

| | turbo `lattice` | lindera 6.2.0 | vibrato 0.5.2 |
|---|---|---|---|
| Input | MeCab source directory: `*.csv`, `matrix.def`, `char.def`, `unk.def` | MeCab source directory, with a `metadata.json` per dictionary that names its columns, encoding and options | MeCab source files, given as readers (the lexicon as one concatenated `lex.csv`) |
| Encodings | UTF-8 only (IPADIC from a UTF-8 copy) | any `encoding_rs` encoding the metadata names | UTF-8 |
| MeCab's compiled `sys.dic`, `matrix.bin`, `char.bin`, `unk.dic` | not read | not read | not read |
| Own compiled format | none: the source is parsed at load | yes (built with `rkyv`) | yes (built with `bincode`) |
| Dictionaries packaged | none | IPADIC, IPADIC-NEologd, UniDic 2.1.2 (from the source distribution), mecab-ko-dic, CC-CEDICT, Jieba, SudachiDict | none in the crate |
| Feature columns | the string; typed readings via `DictionaryFormat` | the string, and named fields from `metadata.json` (`token.get("reading")`) | the string |
| Origin | a port of OpenNLP's `LatticeTokenizer`, checked against JVM goldens | a fork of kuromoji-rs | its own, with options for MeCab-compatible output |

Neither reads MeCab's compiled files: both build from the source
distribution into a format of their own. UniDic is published in source
form as well as compiled: lindera-unidic and `turbo-goldens fetch-models`
both use `unidic-mecab-2.1.2.tar.gz`, the 17-column source distribution,
under UniDic's BSD licence (UniDic is offered under BSD, GPL or LGPL).
Only unidic-lite was found to ship compiled files alone. So every
dictionary this reader is meant for can be read from source, and reading
compiled dictionaries is not added. What reading one would take, from
unidic-lite's files:

- `sys.dic` and `unk.dic`: a 72-byte header (ten little-endian `u32`,
  the first the file size XOR `0xEF718F77`, the second the
  version, 102, then the entry count, the left and right id counts and
  the sizes of the next three parts, and a 32-byte charset name), then a
  Darts double array mapping each surface to a run of entries, the
  entries (16 bytes each: left id, right id, part-of-speech id, cost,
  feature offset, compound), and the NUL-terminated features.
  unidic-lite's has 756264 entries and 5981 context ids each side.
- `matrix.bin`: two `u16` dimensions and the costs as `i16`.
- `char.bin`: the category count, 32-byte category names and a packed
  `u32` per code point from U+0000 to U+FFFE. It has no entry for a
  supplementary character, where `char.def` can map one.

That is a Darts reader and four small parsers, a few hundred lines with
tests, but with no reference to check them against: OpenNLP reads only
the source, so no golden covers the compiled path. The entries of one
surface are in the order `mecab-dict-index` wrote them, and that order
decides ties, so a compiled dictionary could segment differently from
its source here without any test noticing. The source distributions
remain the input, and `unidic-mecab-2.1.2` is fetched by
`turbo-goldens fetch-models` and checked in the lattice goldens.

#### Throughput

`turbo-bench` records embedding runs only. The command for a lattice run
is

```text
cargo run --release -p turbo-tokenize --example lattice_throughput [ipadic|ko-dic|unidic] [readings]
```

with the dictionaries in `TURBO_MECAB_DICTIONARIES` or
`target/turbo-goldens/models`. It reads the golden inputs of the
language (`inputs/ja` or `inputs/ko`) repeated to at least 10 MiB and
prints the dictionary's manifest digest, the text's SHA-256 and size,
the morpheme count, and the best of five timed runs after a warm one,
and with `readings` also the analyzer (`Tokens`, `PosTags` and
`Readings`) through a document.

On an `Intel(R) Core(TM) Ultra 9 285K` (Linux x86_64, `rustc` release
profile, Eclipse Temurin `21.0.12.1+1-LTS` on OpenNLP
`cdf33c653d5d61d5af7d37ffee189a83bd8392ea`), the matched runs on the
three published dictionaries and two 10 MiB texts are:

- Texts:
  - `inputs/ja`: `10,485,900` bytes, SHA-256
    `97b817559b05be586c663b559bc837996a1473eae7b731c1dbe3f4805d5c6d24`
  - `inputs/ko`: `10,486,625` bytes, SHA-256
    `9d0b2460367c153343f93d589eb3229c100fc99db5433ff812835f0478864667`
- Dictionaries (manifest SHA-256 over `matrix.def`, sorted `*.csv`,
  `char.def`, `unk.def`):
  - `ipadic`: `mecab-ipadic-2.7.0-20070801.utf8`,
    `6103f8e96a3c1cc7275da1efea6257656f8e50d2c427ab08646d920a4aece735`
    (JVM EUC-JP `mecab-ipadic-2.7.0-20070801`:
    `e85ee610665107cf79ecaa877c642362a64ce7ef32ddbe302956144ec901d8a8`)
  - `ko-dic`: `mecab-ko-dic-2.1.1-20180720`,
    `5fe7fa0a0274fbb36ae3cce5c94f7180adbdc8793a5fdf10b83f954ea2c42f83`
  - `unidic`: `unidic-mecab-2.1.2`,
    `a14472763e76de14df09f12bc66896e4bb69d4fea35c43eba8bfa3dc901d5505`

| Dictionary | Engine | Build / load | Morphemes | Best of 5 (s) | Throughput (MB/s) | Morphemes/s |
|---|---|---|---:|---:|---:|---:|
| `ipadic` (`inputs/ja`) | `turbo_tokenize::lattice` (`analyze_with`) | load `0.468 s` (source) | 2,220,400 | 0.2845 | 36.86 | 7,804,816 |
| `ipadic` (`inputs/ja`) | `turbo_tokenize::lattice` (`LatticeAnalyzer` + `Readings`) | load `0.468 s` (source) | 2,220,400 | 0.3754 | 27.93 | 5,914,758 |
| `ipadic` (`inputs/ja`) | OpenNLP `LatticeTokenizer` (JVM, UTF-8 / EUC-JP) | load `0.739 s` / `1.243 s` (source) | 2,220,400 | 0.5586 / 0.5302 | 18.77 / 19.78 | 3,975,142 / 4,187,495 |
| `ipadic` (`inputs/ja`) | `lindera 6.2.0` (`Segmenter::segment_with_lattice`) | build `0.778 s`, load `0.0006 s` | 2,281,400 | 0.3156 | 33.22 | 7,228,578 |
| `ipadic` (`inputs/ja`) | `vibrato 0.5.2` (`Worker`, whitespace-split / lines) | build `0.613 s`, load `0.025 s` | 2,220,400 / 2,250,900 | 0.2851 / 0.2784 | 36.78 / 37.66 | 7,788,464 / 8,083,879 |
| `ko-dic` (`inputs/ko`) | `turbo_tokenize::lattice` (`analyze_with`) | load `1.280 s` (source) | 1,824,185 | 0.7582 | 13.83 | 2,405,993 |
| `ko-dic` (`inputs/ko`) | `turbo_tokenize::lattice` (`LatticeAnalyzer` + `Readings`) | load `1.280 s` (source) | 1,824,185 | 0.7747 | 13.54 | 2,354,699 |
| `ko-dic` (`inputs/ko`) | OpenNLP `LatticeTokenizer` (JVM, UTF-8) | load `2.565 s` (source) | 1,824,185 | 1.0919 | 9.60 | 1,670,582 |
| `ko-dic` (`inputs/ko`) | `lindera 6.2.0` (`Segmenter::segment_with_lattice`) | build `2.230 s`, load `0.0005 s` | 1,863,205 | 0.6650 | 15.77 | 2,801,977 |
| `ko-dic` (`inputs/ko`) | `vibrato 0.5.2` (`Worker`, whitespace-split / lines) | build `2.437 s`, load `0.045 s` | 1,824,185 / 1,863,205 | 0.6358 / 0.6418 | 16.49 / 16.34 | 2,869,244 / 2,903,253 |
| `unidic` (`inputs/ja`) | `turbo_tokenize::lattice` (`analyze_with`) | load `4.015 s` (source) | 2,348,500 | 0.4225 | 24.82 | 5,558,656 |
| `unidic` (`inputs/ja`) | `turbo_tokenize::lattice` (`LatticeAnalyzer` + `Readings`) | load `4.015 s` (source) | 2,348,500 | 0.5332 | 19.67 | 4,404,539 |
| `unidic` (`inputs/ja`) | OpenNLP `LatticeTokenizer` (JVM, UTF-8) | load `5.117 s` (source) | 2,348,500 | 0.7587 | 13.82 | 3,095,325 |
| `unidic` (`inputs/ja`) | `lindera 6.2.0` (`Segmenter::segment_with_lattice`) | build `2.207 s`, load `0.0004 s` | 2,366,800 | 0.4330 | 24.22 | 5,466,111 |
| `unidic` (`inputs/ja`) | `vibrato 0.5.2` (`Worker`, whitespace-split / lines) | build `3.049 s`, load `0.110 s` | 2,348,500 / 2,360,700 | 0.4067 / 0.4047 | 25.78 / 25.91 | 5,774,240 / 5,833,625 |

`LatticeTokenizer` (both OpenNLP and `turbo_tokenize::lattice`) splits
at Unicode `White_Space` and runs Viterbi on each whitespace-free
stretch. When `vibrato` (`ignore_space(true)`) is fed the same
whitespace-split stretches, it produces the exact same morpheme counts
(`2,220,400`, `1,824,185`, `2,348,500`). When fed whole lines (`vibrato`
lines) or sentence-split text (`lindera`), non-ASCII whitespace not
listed in the dictionary's `SPACE` category remains in the lattice,
increasing the token count slightly.

### `Detokenizer`

`DictionaryDetokenizer` (`turbo_tokenize::detok`): the detokenizer
dictionary (the same XML, with an `operation` attribute per entry:
`MOVE_LEFT`, `MOVE_RIGHT`, `MOVE_BOTH`, `RIGHT_LEFT_MATCHING`) decides
how each token attaches to its neighbours; a matching token (a straight
quote) attaches to the right on its first occurrence and to the left on
the next, with one flag per distinct token. `DetokenizerDictionary`
reads the XML through `turbo-maxent`'s `Entries` and refuses, as the
Java does, an entry without exactly one token or with an unknown or
missing operation. The dictionary holds a `DictionaryOperation` (the
four in the XML) per token; `Detokenizer` gives the resolved
`Operation` of each token (merge left, right or both, or none, never
the matching one), or the rebuilt text with an optional split marker
wherever two tokens were joined without a space. Its offset form,
`detokenize_spans`, also records the byte offset where each token
starts in the rebuilt text, which OpenNLP does not provide. The
operations are resolved one token ahead while the text is written. A
call reserves the output once, for the tokens and a space or the
marker in every gap, and allocates one other thing: when the dictionary
has more than 64 matching tokens, a vector of one `u64` flag word per
64 matching tokens past the first 64.

It is not an annotator: a `Tokens` layer keeps its spans, so the text
between tokens is never lost and nothing needs rebuilding. The
detokenizer is for tokens that arrive without their text.

## Golden files

Each port is checked against what the OpenNLP 3.x JVM returned on the
same input. The goldens are made by `turbo-goldens` (crate
`turbo-goldens-tool`, `analysis/goldens-tool`), which runs a Java
generator in a container built from a fixed JDK image and OpenNLP
commit, and live in `analysis/goldens/data`:
`tokenize/`, `sentdetect/` and `detokenize/` hold a directory per run,
named after the model and any options (`+legacy` for OpenNLP's legacy
whitespace mode, `+keep-new-lines`, `+alphanumeric`,
`+english-dictionary`), or after the component when it has no model
(`simple`, `whitespace`, `uax29`, `newline`). The format and the list of
runs are in `docs/analysis.md` ("Golden files") and
`analysis/goldens-tool/README.md` ("Runs"). A golden holds typed layers
with UTF-8 byte offsets, the JVM's exact UTF-16 spans where a span cuts
a character, and probabilities as the bits of Java's doubles; the tests
read them with `turbo_goldens` and compare spans exactly and
probabilities exactly (the sentence detector) or within 1e-9 (the
tokenizer).

Beside every maxent golden, a dump records each decision the model was
asked for: the context strings and every outcome's probability, and for
the sentence detector whether the abbreviation dictionary refused a
split the model wanted. The tests check the Rust context generators
feature by feature against it, and that the tokenizer was asked about
exactly the positions its loop visits, the jump over an abbreviation's
dots included.

The unigram segmenter's goldens are under `unigram/`, one directory
per lexicon and mode (`analysis/goldens-tool/README.md`, "Unigram
goldens"), and the lattice tokenizer's under `lattice/`, one directory
per dictionary and mode ("Lattice goldens").

The runs cover cases none of the published models reach: `TokenizerME`
with an abbreviation dictionary (a fixture model, and the English model
with `english-dictionary.xml`), splitting between surrogate halves, the
alphanumeric optimization, two fixture sentence models (one with an
abbreviation dictionary), and the dictionary detokenizer on the
English, French, Portuguese, Latin and quotes dictionaries.

The JVM is used for golden files only. Nothing here calls into it at
run time.

## OpenNLP findings

Things found while porting that OpenNLP could change, kept here so they
reach the OpenNLP side:

- `SimpleTokenizer` splits a character outside the BMP into two tokens
  of half a character each, because it classifies code units.
- `SimpleTokenizer.isLineSeparator` and `WhitespaceTokenizer.isLineSeparator`
  compare a `char` with `Character.LINE_SEPARATOR` and
  `Character.LETTER_NUMBER`, which are general category byte constants
  (13 and 10), not characters. The test is for CR and LF only by
  coincidence.
- `TokenizerME` can place a split between surrogate halves for the
  same reason.
- `TokenizerFactory.getAlphaNumericPattern` passes its `languageCode`
  field to `lang.Factory`, and a factory made for a loaded model leaves
  that field null (only `getLanguageCode()` reads the language from the
  model). A model without `alphaNumericPattern` therefore always gets the
  default `^[A-Za-z0-9]+$`, never its language's pattern.
- In `TokenizerME.tokenizePos`, a split inside a unit the abbreviation
  dictionary lists emits the unit from the token start to its end, then
  keeps splitting after the dot jump, so the tokens overlap.
- `SimpleTokenizer` with `keepNewLines` throws an exception when the
  text starts with a line break (or any line break comes before the
  first token), because it tries to close a token that was never
  opened.
- `SimpleTokenizer` with `keepNewLines` reports a wrong span for a line
  break that follows a space: the start offset is left over from the
  previous token instead of being set to the line break's position.
- `Dictionary(InputStream)` inserts the entries before its
  `isCaseSensitive` field is assigned, so every entry is stored as if the
  dictionary were case-insensitive. In a case-sensitive dictionary, entries
  that differ only in case collapse to the first one: with `Dr.` and
  `dr.` loaded, `contains("dr.")` is false. The Rust reader reproduces
  this so lookups match; a test records it.
- `DefaultSDContextGenerator` tests `Character.isUpperCase` on the
  first code unit of a word, so a word starting with an upper case
  letter outside the BMP (Deseret, Osage, mathematical letters) is
  never `xcap`.
- `DefaultSDContextGenerator` stops its search for the end of the suffix
  one code unit before the end of the text (`nextSpaceIndex` with
  `lastIndex`), so when no whitespace follows a candidate the last
  character of the text is in neither the suffix nor the next word, and
  a text ending in a character outside the BMP gives a suffix holding only
  its high surrogate.
- `SentenceDetectorFactory.getEOSCharacters` and the scanner and context
  generator built from it fall back to the language default through the
  `languageCode` field, which is null on a factory loaded from a model
  file. A model without `eosCharacters` therefore gets `. ! ?` in every
  language, and the Thai context generator is never used for a loaded
  model.
- `Dictionary` lookups hash a probe by its lowercase form and confirm it
  with `String.compareToIgnoreCase`. For 23 code points (such as U+00B5,
  U+0131, U+017F and U+03C2) `compareToIgnoreCase` accepts another
  character whose lowercase form differs, so the two hash apart and match
  only on a 32-bit hash collision. Equal lowercase forms always pass
  `compareToIgnoreCase` (checked over every code point on JDK 21), so
  comparing lowercase forms, as the Rust dictionary does, agrees with the
  JVM except on such a collision.
- `UnigramSegmenter` computes its scores with `Math.log`, whose result
  the JDK allows to differ by one unit in the last place between
  platforms, so two paths whose exact sums are equal can be cut
  differently on different JVMs. `StrictMath.log` would make the
  segmentation the same everywhere.
- `UnigramSegmenter` makes each character the lexicon does not list a
  word of its own, so an unlisted number such as `2000` is four words;
  jieba keeps runs of unlisted ASCII letters and digits together, which
  is most of what separates the two on the PKU test set.
- `UnigramSegmenter.load` accepts a line whose word is empty in the
  legacy whitespace mode (one starting with `U+001C` to `U+001F`, which
  `trimUnicodeWhitespace` keeps and `isWhitespace` splits on); the empty
  word goes on the trie's root and is never used, but its count is in
  the total.
- `MecabDictionary.load` reads `char.def` with `trimUnicodeWhitespace`
  and then splits it with `isWhitespace`. In the legacy whitespace mode,
  a line made only of `U+001C` to `U+001F` therefore has no fields, and
  it throws an `ArrayIndexOutOfBoundsException` instead of an
  `IOException`.
- `MecabDictionary.load` checks the categories `unk.def` names by
  iterating over a `HashMap`. Where several are undefined, which one the
  message names depends on hash order, not on the file.
- `MecabDictionary.load` allocates the connection matrix from the
  `matrix.def` header before reading a line. A header declaring the
  largest allowed matrix makes it allocate 256 MiB for the costs, even
  when the file is a few bytes.
- Java's `EUC-JP` decoder reads 0xA1BD as `U+2014` EM DASH, and glibc's
  `iconv` reads it as `U+2015` HORIZONTAL BAR. IPADIC has that code in
  four lexicon files, `Symbol.csv` among them. So IPADIC loaded by
  OpenNLP in EUC-JP and a UTF-8 copy made with `iconv` have different
  surfaces for that symbol: text with `U+2015` matches the entry in the
  copy but not in OpenNLP. `turbo-goldens fetch-models` writes its copy
  with `U+2014`, as Java reads it, and the generator checks that every
  file of the copy is the text Java decodes from the original.
