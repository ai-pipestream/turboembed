# Golden data

This directory holds the input texts the golden files are made from and
the golden files themselves. `docs/analysis.md` describes the golden
format.

## Inputs

`inputs/<set>/<id>.txt` are UTF-8 texts. A file's bytes are the text:
there is no byte order mark and the final line end is part of the text.

The texts were written for this project to exercise the annotators, and
are not taken from any corpus.

- `inputs/en/`: English texts with abbreviations ("Dr.", "U.S."), straight
  and curly quotes, numbers with periods, ellipses, URLs and an email
  address, a numbered and a bulleted list, all caps sentences, emoji
  (characters outside the Basic Multilingual Plane, so surrogate pairs in
  Java), lines holding only whitespace, and paragraphs. `emoji-split.txt`
  puts an emoji after a quote, a comma and an ampersand, where
  `TokenizerME` splits it between its two UTF-16 units.
- `inputs/multi/`: one text each in German, French, Spanish, Portuguese,
  Italian, Dutch, Russian, Japanese, Chinese and Arabic, and a mixed
  script text with emoji, CJK Extension B characters and mathematical
  letters outside the Basic Multilingual Plane.
- `inputs/edge/`: 34 short texts for the edge cases of the segmenters:
  abbreviations, numbers, quotes and brackets, dashes and hyphens,
  ellipses, URLs and e-mail addresses, apostrophes, CR LF and lone CR,
  tabs, no-break and ideographic spaces, `U+0085`, `U+2028`, `U+2029`,
  the separators `U+001C..U+001F`, leading and trailing whitespace, an
  empty input and a whitespace only input, Spanish, French and German,
  Greek, Cyrillic, Hebrew, Arabic, Japanese, Chinese, Korean, Thai, emoji
  (single, with skin tone, ZWJ sequence, flags, keycap), combining marks,
  characters outside the Basic Multilingual Plane, and two long words (300
  letters, and 254 letters followed by an emoji so that a cut after 255
  UTF-16 units falls inside a pair).
- `inputs/surrogate/`: texts on which a tokenizer model splits between the
  two halves of a surrogate pair. The name starts with the language whose
  model splits it; every model runs every file.
- `inputs/abbreviations/`: texts for the abbreviation dictionary of a
  tokenizer: known abbreviations, units, and a run of dots that the
  dictionary check jumps over.
- `inputs/alphanumeric/mixed.txt`: letters, digits and accented letters
  for the alphanumeric optimization.
- `inputs/subwords/`: 14 texts for the WordPiece encoder's BERT
  normalization: precomposed accents, the same accents as combining
  marks, marks out of canonical order, capital sigma at the end of a
  word, the dotted capital I, Hangul, kana and CJK, CJK Extension E
  ideographs on both sides of U+2B920, format characters, punctuation,
  spacing marks and a virama, emoji and symbols with no piece, special
  token strings, fullwidth letters and ligatures, and words of 100 and
  101 letters.
- `inputs/zh/`: 12 Chinese texts for the unigram segmenter: sentences
  whose segmentation is ambiguous, a news paragraph, Chinese mixed with
  Latin words, digits, a URL path and an e-mail address, Chinese and
  Western punctuation, traditional characters, characters outside the
  Basic Multilingual Plane and emoji, whitespace of every kind (CR LF,
  tab, ideographic, no-break, `U+0085`, `U+001C`, `U+2028`, `U+2029`), a
  long run without punctuation, classical text, Chinese, fullwidth and
  Roman numerals, the fixture lexicon's words (`fixture-words.txt`) and
  the words of the whitespace mode lexicon (`modes.txt`). Two files are
  not new text: `classical.txt` quotes Li Bai's poem 静夜思 and the
  opening of the Analects, both in the public domain, and the sentences
  of `ambiguity.txt` are the stock examples of segmentation ambiguity
  that Chinese word segmentation papers and textbooks repeat.
- `inputs/ja/`: 14 Japanese texts for the lattice tokenizer, written for
  this project: plain sentences, segmentation ambiguity (`東京都`,
  `今日本日`), kanji the lexicons do not list, katakana with the long
  vowel mark, hiragana runs, halfwidth katakana, Japanese mixed with
  Latin, fullwidth, Greek and Cyrillic letters, kanji and Arabic
  numerals, Japanese punctuation and brackets, characters outside the
  Basic Multilingual Plane and emoji, whitespace of every kind (tab,
  ideographic, no-break, `U+0085`, `U+001C`, CR, `U+2028`, `U+2029`), a
  long run without spaces, the fixture dictionary's tied words
  (`ties.txt`) and its surface with a comma (`quoted-surface.txt`).
- `inputs/ko/`: 10 Korean texts for the lattice tokenizer, written for
  this project: sentences with and without spaces between words, words
  the lexicons do not list, Hanja, Korean mixed with Latin words and
  numbers, compatibility and conjoining jamo, punctuation, whitespace of
  every kind, characters outside the Basic Multilingual Plane and a long
  run without spaces.
- `inputs/detokenizer/`: one token per line, each line ending with a line
  feed (an empty file is an empty token list). The part of the name before
  the first `-` names the dictionary, `fixtures/<name>-detokenizer.xml`.
- `inputs/sentence-fixture/`: 17 texts for the two fixture sentence
  models: abbreviations at the start, in upper case, after a bracket or an
  apostrophe; numbers with dots; quotes after the period; `z.B.` style
  runs and an upper case letter after a period; a split whose token runs
  on past it (`hard.Bekannt.`); a lone `.`; a text without a final
  period; whitespace only; CR LF and lone CR; leading and trailing
  whitespace; `U+0085`, `U+2028`, `U+2029`; `U+001C` and `U+001F`; emoji
  inside a sentence; a character outside the Basic Multilingual Plane
  right after a period, also at the very end.

- `inputs/bpe/`: six texts written for OpenNLP's BPE trainer: prose
  (`prose`), words whose pairs tie (`ties`), runs of one letter whose
  pairs overlap (`overlap`), several scripts (`multilingual`), characters
  outside the Basic Multilingual Plane (`emoji`) and words holding `</w>`
  as text (`end-marker`).

`analysis/goldens-tool/README.md` ("Runs") lists which run reads which set.

## Fixtures

`fixtures/` holds the models and dictionaries that are not published
models. They are checked in, and `analysis/goldens-tool/src/recipe.rs`
lists each with its SHA-512; `generate` and `verify` refuse a directory
that differs.

| File | What it is |
|---|---|
| `en-abbrev.bin`, `en-plain.bin` | English sentence models trained by `analysis/sentdetect/tests/fixtures/src/TrainSentenceModel.java` (see the README there): `en-abbrev` has an abbreviation dictionary of 13 entries and `useTokenEnd=true`, `en-plain` has no dictionary and `useTokenEnd=false` |
| `pos-fixture.bin` | a copy of `analysis/sequence/golden/models/pos-fixture.bin`, a part-of-speech model OpenNLP trained by GIS on 15 sentences with Penn Treebank tags, the default descriptor and the case-insensitive tag dictionary `tags.tagdict` (`the`, `Run`, `fell`, `Paris`; `analysis/sequence/README.md`) |
| `lemmatizer-dictionary.txt` | a `DictionaryLemmatizer` dictionary written for this project: 48 `word<tab>tag<tab>lemma` lines with Universal Dependencies tags for words of the `en` and `multi` inputs, among them entries with several lemmas (`leave#leaf`), a word listed twice with one tag (the last wins), a word with two tags, an empty lemma field (`#`), fields after the lemma and a trailing tab (both ignored), entries with a capital letter (never found, since a lookup lowercases the token) and non-ASCII words |
| `chunker-fixture.bin` | a copy of `analysis/sequence/golden/models/chunker-fixture.bin`, a chunker model OpenNLP trained by perceptron on 8 sentences with Penn Treebank tags and `B-`, `I-` and `O` chunk tags for `NP`, `VP`, `PP`, `ADJP` and `SBAR` (`analysis/sequence/golden/inputs/train-chunk.txt`, `analysis/sequence/README.md`); the only chunker model here, since the published chunker models are OpenNLP 1.5 models, which this project does not use |
| `tokenizer-abbreviations.bin` | a copy of `analysis/maxent/tests/fixtures/models/tokenizer.bin`, a tokenizer model OpenNLP trained on six sentences with the case-insensitive dictionary `Dr.`, `U.S.`, `e.g.` (`analysis/maxent/tests/fixtures/README.md`) |
| `english-dictionary.xml` | an abbreviation dictionary written for this project, added to the English tokenizer model |
| `latin-detokenizer.xml` | Apache OpenNLP's test resource of `DictionaryDetokenizerTest`, unchanged (license header kept) |
| `en-detokenizer.xml`, `fr-detokenizer.xml`, `pt-detokenizer.xml` | Apache OpenNLP's English, French and Portuguese detokenizer dictionaries, `opennlp-tools/lang/{en,fr,pt}/tokenizer/`, unchanged (license header kept) |
| `quotes-detokenizer.xml` | two `RIGHT_LEFT_MATCHING` entries, written for this project |
| `zh-unigram-fixture.txt` | a frequency lexicon written for this project: the eight words of OpenNLP's `UnigramSegmenterTest`, pairs of words whose counts have equal products (`研究` 6 `生命` 4 and `研究生` 8 `命` 3, `研究` 6 `生物` 12 and `研究生` 8 `物` 9, `人民` 1 `大会` 148 and `人民大` 2 `会` 74, `a` 3 `bc` 2 and `ab` 2 `c` 3), CR LF and lone CR line ends, an empty and a blank line, a tab, a no-break space and an ideographic space before the count, fullwidth digits and a `+` sign in a count, a leading ideographic space, a word outside the Basic Multilingual Plane, Latin words and a word listed twice |
| `zh-unigram-modes.txt` | a frequency lexicon written for this project whose words depend on the whitespace mode: `U+001C` ends a word in the legacy mode only and `U+0085` in the Unicode mode only |
| `jieba-dict-subset.txt` | the lines of jieba's `jieba/dict.txt` (github.com/fxsjy/jieba, commit `67fa2e36e72f69d9134b8a1037b83fbb070b9775`, MIT license, see below) whose word occurs in an input of the `zh`, `multi` or `edge` set, unchanged and in the order of `dict.txt`; made with the command below |
| `mecab-fixture-ja/` | a MeCab dictionary directory written for this project: two lexicon files (`A-nouns.csv`, `b-others.csv`) of 37 entries with IPADIC-style part-of-speech features, a 5 by 5 `matrix.def`, a `char.def` with overlapping categories, a category without grouping and supplementary ranges, and an `unk.def`. Its costs make ties of each kind the lattice tokenizer breaks, and one surface has a comma in it (quoted, as MeCab quotes it) |
| `mecab-fixture-ko/` | a MeCab dictionary directory written for this project: one lexicon file of 23 entries with mecab-ko-dic-style features, a 4 by 4 `matrix.def`, and a `char.def` and `unk.def` with Hangul, jamo and Hanja categories |
| `mecab-fixture-partial/` | a MeCab dictionary directory written for this project whose `unk.def` has no `DEFAULT` template, so a character that starts no lexicon word and is outside its `KANJI` category has no candidate |
| `all-minilm-l6-v2-vocab.txt` | the WordPiece vocabulary of `testdata/all-minilm-l6-v2/tokenizer.json` (sentence-transformers/all-MiniLM-L6-v2, Apache-2.0), one piece per line in id order with a line feed after each, written with `jq -r '.model.vocab \| to_entries \| sort_by(.value) \| .[].key'`; 30522 pieces, the uncased BERT vocabulary |

`jieba-dict-subset.txt` was written from the repository root with

```
for f in analysis/goldens/data/inputs/{zh,multi,edge}/*.txt; do cat "$f"; echo; done > /tmp/inputs.txt
ALL=/tmp/inputs.txt perl -CSD -ne 'BEGIN { local $/; open my $f, "<:encoding(UTF-8)", $ENV{ALL} or die; $t = <$f> }
  ($w) = /^(\S+)/; print if index($t, $w) >= 0' <jieba>/jieba/dict.txt \
  > analysis/goldens/data/fixtures/jieba-dict-subset.txt
```

and has to be written again, with the goldens, when an input of those
sets changes. jieba's license:

```
The MIT License (MIT)

Copyright (c) 2013 Sun Junyi

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Dictionary notices

The goldens under `lattice/mecab-ipadic-2.7.0-20070801/` hold the
feature columns (part of speech and reading) of the IPADIC entries the
JVM chose for each input, and the dictionary itself is not checked in.
IPADIC's `COPYING` asks that any copy, whole or modified, carry its
notice, which follows as it is in `mecab-ipadic-2.7.0-20070801.tar.gz`:

```
Copyright 2000, 2001, 2002, 2003 Nara Institute of Science
and Technology.  All Rights Reserved.

Use, reproduction, and distribution of this software is permitted.
Any copy of this software, whether in its original form or modified,
must include both the above copyright notice and the following
paragraphs.

Nara Institute of Science and Technology (NAIST),
the copyright holders, disclaims all warranties with regard to this
software, including all implied warranties of merchantability and
fitness, in no event shall NAIST be liable for
any special, indirect or consequential damages or any damages
whatsoever resulting from loss of use, data or profits, whether in an
action of contract, negligence or other tortuous action, arising out
of or in connection with the use or performance of this software.

A large portion of the dictionary entries
originate from ICOT Free Software.  The following conditions for ICOT
Free Software applies to the current dictionary as well.

Each User may also freely distribute the Program, whether in its
original form or modified, to any third party or parties, PROVIDED
that the provisions of Section 3 ("NO WARRANTY") will ALWAYS appear
on, or be attached to, the Program, which is distributed substantially
in the same form as set out herein and that such intended
distribution, if actually made, will neither violate or otherwise
contravene any of the laws and regulations of the countries having
jurisdiction over the User or the intended distribution itself.

NO WARRANTY

The program was produced on an experimental basis in the course of the
research and development conducted during the project and is provided
to users as so produced on an experimental basis.  Accordingly, the
program is provided without any warranty whatsoever, whether express,
implied, statutory or otherwise.  The term "warranty" used herein
includes, but is not limited to, any warranty of the quality,
performance, merchantability and fitness for a particular purpose of
the program and the nonexistence of any infringement or violation of
any right of any third party.

Each user of the program will agree and understand, and be deemed to
have agreed and understood, that there is no warranty whatsoever for
the program and, accordingly, the entire risk arising from or
otherwise connected with the program is assumed by the user.

Therefore, neither ICOT, the copyright holder, or any other
organization that participated in or was otherwise related to the
development of the program and their respective officials, directors,
officers and other employees shall be held liable for any and all
damages, including, without limitation, general, special, incidental
and consequential damages, arising out of or otherwise in connection
with the use or inability to use the program or any product, material
or result produced or otherwise obtained by using the program,
regardless of whether they have been advised of, or otherwise had
knowledge of, the possibility of such damages at any time during the
project or thereafter.  Each user will be deemed to have agreed to the
foregoing by his or her commencement of use of the program.  The term
"use" as used herein includes, but is not limited to, the use,
modification, copying and distribution of the program and the
production of secondary products from the program.

In the case where the program, whether in its original form or
modified, was distributed or delivered to or received by a user from
any person, organization or entity other than ICOT, unless it makes or
grants independently of ICOT any specific warranty to the user in
writing, such person, organization or entity, will also be exempted
from and not be held liable to the user for any such damages as noted
above as far as the program is concerned.
```

The goldens under `lattice/mecab-ko-dic-2.1.1-20180720/` hold the
feature columns of the mecab-ko-dic entries the JVM chose. mecab-ko-dic
is under the Apache License 2.0 (its `COPYING`,
https://www.apache.org/licenses/LICENSE-2.0); its archive has no
`NOTICE` file. The dictionary itself is not checked in.

## Goldens

`<annotator>/<model>/<set>/<id>.golden` is what the OpenNLP annotator
`<annotator>` produced with model `<model>` (the model file's name
without `.bin`, `.xml` or `.txt`) on `inputs/<set>/<id>.txt`. A run with options
has `+` and their names after the model (`+legacy`, `+keep-new-lines`,
`+alphanumeric`, `+english-dictionary`, `+penn`, `+beam1`), and a run without a model names
itself there (`simple`, `whitespace`, `uax29`, `newline`). Each file's
header names the OpenNLP commit, the JDK, the input's hash, each model's
hash and the run's options.

Where a token or sentence starts or ends between the two halves of a
surrogate pair, its row holds the whole character's bytes and the golden
also has `turbo:utf16<opennlp:tokens>` (or `<opennlp:sentences>`) with
the JVM's exact UTF-16 spans. `en/emoji-split`, eight of the
`surrogate` goldens, the `simple` goldens of inputs with characters
outside the Basic Multilingual Plane (`SimpleTokenizer` classifies UTF-16
units) and four of the `tokenizer-abbreviations` goldens have one. A golden whose annotator threw on its input has an `error`
record and no layers; `SimpleTokenizer` with `keepNewLines` throws on a
whitespace only input, and those goldens have one. `docs/analysis.md` ("Generating
golden files") describes both.

Do not edit the goldens by hand. They are regenerated with the driver in
`analysis/goldens-tool` (see its README):

```
cargo run -p turbo-goldens-tool -- generate
cargo run -p turbo-goldens-tool -- verify
```

`generate` builds nothing itself: it needs the image from
`turbo-goldens image` and runs the generator in that container with the
inputs and models mounted read only, then replaces the annotator
directories here with the new output. Adding or changing an input means
running `generate` and checking in the inputs and the goldens together.

## BPE goldens

`bpe/merges-<n>/bpe/<id>.golden` is what OpenNLP's `BPETokenizerTrainer`
learned from `inputs/bpe/<id>.txt` alone with `numMerges` `<n>` (40 and
400), and what `BPETokenizer` with those merges gave for the same text:
the merges in order (`turbo:bpe-merges`, each side a string, a lone
surrogate written `\uXXXX`) and the tokens (`opennlp:tokens`). The run
has no model file; `merges-<n>` names it. The trainer counts symbols in
UTF-16 units, so on `emoji` its merges hold halves of surrogate pairs;
its tokens are whole characters. `analysis/bpe-trainer/tests/opennlp.rs`
compares the Rust trainer and tokenizer with them (docs/bpe-trainer.md).

## Normalizer goldens

`inputs/norm/` holds 61 short texts written for the normalizers: curly
quotes and dashes, line ends and Unicode spaces, hyphenated line breaks,
URLs and e-mail addresses, tweets, numbers, ligatures and normalization
forms, Hangul, German, Greek and Turkish letters, confusables, emoji,
English contractions, lone symbols and a few texts for the differences
recorded in `docs/normalizers.md`. Every character is in the Unicode 15.0
repertoire, which JDK 21 implements. `normalize/<name>/norm/<id>.golden`
holds a `turbo:view` layer with the edits OpenNLP's normalizer `<name>`
made to `inputs/norm/<id>.txt`; see `analysis/goldens-tool/README.md`.

## Number goldens

`inputs/num/` holds 60 short texts written for number scanning: US and EU
grouping and decimals, values that read differently in the two notations
(`1.234`), invalid grouping, repeated separators, `k`/`m`/`bn` suffixes and
`thousand` to `trillion` words with different spaces before them,
exponents, negatives with and without a sign boundary, thin space,
no-break space, apostrophe and Arabic separators between digits, numbers
next to letters, underscores and combining marks, non-ASCII digits (which
are not scanned), 40 digit numbers, zero with a fraction, texts that end
in a separator, and prose in several languages. Every character is in the
Unicode 15.0 repertoire. `numbers/<notation>/num/<id>.golden` holds the
`turbo:numbers` layer for `inputs/num/<id>.txt` in `latin-us` or
`latin-eu`; see `analysis/goldens-tool/README.md`. Only the `numbers`
annotator runs on this set, and it has no dumps.

## Money goldens

`inputs/money/` holds 64 short texts written for money extraction:
currency symbols before and after the amount, ISO codes and English
currency words, `k`/`M`/`bn` suffixes and scale words, negative amounts
and the places a minus sign does not count, US and European notation and
amounts that read differently in each, invalid and Indian grouping,
several amounts per text, near misses (percent, bare numbers, `HK$50`,
letters running into the amount, dates and identifiers) and non-ASCII text
around the amounts (German, French, Russian, Japanese and Chinese, Arabic,
Hindi and Thai, emoji, combining marks, non-ASCII digits).
`conversion-*.txt` hold mixes of currencies for the conversion goldens.

`rates/ecb-snapshot.csv` is the rate table the conversions use: a small
hand made file in the format of the ECB's `eurofxref-hist.csv` (newest
row first, one column per currency, `N/A` for no quote) with columns USD,
JPY, GBP, CHF, CNY, INR and RUB and 12 weekdays from 2026-03-02 to
2026-03-17. The values are invented. The ECB publishes the real series
at <https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html>;
it may be reused with the source acknowledged.

`money/<notation>/money/<id>.golden` and
`money-convert/<notation>/money/<id>.golden` both hold the
layer `turbo:money`; see
`analysis/goldens-tool/README.md`. Only the two money annotators run on
the set, and they have no dumps.

## Quantity goldens

`inputs/quantity/` holds 61 short texts written for quantity extraction:
the default units (length, mass, volume, time, speed, frequency, power,
data) attached to the number or after one space, units that are case
sensitive or read only by their first token, percentages and the word
`percent`, negative values and the places a minus does not count, US and
European notation and grouping, scale markers and exponents that are not
read, amounts that money extraction takes first, and non-ASCII text around
the values. `quantity/<notation>/quantity/<id>.golden` holds the layer
`turbo:quantities`; see `analysis/goldens-tool/README.md`. Only the
quantity annotator runs on the set, and it has no dumps.

## Decision point dumps

`<annotator>/<model>/<set>/<id>.dump`, beside each golden of
`sentdetect`, `tokenize`, `postag`, `lemmatize`, `chunk` and `langdetect` that
names a model and has no `error` record, records every
decision the annotator asked its maxent model for on that input: the
context strings, at the UTF-8 byte offset the decision is about, and the
probability of each outcome the model returned. `docs/analysis.md`
("Decision point dumps") describes the format and how each annotator's
decisions are paired with positions. `generate` writes the dumps
together with the goldens, and the same rule holds: do not edit them by
hand.

The goldens and dumps checked in now were made in the generator container by
`turbo-goldens generate`, in one run, and `turbo-goldens verify`
reproduces them byte for byte. A normalizer asks no model, so the
`normalize` goldens have no dumps; neither do the `numbers` goldens.
