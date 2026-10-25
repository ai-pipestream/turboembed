# Text quality

`turbo_analysis_layers::{TextQuality, SpanQuality}` retain view identity and
finite `f32` metrics through the standard document graph codecs.
`turbo_normalize::TextQualityAnnotator` (also exported from
`turbo_normalize_layers`) reads a view, Tokens, Sentences and optional Structure.
It writes one whole-view summary and disjoint region summaries. These are
deterministic triage signals, not calibrated probabilities or model accuracy.

## Counting and reference definitions

Words are input tokens containing a Unicode letter or decimal digit. Punctuation
tokens remain in the information-density denominator. Alphabetic content tokens
contain at least one letter and otherwise only letters or combining marks, and
are absent from the existing OpenNLP English stopword list. Token spelling is
lowercased using the shared Java simple-character mapping, without normalization
or accent folding. Discontinuous token spellings concatenate their fragments.

MTLD uses the 0.72 TTR threshold, counts complete sequential factors and the
fractional remainder `(1 - TTR) / (1 - 0.72)`, and averages forward and reverse
values. This follows the method of
[McCarthy and Jarvis (2010)](https://doi.org/10.3758/BRM.42.2.381).
For an empty sequence it is zero. For an all-unique sequence the partial factor
is zero, so its undefined quotient is represented by the observed word count.
Short-text MTLD is not evidence of reliable lexical diversity; fewer than twenty
words sets `TOO_SHORT`. The score does not promise length invariance on such input.

For W words, S sentences, Y syllables, L letters and D decimal digits:

* Flesch-Kincaid grade = `0.39 W/S + 11.8 Y/W - 15.59`.
* ARI = `4.71 (L+D)/W + 0.5 W/S - 21.43`.
* Coleman-Liau = `5.88 L/W - 29.6 S/W - 15.8`.

Sources: [Kincaid et al. (1975)](https://stars.library.ucf.edu/istlibrary/56/),
[Smith and Senter (1967)](https://apps.dtic.mil/sti/citations/AD0667273),
and [Coleman and Liau (1975)](https://doi.org/10.1037/h0076540).
The grades are English readability heuristics and may be negative; they are
not clipped to a school-year range. Syllables count runs of ASCII `aeiouy` after
simple lowercasing, with a minimum of one for a token containing a letter.
There is no pronunciation dictionary or silent-e correction. Numeric-only words
have zero syllables. These input policies affect the indices: formula correctness
does not imply human syllable-count or readability accuracy. With words but no
sentences, S is one; with no words all three indices are zero.

Character ratios count Unicode scalar values rather than UTF-8 bytes. Alpha
includes letters and combining marks, digit includes decimal digits, whitespace
uses Java whitespace plus Unicode separator categories, and symbol contains all
remaining values. These four ratios partition nonempty text. Replacement
characters are a subset of symbol, also exposed separately. ARI and Coleman-Liau
count actual letters, excluding combining marks.

## Regions and complexity

With Structure, each nonempty continuous fragment of a leaf block gets a row;
section and other containing blocks are not rescanned, and fragment gaps do not
become covered text. Without nonempty Structure, blank lines delimit
paragraphs, including CRLF lines. Uncovered text still contributes to the
whole-view summary. A token contributes to the region containing its start;
even an overlapping token contributes to at most one region. Sentences touching
a region contribute to its sentence count. This policy avoids duplicating long
cross-boundary tokens or expanding nested blocks quadratically.

After boundary discovery, token spellings are scanned once and view characters
are scanned once. MTLD and trigrams use cached token IDs, not rescanned strings.
Arbitrary token order is handled by stable four-pass radix sorting of byte starts.
The work is O(bytes + tokens + sentences + blocks), with expected constant-time
hash lookups. FNV-1a hashes index retained lowercase spellings; collisions are
resolved by comparing spellings. Scratch owns reusable hash tables, arenas,
epoch marks, region summaries and sorting buffers. The measurement API borrows
its rows from Scratch. The pipeline adapter uses thread-local Scratch and
allocates its owned output layers separately. Structure validation can allocate;
the zero-allocation test covers warmed measurement without Structure.

## Flag and composite-score policy (version 1)

* Any U+FFFD or C1 control sets `MOJIBAKE_SUSPECT`.
* Two or more line-final letter-hyphens followed by lowercase continuation,
  at a density of at least 0.02 per word, set `OCR_BROKEN_HYPHENS`.
* With at least eight non-whitespace characters, symbol density above 0.30
  sets `HIGH_SYMBOL_NOISE`; decimal digit density above 0.50 sets
  `HIGH_DIGIT_DENSITY`.
* At least twelve words and repeated-trigram occurrences at or above 0.50
  of all trigram windows set `REPETITIVE_NGRAM`.
* At least two pipes, at most five words per pipe-delimited fragment on
  average, and no `.`, `!` or `?` set `NAV_BOILERPLATE`.
* Fewer than twenty words sets `TOO_SHORT`.

The composite starts at `0.60 * nonspace_alpha_ratio + 0.25 * information_density
+ 0.15 * min(MTLD/50, 1)`. It is multiplied by 0.25 for mojibake, 0.75 for broken
hyphens, 0.4 for symbol noise, 0.6 for digit density, 0.5 for repetition, 0.3 for
navbar and 0.75 for short text, then clipped to [0,1]. No words yields zero.
These constants are explicit heuristics for triage, not learned thresholds.
Changing this policy requires an annotator version bump.
