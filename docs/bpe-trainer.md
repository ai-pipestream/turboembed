# BPE trainer

`turbo-bpe-trainer` (`analysis/bpe-trainer/`) learns a byte pair
encoding from a corpus and writes it in a form the library reads. It is
a tool run before a model is used, never part of encoding. It trains
three kinds of BPE, each after a reference trainer:

| Command | Reference | Writes | Read by |
|---|---|---|---|
| `byte-level` | upstream `tokenizers` 0.23.2, `BpeTrainer` with the `ByteLevel` pre-tokenizer and alphabet (GPT-2, RoBERTa) | `tokenizer.json` | the core's `bpe` tokenizer ([tokenizer.md](tokenizer.md)), and upstream |
| `sentencepiece` | the sentencepiece library 0.2.2, `model_type=bpe` with the `identity` normalization | `.model` | the core's `sentencepiece` tokenizer, and the library |
| `opennlp` | OpenNLP's `BPETokenizerTrainer` (ai-pipestream/opennlp at `cdf33c653d5d`) | one merge per line, `left right` | `OpenNlpTokenizer` in the crate, a port of OpenNLP's `BPETokenizer` |

```text
turbo-bpe-trainer byte-level <corpus> <tokenizer.json> --vocab-size <n>
    [--min-frequency <n>] [--special <token>]... [--lstrip <token>]...
    [--add-prefix-space] [--roberta <cls> <sep>]
turbo-bpe-trainer sentencepiece <corpus> <model> --vocab-size <n>
    [--character-coverage <c>]
turbo-bpe-trainer opennlp <corpus> <merges.txt> --merges <n>
    [--ties jvm|made-first]
```

The corpus is UTF-8, one text per line, split at line feeds only (a
carriage return stays in the text, as the sentencepiece library reads a
file). For `byte-level` this is upstream's `train_from_iterator` over
the lines; upstream's `train` on files keeps each line's ending in the
text, which gives other words.

## The merge loop

The three trainers share one loop
(`analysis/bpe-trainer/src/merger.rs`). The words of the corpus are kept
once each, with how often they occur, and the count of every pair of
adjacent pieces is kept up to date as merges are made, never counted
again: a merge visits only the words that hold its pair, and in each
only the pieces on either side of each place it is merged. The next pair
comes from a binary heap whose entries may be out of date; an entry is
pushed when a pair's count goes up, and one whose count went down is
pushed again at its current count when it reaches the top. A test
(`analysis/bpe-trainer/tests/merger.rs`) runs the loop against a trainer
written plainly, which recounts every pair before each merge, on 400
random corpora under two orders, with pairs the order refuses and pairs
set aside, and checks that both make the same merges with the same
counts.

Of pairs with the same count, each trainer gives an order, so the same
corpus and settings always give the same model. For byte-level,
sentencepiece and opennlp's `made-first` order, that model does not
depend on the order of the texts either (`training_is_deterministic...`
and `made_first_is_deterministic...` in the trainers' tests):

- byte-level: the pair whose left piece has the smaller id, then the
  one whose right piece has, which is upstream's order;
- sentencepiece: the pair whose joined piece is shorter in characters,
  then the one that comes first in code point order, which is the
  library's; two pairs that make the same piece, the one tie the
  library leaves to the layout of its heap, go to the smaller ids;
- opennlp: by default the JVM's order (below), which depends on the
  order of the texts as the JVM's does; with `--ties made-first`, the
  pair whose left symbol was made first, then the one whose right symbol
  was, the symbols a word starts as counting as made in Java's order of
  their strings.

## Byte-level

Each text is cut into words by GPT-2's pattern, the core's own
`turbo::bpe::split`, with a space put in front first when
`--add-prefix-space` is given and the text does not start with one, and
a word's bytes are written as GPT-2's characters. The vocabulary starts
with the special tokens in the order given, then the 256 byte
characters in code point order; each merge adds the two pieces joined.
Training stops when the vocabulary has `--vocab-size` pieces, when no
pair is left, or when the most frequent pair occurs fewer than
`--min-frequency` times. The file is written in upstream's layout, with
RoBERTa's post-processor when `--roberta` is given.

Compared with upstream (`analysis/bpe-trainer/tests/byte_level.rs`):

- `testdata/tiny-bpe/tokenizer.json`, which upstream trained, is written
  again byte for byte (939 merges, 28901 bytes) from its corpus,
  `testdata/tiny-bpe/corpus.jsonl`.
- Upstream is run in the test on three corpora (the tokenizer texts,
  the BPE golden inputs, the lines of `docs/analysis.md`) at four sizes
  and four settings of minimum frequency, prefix space and special
  tokens: 48 runs, each the same file byte for byte. 1000 random corpora
  of one to three letters, where merges overlap and counts tie: the same
  file each time.
- A trained file loads in the core's tokenizer; every text gives
  upstream's ids for the same file and decodes as upstream decodes it,
  which is the text itself unless a space was put in front or `<mask>`
  took the whitespace before it; each word of the corpus encodes to the
  pieces training left it as.
- Informal, not a committed test and not reproducible from the
  repository: on a 15.3 MB corpus of source and text files at 32000
  pieces, the file was upstream's byte for byte.

Where it differs: there is no `limit_alphabet`, `initial_alphabet`,
`continuing_subword_prefix`, `end_of_word_suffix` or
`max_token_length`. The alphabet is always the 256 byte characters,
which the core's encoder needs, and the core reads no prefix or suffix.

## SentencePiece

Lines that are empty, longer than 4192 bytes or hold U+2585 (the
library's mark for an unknown character) are left out. A line is
normalized as the identity rule does (spaces at the ends dropped, runs
of spaces made one, each space `▁`, a `▁` in front and none at the end),
and `<unk>`, `<s>`, `</s>` and tabs in it become boundaries no piece
crosses. The most frequent characters are kept until
`--character-coverage` (default 0.9995) of all characters is covered;
the rest are unknown. The words start at each `▁`. A pair is never
merged when its piece would hold an unknown character, NUL or a tab,
hold `▁` other than at its start, mix two Unicode scripts (Hiragana,
Katakana and U+30FC count as Han, a combining mark takes the script
before it, a character in no script is Common) or be longer than 16
characters; a pair whose piece was already made is set aside. When no
pair occurs any more and the vocabulary is not full, the library goes on
with the pairs that occurred at some point and no longer do, at count
0, in the same order; so does the trainer. The model's pieces are
`<unk>`, `<s>`, `</s>`, the merged pieces with the score minus their
rank, then the kept characters, most frequent first. A size the corpus
cannot fill, or too small for the meta pieces and the characters, is an
error that names the largest or least size, as the library's does.

Compared with the library (`analysis/bpe-trainer/tests/sentencepiece.rs`):

- `analysis/bpe-trainer/testdata/sentencepiece.py` ran the library
  (Python package 0.2.2) offline and wrote `sentencepiece.jsonl`: 60 runs
  on four corpora (the tokenizer texts, the tiny-bpe corpus, the BPE
  golden inputs, and `edge.jsonl`, lines written for the corners above)
  at three coverages and several sizes, among them the largest each
  corpus allows. Every piece, score and type is the library's, and each
  error has the library's numbers. `sentencepiece-random.jsonl` holds
  300 random corpora at the largest size the library allows: the same
  pieces each time.
- A trained model loads in the core as a BPE model; its normalization
  is the core's, every line of the corpus that holds no boundary
  decodes from its ids to its normalized text, and characters the
  coverage leaves out encode to `<unk>`.
- Informal, not a committed test: on the 15.3 MB corpus above, at 16000
  pieces with coverage 0.9995, 8000 with 1.0 and 30000 with 0.98, the
  pieces, scores and types were the library's.

Where it differs: only the identity normalization is run (a corpus to
be normalized otherwise, such as with `nmt_nfkc`, must be normalized
first); the other trainer options are the library's defaults and cannot
be set; scripts are current Unicode's, where the library's table is of
Unicode 9 and a character assigned since is Common to it
(`scripts_are_current_unicodes`, which gives the library's pieces for
two such corpora); and the same-piece tie above, which did not change a
model on any corpus tested.

## OpenNLP

Each text is cut into words at whitespace as OpenNLP's
`WhitespaceTokenizer` cuts it; a word is its characters with `</w>`
joined to the last, and the most frequent pair is merged in every word,
`--merges` times or until no pair is left. `OpenNlpTokenizer` applies
merges as `BPETokenizer` does: in each word the pair of lowest rank is
merged everywhere, left to right, until no pair has a rank, and `</w>`
is taken off the end of each token.

Of pairs with the same count, the JVM merges the first in the iteration
order of the `HashMap` it counted them in. That order is the pairs' Java
hash codes, spread and masked by the table's size, and within a bin the
reverse of the order the pairs were first counted, which is the order of
another `HashMap` of the words, rebuilt after every merge. The default
order (`--ties jvm`, `TieOrder::Jvm`) follows the JVM's loop step for
step (`analysis/bpe-trainer/src/opennlp/jvm.rs`): it works out the
order of each of the JVM's three maps from the hash codes of the words,
the symbol lists and the pairs, with the table growth of `put` and of
`merge` (which also grows a full table when the key is already there),
and without building the maps (`analysis/jvm/src/hash_map/mod.rs`).
Like the JVM, it counts every pair again before each merge, so a merge
costs a pass over the distinct words. `--ties made-first` keeps the
counts as the shared merge loop does instead, and its order does not
depend on the order of the texts.

Two cases are not reproduced, and `OpenNlpBpe::is_jvm_exact` (and a
warning from the command) says when either can apply:

- A bin of eight or more keys in a table of 64 or more becomes a
  red-black tree in the JVM. The tree's order is reproduced, except
  between keys with equal hash codes: the pairs (a record) and the
  symbol lists are not classes `HashMap` compares, so the JVM orders
  them by identity hash code, which changes from run to run. The words
  are strings, which the JVM compares, but the trainer works from their
  hash codes alone and reports equal ones in a tree bin the same way.
  The order changes the merges only when two pairs in that bin tie for
  the highest count.
- A symbol is a Unicode character, where the JVM's is a UTF-16 unit: the
  JVM splits a character outside the Basic Multilingual Plane into its
  surrogates and spends merges joining them. Such a merge has a lone
  surrogate on one side, which a Rust string cannot hold and which the
  JVM writes as `?` when it saves the model (its serializer encodes
  UTF-8), so a UTF-16 mode would make merges the JVM itself cannot save
  and read back. The trainer does not split.

The JVM goldens are the `bpe` annotator of the goldens generator
(`analysis/goldens/data/bpe/merges-{40,400}/bpe/`): `BPETokenizerTrainer`
on each of the six texts of `inputs/bpe/` with 40 and with 400 merges,
its merges in order (`turbo:bpe-merges`) and the tokens `BPETokenizer`
gave for the same text (`opennlp:tokens`). Compared with them and with
the JVM on random corpora (`analysis/bpe-trainer/tests/opennlp.rs`):

- The trainer makes all 686 of the JVM's merges on the ten goldens whose
  symbols are whole characters, 623 of them made among tied pairs.
- `analysis/bpe-trainer/testdata/OpenNlpRandom.java` ran
  `BPETokenizerTrainer` (the commit above, JDK 21) on 160 random corpora
  of small alphabets and up to 3000 words, and wrote
  `opennlp-random.jsonl`: the trainer makes the same 9907 merges. Six of
  the corpora reach a bin that became a tree; five of them hold keys
  with equal hash codes in it, which the trainer reports, and their
  merges are the JVM's too.
- A model of the JVM's loop with explicit `HashMap` bins, written in the
  test, makes all 748 of the JVM's merges on the twelve goldens, the
  emoji golden's surrogate merges among them.
- Every merge the JVM made is a pair of the highest count when it is
  made in turn here, and with `made-first` the trainer makes the JVM's
  merges up to the first tie.
- Given the JVM's merges, the tokenizer gives the JVM's tokens on every
  golden, words holding `</w>` as text among them.
