# The tokenizer

`turbo_tokenizer_create` builds the tokenizer a bundle names from the
upstream `tokenizer.json` it carries, and checks it against the bundle's
reference ids before handing it out (docs/bundle.md, loader rule 5). It is
the library's own code: upstream `tokenizers` is only what the tests
compare it with. The same text gives the same ids on every machine.

## Models

| Kind | Models | What it runs |
|---|---|---|
| `wordpiece` | BERT, MiniLM, BGE (English) | BertNormalizer (clean, split CJK, strip accents, lowercase), BertPreTokenizer, greedy longest-match WordPiece. |
| `unigram` | XLM-RoBERTa, BGE-M3, Model2Vec's multilingual models | SentencePiece's precompiled character map, Metaspace, the segmentation of highest total score. |
| `bpe` | GPT-2, RoBERTa | GPT-2's pattern, byte-level characters, merges by rank. |
| `sentencepiece` | T5, mT5, XLM-RoBERTa, Llama 2, any SentencePiece `.model` | Google's sentencepiece library itself: the model's normalizer, then its unigram search or its BPE, with byte fallback. |

`docs/bundle.md` lists each kind's manifest fields and what the tokenizer
file must hold for it.

A `bpe` tokenizer file, or a SentencePiece BPE `.model`, can be trained
from a corpus with `turbo-bpe-trainer` ([bpe-trainer.md](bpe-trainer.md)).

## SentencePiece models

The `sentencepiece` kind reads a SentencePiece `.model` file, the
`ModelProto` the library writes, with a protobuf reader of its own for
the fields it needs, and runs it as the library (version 0.2.2) does
(`core/src/sentencepiece/mod.rs`):

- The normalizer: the user-defined pieces kept as they are, else the
  longest rule of the precompiled character map (a darts-clone double
  array and its replacement strings), else the character as it is;
  then the whitespace switches of the model's `NormalizerSpec`: extra
  whitespace removed, a space in front (or behind, for a model that
  treats whitespace as a suffix), each space written as U+2581. It
  records the input byte each normalized byte came from, the library's
  `norm_to_orig`; `Normalized::orig_to_norm` gives the other way.
- A unigram model: the best path, scores added in f32 and rebased past
  100000 as the library rebases them; a user-defined piece scores
  0.1 per byte after the first; a character no piece starts is the
  unknown piece, scored 10 under the lowest score.
- A BPE model: the adjacent pair whose join is the piece of highest
  score merged first, the leftmost of equal scores, a merged UNUSED
  piece split back into its two parts. When no piece holds a space
  except in the run it starts with, the text is merged word by word,
  which gives the same pieces with a heap the size of a word.
- A run of unknown pieces is one piece; with byte fallback each unknown
  character is instead its UTF-8 bytes' pieces (`<0xE2>`).
- Control pieces are never matched in the text: `</s>` written there
  is its characters. The model's self-test samples are run on load,
  and a model that fails them is refused, as the library refuses it.

`SentencePiece::encode` gives the library's ids and offsets exactly.
`core/tests/sentencepiece.rs` compares the normalized text, its
`norm_to_orig`, the ids, the offsets and the decoded text with the
library's on 131 texts and a 60 KB text for each model of
`testdata/sentencepiece` (T5, XLM-RoBERTa's model rebuilt from
`testdata/bge-m3`, and six small models trained to cover byte fallback,
user-defined and UNUSED pieces, whitespace as a suffix, a custom
character map and a denormalizer), and for mT5, umT5 and Llama 2 when
`TURBO_SENTENCEPIECE_MODELS` names a directory holding them
(testdata/README.md). It checks the same on 143 texts for 48 edits of
five of the committed models: the whitespace switches
(`add_dummy_prefix`, `remove_extra_whitespaces`,
`treat_whitespace_as_suffix`) in every combination, and user-defined
pieces that overlap the model's own pieces and each other.

These differ from the library, each tested there:

- A piece that is not UTF-8 is refused on load; the library takes any
  bytes. A self-test sample whose input is not UTF-8 counts as failed.
- Word and character models are refused: only unigram and BPE are read.
- A model whose denormalizer has a broken character map is refused on
  load; the library loads it and decodes every row to an empty text.
- A BPE model whose CONTROL piece is one character gives that piece's
  id for the character, as the library's encode into ids does, with an
  empty offset; the library's encode with offsets fails on such a text.
- Through the `Tokenizer`, spans follow the rules below rather than the
  library's offsets: the library gives `▁world` the space before it,
  and a byte piece other than the last of its character, or a piece
  made for part of one input character (the `f` of the `fi` a ligature
  becomes), an empty offset at the character's start. Here whitespace
  at the ends is left out and such a piece spans the whole character.
  `analysis/embed/tests/subwords.rs` checks every span against the
  library's offsets under exactly these rules.

## Encoding

`turbo_tokenizer_encode` writes rows of ids, an attention mask and token
types into caller-owned arrays, with the bundle's template of special
tokens, its truncation and its prompt prefixes, unless the call's options
say otherwise. `turbo_tokenizer_count` gives the length of a text's row
without cutting it.

## Spans

`turbo_tokenizer_encode_spans` writes the same arrays, and beside each
token its span: the bytes `[start, end)` of the caller's text it came
from. Spans are what ties a token back to the text, so whatever is worked
out per token (an embedding, a label) can be put back on the text it came
from, however the text was normalized on the way.

- The normalizer records, for each byte it writes, the character of the
  text it came from. A character written beside another (the spaces
  around a CJK character, the marks NFD splits off, a lowercase longer
  than one character, the metaspace or space put in front of a piece of
  text) comes from that character. A token's span runs from the first of
  its bytes' sources to the last.
- Whitespace at either end of a span is left out, unless the token is
  whitespace alone; then its span is that whitespace.
- A special token written in the text is where it is written. A special
  token the template adds, a token of the prompt's prefix, and padding
  are `[0, 0]`; spans are into the text as the caller passed it, without
  the prefix.
- A byte-level token holding part of a character spans the whole
  character, so two tokens can share one.
- Spans are on character boundaries, inside the text, and the spans of
  the text's tokens start in order.

For the `sentencepiece` kind they are the library's offsets under these
rules (above). For the others they are upstream's offsets except where
those are wrong: where the
character map composes a grapheme cluster whole (`e` and a combining
accent into `é`) the span covers the cluster, not the `e` alone; a
byte-level token of whitespace alone spans that whitespace rather than
the empty offset at its end; and the first token of a piece of text a
space was put in front of starts where the text does, not a byte later.
`core/tests/tokenizer.rs` compares every span with upstream's on the test
texts and allows those differences only.

The Rust interface has the same as `Tokenizer::encode_spans`. A row of
ids alone makes no spans and costs nothing for them.

## Decoding

`turbo_tokenizer_decode` writes the text a row of ids stands for, as the
tokenizer file's decoder does, the special tokens left out if asked:

- WordPiece: a piece with the continuing prefix joins the one before it,
  others follow a space, and the space before `.` `?` `!` `,` and English
  contractions is taken out.
- Metaspace: each metaspace is a space, less the one put in front.
- SentencePiece: the library's `Decode`: control pieces left out, the
  unknown piece as the model's unknown surface (` ⁇ `), byte pieces
  read as UTF-8, each U+2581 a space, the model's denormalizer last.
- ByteLevel: each character is its byte; bytes that are not UTF-8 are
  U+FFFD.

A decoder the library does not run is `TURBO_E_UNSUPPORTED`. Decoding
writes text back from ids, such as a model's output; for where an input
token came from, use its span, which decode cannot recover (the
normalizer lowercases, strips accents and folds whitespace).
