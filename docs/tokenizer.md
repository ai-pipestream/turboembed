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

`docs/bundle.md` lists each kind's manifest fields and what the tokenizer
file must hold for it.

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

They are upstream's offsets except where those are wrong: where the
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
- ByteLevel: each character is its byte; bytes that are not UTF-8 are
  U+FFFD.

A decoder the library does not run is `TURBO_E_UNSUPPORTED`. Decoding
writes text back from ids, such as a model's output; for where an input
token came from, use its span, which decode cannot recover (the
normalizer lowercases, strips accents and folds whitespace).
