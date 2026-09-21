# Native BERT tokenizer compatibility

The native tokenizer must produce the model's token IDs before its output can
be used for numerical or performance comparisons. The previous implementation
accepted vocabulary-shaped JSON without validating the tokenizer and used
handwritten Unicode substitutions. Those behaviors are correctness defects.

The repair preserves the C entry points, layouts, and caller-owned output
buffers. `wordpiece_vocab_load_dir` retains its documented TXT-first order.
An explicit `vocab.txt` load selects the uncased BERT preset. Model providers
must instead load the model's own `tokenizer.json`, validate its configuration,
and reject unsupported native tokenizers or use their existing reference path.
A vocabulary from another model is never a fallback for missing metadata.

The initial native JSON contract is uncased BERT WordPiece with text cleaning,
Chinese character splitting, canonical accent removal, BertPreTokenizer,
`[UNK]`, `##`, and a 100-character word limit. Supported BERT postprocessing
uses `[CLS] A [SEP]` and `[CLS] A [SEP] B [SEP]`. Standard added special tokens
are matched before normalization. Different preprocessing or added-token rules
must be rejected until implemented and verified.

The caller's sequence length controls right truncation and padding. WordPiece
segmentation completes for each word before taking its output prefix, so a
later failed subpiece still makes the whole word unknown. Overlong words emit
one unknown token. Invalid UTF-8 returns `INVALID_ARGUMENT`. JSON parsing and
native allocation failures are contained at the C boundary. Load-time parsing
may allocate; inference uses a frozen vocabulary and bounded stack scratch.

Exact IDs, masks, and type IDs are compared with the pinned Hugging Face
`tokenizers` dependency using a synthetic vocabulary that distinguishes Unicode
normalization, literal special tokens, and subword truncation. Model validation
also uses the actual model tokenizer. This is a compatibility correction, not
a new ABI revision or a claim of general tokenizer support.
