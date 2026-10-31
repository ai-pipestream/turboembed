# turbo-tagger

Part-of-speech, entity, dependency and lemma annotators over a bundle's own
heads (`manifest.heads`, docs/bundle.md "Heads"), on any backend the
encoder runs on. Heads are trained by `turbo-trainer` (`trainer/`).

- `words`: per-word encoder features. Each word is tokenized on its own;
  a sentence is read in one row, or in overlapping windows of whole words
  when it is longer, and each word's pieces are pooled on the device by
  the range write (first piece, or the mean of its pieces, as the head
  says). Training and inference read words through this one path. A
  document's `Subwords` are read in place of tokenizing when they are
  this model's, made per word token by a WordPiece tokenizer, each piece
  inside its token and each token on BERT pre-tokenizer word bounds (a
  token inside a word would get continuation pieces it does not get on
  its own); other subwords of the model are passed over.
- `annotate`: a head and its encoder (`HeadRunner`): sentences from the
  token layer's sentence column, else the sentence layer, else the whole
  view, each token in exactly one sentence (a token across a sentence
  boundary is refused); scores and decode by the core's `TokenTagger` on the host.
- `PosTagger` writes the part-of-speech layer from a `SCHEME_TAGS` head;
  `EntityTagger` writes the entity layer from a `SCHEME_BIO` or
  `SCHEME_BILOU` head, each span on token bounds.
- `parse`: `DependencyParser` writes the dependency layer from a
  `HEAD_BIAFFINE_PARSER` head: arcs scored by the core's `BiaffineParser`,
  one tree a sentence by MST or Eisner as the head says, a relation for
  each word under its head. A word under the root has no head.
- `lemma`: `Lemmatizer` writes the lemma layer from a
  `HEAD_EDIT_TREE_LEMMATIZER` head: with a `pos_input`, the tagger head
  it names tags each sentence first (on its own pooling); the core's
  `EditTreeLemmatizer` scores the trees and applies the best one that
  fits each word, the word itself when none fits. A `LemmaDictionary`
  (any `Fn(form, tag) -> Option<String>`) overrides the head where it
  lists a lemma; its id is part of the config hash.
- `dictionaries`: the lexicon lookups as `LemmaDictionary`
  implementations: `MorfologikLemmas` over a Morfologik dictionary (the
  smallest lemma it lists under the tag, in byte order) and `MorphyLemmas`
  over WordNet's Morphy on any lexicon it reads, Princeton's database or
  an Open English WordNet LMF file (its first candidate under the tag's
  part of speech). Both ask under the tag the head decoded, mapped by an
  optional tag map, and give no override without one.
- `segment`: `SentenceTagger` writes the sentence layer from a
  `HEAD_SENTENCE_BOUNDARY` head, reading the view's tokens as one
  sequence of words (in windows past a row) and ending a sentence after
  each word the core's `SentenceBoundary` marks, the last word always.
  `WordTagger` writes the syntactic words layer (`Words`) from a
  `HEAD_TOKEN_SPLIT` head, keeping a token 1-to-1 on `_`, recording byte
  subspans when the split words tile the surface token, and recording
  explicit forms on fusional contractions.
  `SpanTagger` writes typed spans from a `HEAD_SPAN_CLASSIFIER` head into
  the entity layer, nested or flat as the head decodes, a sentence at a
  time; each span's score is its label's probability among the span's
  scores. Both sentence and span taggers put every span on token bounds.
- `relation`: `RelationTagger` writes the relation layer and its
  entity-to-entity edges from a `HEAD_RELATION_CLASSIFIER` head, over the
  entities another annotator wrote. An entity is a mention when its first
  and last tokens are in one sequence and, for a head with
  `relation.entity_labels`, its kind is one of them by name; each kept
  pair covers both entities' spans. A pair may hold several relations,
  each label that scores above `none`; each one's score is the logistic
  of its margin over `none`.
- `multi`: `MultiTask<W>` writes several of those layers from one
  encoder pass (#1517). `W` is a tuple of the layers to write, such as
  `(Sentences, Words, PosTags, Entities, Dependencies<Tokens>, Lemmas)`,
  each from the head named for it or the one head that writes its layer;
  two outputs of one layer name their heads. Each window of words is read
  once, and every pooling the heads ask for is taken from that read, so
  several heads cost one encoder pass and one host decode per head.
  Without a sentence head (or when prior sentences are given), each layer
  is written from the same per-sentence features as its single-head
  annotator (the lemmatizer without a dictionary), and `Sentences` echoes
  the given sentence sequences on token bounds. When `Sentences` is in `W`
  and no prior sentences are given, its `HEAD_SENTENCE_BOUNDARY` head
  decodes sentence boundaries first from the whole-view pass and cuts the
  other heads' sequences and features by those decoded sentences (so their
  token states carry cross-sentence context from that single whole-view
  pass rather than running a separate encoder pass per sentence).
  `license_tier` is the bundle's tier, the strictest class of the encoder
  (`model.license_class`) and every head's training corpora
  (`heads[].license_class`).
- `examples/multitask_cost.rs`: the cost of one pass with five heads
  against one head, on any bundle with an encoder (the heads are seeded
  and added to a copy), in the receipt form.

```text
cargo test -p turbo-tagger
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 cargo test --release -p turbo-tagger --features turbo/levelzero --test device
TURBO_TEST_DEVICE=cuda TURBO_TEST_REQUIRE_CUDA=1 cargo test --release -p turbo-tagger --features turbo/cuda --test device
TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 cargo test --release -p turbo-tagger --features turbo/metal --test device
```

Each backend is a feature of `turbo` (core/Cargo.toml) that this crate does
not turn on, so a device run names it with `--features turbo/<backend>`.
