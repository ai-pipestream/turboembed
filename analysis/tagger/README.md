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
  inside its token; other subwords of the model are passed over.
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
  (any `Fn(form, tag) -> Option<String>`, so a Morfologik or WordNet
  Morphy lookup as a closure) overrides the head where it lists a lemma;
  its id is part of the config hash.

```text
cargo test -p turbo-tagger
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 cargo test --release -p turbo-tagger --features turbo/levelzero --test device
TURBO_TEST_DEVICE=cuda TURBO_TEST_REQUIRE_CUDA=1 cargo test --release -p turbo-tagger --features turbo/cuda --test device
TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 cargo test --release -p turbo-tagger --features turbo/metal --test device
```

Each backend is a feature of `turbo` (core/Cargo.toml) that this crate does
not turn on, so a device run names it with `--features turbo/<backend>`.
