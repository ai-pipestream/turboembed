# turbo-tagger

Part-of-speech and entity annotators over a bundle's own token tagger
heads (`manifest.heads`, docs/bundle.md "Heads"), on any backend the
encoder runs on. Heads are trained by `turbo-trainer` (`trainer/`).

- `words`: per-word encoder features. Each word is tokenized on its own;
  a sentence is read in one row, or in overlapping windows of whole words
  when it is longer, and each word's pieces are pooled on the device by
  the range write (first piece, or the mean of its pieces, as the head
  says). Training and inference read words through this one path.
- `annotate`: a head and its encoder (`HeadRunner`): sentences from the
  token layer's sentence column, else the sentence layer, else the whole
  view; scores and decode by the core's `TokenTagger` on the host.
- `PosTagger` writes the part-of-speech layer from a `SCHEME_TAGS` head;
  `EntityTagger` writes the entity layer from a `SCHEME_BIO` or
  `SCHEME_BILOU` head, each span on token bounds.

```text
cargo test -p turbo-tagger
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 cargo test --release -p turbo-tagger --features turbo/levelzero --test device
TURBO_TEST_DEVICE=cuda TURBO_TEST_REQUIRE_CUDA=1 cargo test --release -p turbo-tagger --features turbo/cuda --test device
TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 cargo test --release -p turbo-tagger --features turbo/metal --test device
```

Each backend is a feature of `turbo` (core/Cargo.toml) that this crate does
not turn on, so a device run names it with `--features turbo/<backend>`.
