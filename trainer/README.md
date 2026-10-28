# turbo-trainer

Trains bundle heads in Rust, with no Python. Stage R1 (#1518) trains
heads on a frozen encoder: a head learns from the per-word features the
encoder gives for a corpus, and writes exactly the tensors the core
reads (`turbo::head::tagger::tensors`, docs/bundle.md "Heads"), so a
head held in memory and the same head loaded from a bundle score
identically.

What is here:

- `linear`, `lstm`, `crf`, `xent`: forward and backward of each op,
  generic over `Real` (f32 for training, f64 for gradient checks). The
  LSTM follows the core's numerics (gates i, f, g, o; zero initial
  state; a bidirectional row is [forward, backward]). The CRF trains over
  the lattice the scheme allows, the one decode uses, and refuses a gold
  sequence the scheme forbids.
- `adamw`: AdamW with global-norm clipping, linear warmup and linear
  decay, decoupled weight decay on matrices only.
- `tagger`: a `HEAD_TOKEN_TAGGER` in training, seeded init, loss and
  gradients on one sequence, export to the core's `TokenTagger`. With a
  `hash_embed` it reads each word's text, hashed as the core hashes it,
  and learns the four feature tables (an `Example`'s `forms`).
- `adapter`: the BiLSTM adapter forward and back, shared by both heads.
- `parser`: a `HEAD_BIAFFINE_PARSER` in training (#1514): the core's
  forward (root vector, LeakyReLU MLPs, arc and relation biaffines), the
  arc cross-entropy over every head but the word itself plus the
  relation cross-entropy under the gold head, export to the core's
  `BiaffineParser`. `fit_parse` trains it on the same loop, dev scored by
  labelled attachment through the core's decode.
- `fit`: the loop. Seeded shuffles, minibatch gradients summed in a
  fixed order, dev scoring through the core's own inference path
  (accuracy, or exact span F1 under BIO and BILOU), early stopping that
  keeps the best weights. `EarlyStop` is armed only after warmup, and a
  dev score within `tolerance` of the best holds the run going; the
  defaults are 60 epochs, patience 6, tolerance 0.001. An equal score
  holds under that default, so a run whose dev score sits flat goes on
  to its epoch cap; a tolerance of 0 is classic patience, where only a
  better score holds and a flat run stops. A batch's
  examples run on `threads` threads (0, the default, takes every core),
  each into a gradient buffer of its own, and the buffers are summed
  element by element in example order, so the weights do not depend on
  the thread count; `verbose` logs each epoch's loss, dev score, seconds
  and words a second. Two runs with one seed give identical bytes.

Corpora, scoring, receipts and the `turbo-train-head` command are in
`trainer/recipes` (R1b). Not yet here: a feature cache on disk, bundle
output; the edit-tree head (R1c); encoder fine-tuning (R2, #1519);
distillation (R3, #1520).

```text
cargo test -p turbo-trainer
```
