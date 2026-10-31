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
  keeps the best weights. `EarlyStop` is armed only after warmup, and
  only a dev score more than `tolerance` above the last one that counted
  resets patience (Keras's `min_delta`); the defaults are 60 epochs,
  patience 6, tolerance 0.001. A run whose dev score sits flat or creeps
  by less than that stops after `patience` epochs, keeping its best
  weights; a tolerance of 0 counts any better score. A batch's
  examples run on `threads` threads (0, the default, takes every core),
  each into a gradient buffer of its own, and the buffers are summed
  element by element in example order, so the weights do not depend on
  the thread count; a batch runs in rounds of `lanes` examples (0, the
  default, takes the whole batch), each round added to the sum in turn,
  so memory is bounded by the lanes and the weights do not depend on
  them either (gradient accumulation); `loss_scale` turns on dynamic
  loss scaling (`LossScale`: the loss gradient is scaled before backward
  and the gradients unscaled before the step, a step whose gradients are
  not finite is skipped and backs the scale off, a run of finite steps
  grows it; `Report::skipped_steps` counts the skips), which a model
  takes through `Model::loss_at` with the example's place in the run;
  `verbose` logs each epoch's loss, dev score, seconds and words a
  second. Two runs with one seed give identical bytes, and a scale that
  stays a power of two gives the unscaled run's.

- `encoder`: stage R2 (#1519), a ModernBERT encoder in training. Its
  ops, each forward and backward and generic over `Real`: LayerNorm with
  and without a bias (`norm`), rotary positions from the core's own
  cos/sin table (`rope`), the gated GELU MLP with an erf to f64
  rounding (`gated`, `math`), softmax attention with the sliding window
  of a local layer (`attention`), which recomputes its probabilities in
  backward rather than keeping heads x n^2 of them a layer, the
  bias-free projections (`matmul`) and low-rank adapters on them
  (`lora`). `Encoder` loads a ModernBERT bundle's F32 weights, runs one
  sequence forward with the core's numerics (held to the sealed tiny
  ModernBERT reference within the conformance tolerance), and backward
  to every weight; `with_lora` freezes the base and trains adapters on
  every layer, `with_lora_top` on the top layers alone, and a frozen
  slot holds no gradient buffer; `merged` folds them back into weights a
  bundle carries. `pool` turns subword states into words as the core's
  word reader does. `tuned` trains the encoder under a token tagger
  head on the shared loop: the head's loss flows through the pooling
  into the encoder's trainable weights, dev is scored through the core's
  tagger over the encoder's states, and `fit_tuned` takes the same
  `Config` as `fit`, threads and lanes included. `with_dropout` sets
  ModernBERT's three dropout rates (after the embedding norm, on
  attention's output, on the MLP's activated hidden) and a seed; a
  training forward (`forward_at`, which the loop uses) drops under
  masks keyed by the seed, the example's place in the run (its epoch and
  index), the layer and the branch, so the same place gives the same
  mask on any thread or lane and backward redraws it rather than keeping
  it; `forward`, dev scoring and inference drop nothing. `with_bf16`
  computes in bfloat16 emulated on the single (`Compute::Bf16`): the
  weights are read from a shadow rounded to bfloat16 that the single
  master weights refresh on every step, every op's output and every
  activation's gradient is rounded to bfloat16 (`Real::bf16`, ties to
  even, straight from the type's own bits), attention's probabilities
  included, while the sums inside each op, the weight gradients and the
  optimizer stay single; the states it gives sit within 2 percent of the
  single's over two layers and the gradients within 0.2 percent in norm
  on the tests' geometry, and the toy task is learnt under it with loss
  scaling on. One sequence a thread, with no sum split: the reference
  the device is held to. `cuda` (the `cuda` feature, nvcc; or
  `cuda-shim`, the same kernels as plain C++ on the host, which CI
  runs) is the encoder on a CUDA device: `CudaEncoder` takes an
  `Encoder<f32>`, holds its weights (rounded under bfloat16), the
  activations of one forward and the gradients on the device, runs the
  ops of `trainer/cuda/train.cu` in the CPU encoder's order, draws
  dropout's masks from the same keys, and sums the gradients on the
  device over as many sequences as the caller runs; `grads` brings
  them to the host at once. Each output element is owned by one
  thread and summed in a fixed order, so a device gives the same bytes
  every run; against the CPU the states and gradients sit within 1e-4
  of their largest (1e-6 measured through the shim), bfloat16 within
  3e-2. docs/cuda.md, "The trainer's kernels", says how it is built
  and tested on a GPU.

Corpora, scoring, receipts and the `turbo-train-head` command are in
`trainer/recipes` (R1b). Not yet here: a feature cache on disk, bundle
output; the edit-tree head (R1c); the rest of R2 (the device encoder
under `Tuned` and the loop, with the step on the device; a tiled GEMM
in place of the register-tiled one; checkpointing of the other
activations; full fine-tuning runs on a corpus and bundle output,
#1519); distillation (R3, #1520).

```text
cargo test -p turbo-trainer
```
