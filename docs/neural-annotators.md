# Neural annotators on the embedder's runtime

Two neural annotators were asked for in #391: zero-shot named entities
with GLiNER, written as `opennlp:entities`, and sentence boundaries with
SaT (Segment any Text, the `wtpsplit` package), written as
`opennlp:sentences`. Both are encoders with a small head, and the
question is whether they can run on turboembed's own runtime, with no
ONNX Runtime and no new dependency, and whether they can share the
embedding pass.

The answer this page records:

- SaT's `-sm` checkpoints need nothing the runtime lacks. Their encoder
  is XLM-RoBERTa, which the core runs as `FAMILY_ROBERTA`; the head is
  one linear layer. The `turbo-sat` crate (`analysis/sat/`) is that
  annotator, on the CPU, compared with wtpsplit itself on a small model
  with random weights (below); no real checkpoint has been run yet.
- GLiNER does not. Every GLiNER checkpoint with a permissive licence is
  built on DeBERTa-v3, whose attention the core does not have, and its
  head is a BiLSTM and span scorer the analysis crates do not have.
  What is missing is listed below; it is not built in this change.
- Neither can read the states of the embedding pass: each checkpoint
  fine-tuned its own encoder, so its head is only right on its own
  encoder's states.

## What the runtime gives an annotator

The core encodes rows of token ids and returns pooled vectors. An
annotator that needs a state per token asks for ranges of one token:
`turbo_embed_write_ranges` with `POOLING_MEAN` over `[t, t + 1)` is the
final state of token `t`. `turbo_analysis_embed::Encoder::run_ranges`
wraps that, and the head runs on the host over those states. Nothing in
the core's encoder changes for an annotator built this way, and every backend
that supports range writes can run it; the backends that do not refuse
the range write with an error that says so.

## Sharing the embedding pass

A head reads the final states of the encoder it was trained with. SaT
trains all of XLM-RoBERTa's layers it keeps (1, 3, 6, 9 or 12) together
with its classifier, and GLiNER trains its DeBERTa together with its
span head, so neither head means anything on the states of a separate
embedding model such as BGE-M3, even where the architecture and the
vocabulary are the same (BGE-M3 is XLM-RoBERTa large with XLM-R's
vocabulary). Sharing would need a head trained on the embedder's own
states, which no published checkpoint is.

What can be shared is less: the tokenization, when the two models have
the same tokenizer file (SaT and BGE-M3 do), and the document's views
and offset tables. The rows cannot be shared, as SaT reads windows of
up to 510 tokens every 64 tokens and the embedder reads its own rows.

## SaT

### Checkpoints the runtime can run

SaT's checkpoints come in two kinds (`segment-any-text` on the Hugging
Face hub):

- `sat-1l` to `sat-12l`: trained with limited lookahead, each layer's
  attention masked to a band of the tokens ahead (`lookahead` 48 in
  `sat-3l`'s config). The core has no banded attention mask, so these
  do not run yet; adding one is a change to every backend's attention.
- `sat-1l-sm` to `sat-12l-sm`: the supervised mixture models, with no
  lookahead and one label. These need nothing the core lacks; none has
  been run here yet (Not done, below).

### The step

`SatAnnotator` reads a view and writes `opennlp:sentences`:

1. The view is tokenized whole with the bundle's tokenizer
   (`encode_spans`, no special tokens), keeping each token's span.
2. The tokens are read in windows of `block_size` tokens (all a row has
   room for, 510 for a model of 512) every `stride` tokens (64, or the
   block size when that is smaller), the last window moved back to end
   at the last token, as wtpsplit's `extract` lays them out. Each
   window is one row with the template's `<s>` and `</s>` around it.
   The windows of all documents of a batch fill the session's rows
   together.
3. Each token's final state goes through the classifier
   (`classifier.weight` row 0 and `classifier.bias`, read from the
   bundle's raw weights after its hash is checked). A token in several
   windows gets the mean of its scores.
4. A token whose probability is over `threshold` ends a sentence at
   the end of its span; the cut moves past the whitespace after it.
   With `split_on_newlines` (wtpsplit's default), every line feed also
   ends a sentence. These are wtpsplit's `split`.
5. Unlike wtpsplit, each cut then moves forward to the end of the
   grapheme cluster it falls in (UAX #29), so no sentence ends between
   a letter and its combining mark or inside an emoji sequence, and
   each sentence is trimmed of whitespace clusters at both ends; empty
   ones are left out. Every offset is a UTF-8 byte offset into the view.

The threshold defaults to 0.25, wtpsplit's for the `-sm` models.

### Against wtpsplit

The tests compare the step with wtpsplit 2.2.2 on a SaT model of two
layers of width 32 with random weights (`testdata/tiny-sat-bundle`),
made and run in the container of `analysis/sat/reference`, whose
packages are each named by digest (`analysis/sat/README.md`). On 90
texts (scripts, whitespace, line and paragraph separators, combining
marks, emoji, special-token strings) and three settings of threshold,
stride and block size (0.5, 16, 62; 0.25, 62, 62; 0.5, 7, 30):

- The ids and offsets are wtpsplit tokenizer's on every text.
- The largest difference of a character's probability, on every text
  but the one below, is 3.2e-4, 1.2e-4 and 2.5e-4 for the three
  settings. wtpsplit sums and averages the windows' scores in float16,
  which moves a probability by less than about 6.6e-4 at these settings
  (`analysis/sat/tests/parity.rs`); the test allows 1e-3.
- The same characters are over the threshold on every text but the one
  below, except one character at threshold 0.25 whose probability is
  3.8e-6 from it; where they are the same, the pieces are `split`'s.
- The layer is `split`'s pieces with each cut moved to the end of its
  cluster and trimmed, on 89 texts at each threshold of 0.5 and 88 at
  0.25; the one left out at 0.5 is below, and at 0.25 also the text
  with the character at the threshold.
- `TURBO_TEST_BUNDLE=testdata/tiny-sat-bundle cargo test -p turbo
  --test conformance` checks the encoder against the mean of wtpsplit's
  encoder states for the recipe's cases.

On a device, `analysis/sat/tests/parity.rs` runs the step where
`TURBO_TEST_DEVICE` names (a backend name or a device index), at the
precision `TURBO_TEST_PRECISION` names (`model` by default), and holds
it to the same fixtures and tolerance:

```
TURBO_TEST_DEVICE=cuda cargo test --release -p turbo-sat --features turbo/cuda --test parity -- --nocapture
```

On CUDA (RTX 4080 SUPER, nvcc 12.4, driver 615.71.09), all four tests
pass at `model` and at `exact`. The largest probability differences are
3.2e-4, 1.2e-4 and 2.5e-4, as on the CPU. At `fastest`, which computes
in F16, the first character it fails on is 1.0e-3 from wtpsplit's
probability. That is over what the test allows, so the parity holds at
`model` and `exact` only.

What the step does differently, and why:

- **Tokenizer pipeline.** The reference runs wtpsplit with the model's
  `tokenizer.json` as the file says it (`PreTrainedTokenizerFast`),
  which is how transformers 4, which SaT was trained with, loads
  XLM-R's. transformers 5 builds `XLMRobertaTokenizer` from the
  vocabulary with a pipeline of its own: no collapse of space runs, a
  split at every Unicode whitespace character (so U+2028, U+2029 and
  U+0085 end words, which XLM-R's character map keeps), and no word
  start piece for whitespace at the end of a text. The core follows the
  file, so wtpsplit with transformers 5 gives other ids for such texts.
- **The prefix piece.** Before a text's first word, when no piece
  starts with that word's first characters, the tokenizer emits the
  bare word start `▁`. The core gives it an empty span, as it stands for
  no character; upstream gives it the text's first character, and
  wtpsplit puts its score there. The step gives it that character too
  (`cuts::place_prefix`).
- **Clusters the character map replaces.** Where the character map
  composes a cluster (`e` and U+0301 to `é`), upstream's offset can end
  inside it and the core's span covers it (core/tests/tokenizer.rs).
  The cut is the same once moved to the cluster's end.
- **A pad token in the text.** A text with `<pad>` written in it is
  tokenized to the pad id, as upstream does. HF's RoBERTa numbers
  positions only over tokens that are not the pad id, so the pad token
  gets the padding position and every token after it one position less
  than the core gives it. This changes the scores after it. It is a
  difference of the core's `FAMILY_ROBERTA` encoder, not of this step,
  and the embedder has it too; the test compares that text's tokens
  only.

### Speed on the CPU

`cargo test --release -p turbo-sat --test speed -- --ignored
--nocapture`, on a 4-core container, over 36 documents of 48828 bytes
(16464 tokens, about 460 a document), default options:

| Model | Shape | Time per pass | Tokens per second |
|---|---|---|---|
| `tiny-sat-bundle` | 2 layers, width 32 | 27.8 ms | 593,000 |
| `sat-1l-sm`'s shape, random weights | 1 layer, width 768, 12 heads, 3072 | 2,259 ms | 7,300 |

The second row's bundle was sealed the same way as the tiny one, over
the cut-down vocabulary, which changes the embedding table's size but
not the work per token. A document longer than one window is read
`block_size / stride` times over, about 8 times with the defaults, so
the stride is the main setting for throughput.

### Not done

- Real checkpoints. Huggingface.co refused the downloads from the
  machine this was built on, so no real SaT model was run, and there is
  no zoo entry (an entry needs each file's SHA-512). The checks against
  a real model are: seal a bundle from `sat-1l-sm` at revision
  `00a783439c201b20ee91ee1624dc817f1bf0e578` (`model.safetensors`,
  797951580 bytes, SHA-256
  `81eb7a4fcd4dab85c50866c7e2cf58f3d848e1c830c9207d132277110489ea0c`)
  with a recipe like `testdata/tiny-sat-recipe`'s, then run the
  `segment` command of the reference on it and compare as
  `analysis/sat/tests/parity.rs` does. CI would fetch it by that
  digest and cache it.
- Limited lookahead, for the models without `-sm`.
- A bundle task for token classification. The bundle carries a SaT
  model as `TASK_EMBED` with the classifier in its raw weights, read by
  name; the manifest has no place to say what the head is.
- Devices other than CUDA. The step runs where the range write runs;
  the wtpsplit parity is recorded on the CPU and on CUDA.
- The `-sm` models' class. They were trained on Universal Dependencies
  training sets among other data (the SaT paper, section 4.2), and UD
  holds treebanks under CC BY-NC-SA (English GUM, Czech PDT;
  docs/licences.md). Under the zoo's rule a model's class is the most
  restrictive of its own licence and its data's, so unless the mixture
  left those treebanks out, the `-sm` models are local-only
  (docs/licences.md, "Neural annotator checkpoints").

## GLiNER

### What a GLiNER model is

GLiNER (`urchade/GLiNER`) prepends the entity labels to the text as
prompt tokens (`<<ENT>> label <<ENT>> label <<SEP>> text`), runs the
whole row through a DeBERTa-v3 encoder, and then:

1. takes each word's first subword state, and each label's prompt
   state;
2. runs the word states through a one-layer bidirectional LSTM;
3. builds a state for every span of up to 12 words from its first and
   last word (the `markerV0` span representation: two feed-forward
   projections, concatenated, projected again);
4. projects the label states with a feed-forward layer;
5. scores every span against every label with a dot product and a
   logistic, keeps those over the threshold (0.5), and decodes them
   greedily, flat or nested.

### What the runtime lacks

- **DeBERTa-v3 attention.** DeBERTa has no absolute position
  embeddings in its input; each layer adds content-to-position and
  position-to-content terms from a shared table of relative position
  embeddings, with positions put in log-spaced buckets
  (`position_buckets` 256, `max_relative_positions` 512), keys and
  queries of the positions sharing the content projections
  (`share_att_key`), and the scores scaled by the square root of three
  times the head width. The core's encoder has absolute positions only.
  This is a new `FAMILY_DEBERTA_V2` in the core's CPU encoder, and in
  each backend after it.
- **DeBERTa-v3's tokenizer.** A SentencePiece Unigram model with a
  128,000-piece vocabulary (`spm.model`); the core reads SentencePiece
  model files, so this is a check, not new code.
- **The head.** The BiLSTM, the span representation and the label
  projection, on the host, from the checkpoint's `pytorch_model.bin`.
  The core reads safetensors only, so the head would be converted once
  when the bundle is sealed.
- **Words.** GLiNER splits the text into words with its own regular
  expression and scores word spans; the step would map them to
  `opennlp:tokens` or keep its own word layer, with byte offsets.

No permissive GLiNER checkpoint avoids DeBERTa: `gliner_small-v2.1`
uses `deberta-v3-small`, `gliner_multi-v2.1` `mdeberta-v3-base`,
`gliner_small-v2.5` `deberta-v3-small`, and NuNER Zero
`deberta-v3-large`. The bi-encoder `gliner-bi-small-v1.0` encodes the
labels with a BERT model, which the core runs, but the text still with
DeBERTa.

### The order of the work

1. DeBERTa-v3 on the CPU encoder, with a bundle from
   `microsoft/deberta-v3-small` (MIT) and conformance against
   transformers.
2. The GLiNER head and step on `gliner_small-v2.1` (Apache-2.0),
   compared with the `gliner` package in a container like SaT's, flat
   and nested (#382's spans).
3. `gliner_multi-v2.1` for languages other than English.

## BiLSTM-CRF part-of-speech tagger

`turbo-postag`'s BiLSTM-CRF tagger (`neural::BilstmPosTagger`, from
`opennlp:tokens` to `opennlp:pos`) runs its LSTMs on the CPU backend's
packed f32 GEMM (`turbo::cpu::Matrix`) by default, with the JVM's f64
numerics kept for the golden tests. Its goldens, accuracy and throughput
are in `analysis/postag/README.md`, "BiLSTM-CRF on the CPU backend". It
has only run on the CPU.

## Licences

Each checkpoint's licence, read from its model card's metadata at the
revision named on 2026-10-15, is in docs/licences.md, "Neural annotator
checkpoints". In short: the SaT checkpoints are MIT, the `-sm` ones
subject to their training data (above); GLiNER v2, v2.1 and v2.5 and
NuNER Zero are Apache-2.0 or MIT; `urchade/gliner_small` (v1) and the
other v0 and v1 GLiNER models are CC BY-NC 4.0 and are not used.
