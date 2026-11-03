# Language and script evidence (`turbo-langspan` and `turbo-langdetect`)

Language identification is a hashed character n-gram linear classifier with
script priors, trained by our own Rust trainer. Without a trained model,
`turbo-langspan` falls back to model-free script evidence.

## Native n-gram model

`turbo_langdetect::NgramModel` hashes every Unicode scalar n-gram in its
configured range (1 to 3 by default) with FNV-1a into a fixed number of
buckets (65,536 by default), sums one linear weight row per language, adds a
small prior for languages written in the text's dominant script, and
normalizes with a softmax. Feature extraction iterates buckets in sorted
order, so scores are deterministic. `NgramModel::to_bytes` and
`NgramModel::from_bytes` read and write the model file (entry name
`langid.ngram`, magic `TNL1`, little-endian); a truncated, oversized or
inconsistent file is a `ModelError`, never a panic. Blank text has no
prediction.

`turbo_langspan::NgramDetector` wraps a model as an annotator. It writes
`Language` with the best code and its confidence, and `LanguageSpans` over
the letter runs of the text, each run carrying its own script and the
whole-text prediction; runs are not classified independently. Text with no
letters produces empty layers. In `turbo-query`,
`QueryProfileBuilder::with_ngram_detector` installs the detector in place
of the script fallback.

## Training and evaluation

`turbo_trainer::langid::train` is a deterministic softmax trainer over the
same hashed features: the same examples and configuration give the same
weights. `turbo-langid-train` (in `turbo-train-recipes`) trains on the
checked Tatoeba export plan and the pinned FineWeb-2 ranges, keeps the
held-out split apart from training text, writes `langid.ngram`, and prints
a receipt with input digests and held-out accuracy. `turbo-langid-flores`
scores a model file on converted FLORES+ devtest gold and prints accuracy
and throughput with the model and gold digests. FLORES+ is evaluation only.
No accuracy is published here until those receipts are recorded.

## Model-free fallback

`turbo_langspan::ScriptDetector` consumes an existing `View` and optional
`Sentences` and `Tokens` layers. It writes `LanguageSpans` and `Language`
without loading a model. All emitted language codes are `und`, with confidence
`0.0`: a script cannot distinguish languages that share it.

With input tokens, it preserves each token's boundaries and selects its most
frequent letter script, choosing the first encountered script on ties.
Adjacent evidence of the same script is combined into one source span, which
may include intervening punctuation or whitespace. Without tokens, it scans
Unicode letters directly and retains adjoining combining marks. Common,
Inherited and Unknown script values do not create letter evidence. Empty
or non-letter input produces empty language layers.

Optional sentence and token spans are checked against the source view before
use. Invalid UTF-8 boundaries, reversed, overlapping or out-of-range spans
return an error. No joined text or secondary tokenizer is used.

## Validation

`analysis/langspan/tests/scripts.rs` checks mixed scripts, supplementary
letters and combining marks, exact source spans, existing token boundaries,
empty input and malformed layers. Run:

```sh
cargo nextest run -p turbo-langspan
```

```sh
cargo nextest run -p turbo-langdetect -p turbo-langspan -p turbo-train-recipes
```

The former language-classification accuracy and calibration tables apply only
to the removed model implementation. They are preserved in
[the historical language report](https://github.com/ai-pipestream/turboembed/blob/c71fba1a642e7221d0054528b5ee4fb183b4fd43/docs/langdetect.md),
not acceptance evidence for the native model or the script-only fallback.
