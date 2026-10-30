# Language and script evidence (`turbo-langspan` and `turbo-langdetect`)

Classical language-model loading (`LanguageDetector`, `LanguageModel`,
`ProbingContext`, `ProbingConfig` in `analysis/langdetect/`) and the n-gram
language decoder are disabled under #1511. The replacement trained language head
is tracked in #1525. There is currently no statistical language-confidence claim
for the fallback.

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

The former language-classification accuracy and calibration tables apply only
to the removed model implementation. They are preserved in
[the historical language report](https://github.com/ai-pipestream/turboembed/blob/c71fba1a642e7221d0054528b5ee4fb183b4fd43/docs/langdetect.md),
not acceptance evidence for this script-only fallback or a future head.
