# turbo-phrase-chunker

Zero-training rule-driven POS-tag pattern chunker (`TagPatternRule`,
`PatternChunker`, and `PatternChunkComposer`) on the analysis API:

- `TagPatternRule` and `PatternChunker` (`src/pattern/mod.rs`,
  `src/pattern_chunker.rs`) compile declarative POS-tag and token-literal
  patterns (such as `<DT>?<JJ.*>*<NN.*>+` or `<VB.*><RP|IN>`) directly into a
  token-level automaton over `(tag, token_surface)` without concatenating POS
  tags into synthetic strings or running regexes, writing
  `turbo_analysis_layers::PhraseChunks`.
- `PatternChunkComposer` merges rule chunks with a base `PhraseChunks` layer in
  `ComposeMode::Fallback` or `ComposeMode::Override`.

The crate documentation (`src/lib.rs`) describes the modules. The name keeps it
apart from `turbo-chunker`, which cuts retrieval units.
