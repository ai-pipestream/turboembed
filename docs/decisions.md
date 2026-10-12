# Decisions

Decisions already made for TurboEmbed. A pull request does not reopen
one. To question one, comment on its issue (or open one) first; if it
changes, the pull request that acts on the change edits its entry here.

Each entry says what was decided, and where it is still open, what is
open and what the code does until it is settled.

## Document analysis and the OpenNLP port

Settled in the analysis design (October 2026). The analysis docs page,
added with the analysis crates in
[#142](https://github.com/ai-pipestream/turboembed/pull/142), describes it.

1. **A type per layer.** Each kind of analysis result is its own Rust
   type, stored by column, with tags, lemmas and parses as layers aligned
   to tokens. A step gets only the layers it declares.
2. **Crates in this workspace.** An analysis core, one schema crate for
   the shared layers, and a crate per add-on. A consumer of a layer never
   depends on the add-on that produces it.
3. **Standard ids.** Shared layers take the OpenNLP standard's
   `opennlp:` ids, each mapped to its `StandardLayer` value. Layers the
   standard lacks (structure, boundaries, confidence) are `turbo:` and
   are proposed to it. A shared-layer change bumps the schema version.
4. **Offsets.** UTF-8 byte offsets inside. Exact UTF-16 and code point
   offsets at the Java and wire edge, through a checkpoint table per
   view.
5. **Keeping up with OpenNLP.** Golden typed-layer files are made on the
   JVM from a pinned OpenNLP 3.x commit, and CI compares Rust against
   them, so CI needs no JVM. Regenerating them at a newer commit lists
   what Rust must catch up on.
6. **No native OpenNLP library in turboembed.** `libopennlp.so` stays a
   local tool for differential fuzzing only.
7. **Models.** OpenNLP 3.x `.bin` models are read as they are, by format
   id, with the manifest hash in provenance. New models ship as
   turboembed bundles.
8. **Normalizers first.** Offset-aware NFC, NFKC, case folding,
   whitespace and dehyphenation come first; confusables and the emoji and
   social-media folds come later.
9. **C header additions.** One write and one field, added when stage 2
   is built: late pooling (token ranges in, one vector per range out) and
   a thread count in `turbo_session_desc`. A device-side copy for a
   vectors layer waits for a benchmark record that shows it pays.
10. **Evaluation sets.** A BEIR subset, LegalBench-RAG mini, ConTEB,
    GutenQA, HiCBench and Qasper first; then an enterprise-document
    benchmark built and published by the project.
11. **Training.** Models are trained in a pinned container driven by
    `turbo-bundle`, and the result is sealed as a bundle.

Also settled:

- **OpenNLP 2.x is out of scope.**
- **No greedy chunking.** Chunking reads the analysis layers and keeps
  exact offsets to the source; token-budget packing is not a chunker.
- **When a stage lands.** Its ported annotators reproduce the JVM
  goldens' spans and labels exactly and their probabilities within a
  stated tolerance, with benchmark records on each backend they run on.
  Work with no OpenNLP counterpart (the chunker, boundary costs, late
  pooling, the normalization offset maps) lands through the chunker
  gates and its own tests instead.
