# Decisions

Decisions already made for TurboEmbed. A pull request does not reopen
one. To question one, comment on its issue (or open one) first; if it
changes, the pull request that acts on the change edits its entry here.

Each entry says what was decided, and where it is still open, what is
open and what the code does until it is settled.

## Document analysis and the OpenNLP port

Settled in the analysis design (October 2026). The design is described
in [`docs/analysis.md`](analysis.md).

1. **A type per layer.** Each kind of analysis result is its own Rust
   type, and annotators declare the layer types they read and write.
2. **Crates in this workspace.** The analysis core, one crate for shared
   layers, and a crate per add-on with its own small schema crate.
3. **Standard ids.** Shared layers use `opennlp:` ids; TurboEmbed's own
   layers use `turbo:` ids.
4. **OpenNLP parity by golden files.** The OpenNLP 3.x JVM produces
   golden files that the Rust annotators are tested against.
5. **No run-time bridge to OpenNLP.** No JVM and no native OpenNLP
   library at run time.
6. **OpenNLP 3.x models are read directly.** The `.bin` model files load
   without conversion.
7. **Normalizers first.** Text normalization, which writes new views,
   lands before the annotators that read normalized text.
8. **Evaluation sets.** Ported components report numbers on the agreed
   evaluation sets next to OpenNLP's.
9. **Training in pinned containers.** Model training and golden-file
   generation run in pinned containers driven from Rust.
10. **OpenNLP 2.x is out of scope.**
11. **No greedy chunking.** Chunking reads the analysis layers and keeps
    exact offsets to the source; token-budget packing is not a chunker.

Open:

- **Offsets at the edge.** Inside, offsets are UTF-8 byte offsets into
  one view. How UTF-16 offsets are produced for Java and the wire is not
  decided; until it is, the code is UTF-8 only.
- **Additions to the C header for analysis.** Which writes go into
  `include/turbo/` at stage 2 is still being decided. Do not change the
  header for analysis until it is.
