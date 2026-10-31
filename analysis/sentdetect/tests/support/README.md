# Native boundary targets

`native_targets.rs` contains project-authored text and explicit boundaries,
under the repository's Apache-2.0 licence. These targets were written as
expected annotations, not generated from a model or copied from a corpus.
No .bin reader, Apache model, JVM process or download is needed.

Each target lists the exact source substrings in order. The test derives
byte spans and per-character Continue/End supervision from these annotations,
checks that no non-whitespace source content is missing, then compares the
Rust rule detector and its actual analysis pipeline to those targets.
Repeated substrings are matched sequentially, so identical text at different
positions has distinct offsets. Padding checks preserve those offsets by
translation. UTF-16 and code-point conversions must round-trip to the original
byte ranges.

This small supervision contract covers abbreviations, decimals, repeated text,
quotes, soft lines, paragraphs, combining characters, astral characters and
several language terminators. It is not a measured trained-model accuracy gate.
Changing a target requires an explicit boundary-policy explanation in review;
there is no snapshot regeneration command.

`native_properties.rs` enumerates all 5,832 triples of 18 text fragments under
three configurations (17,496 cases). It checks content preservation, ordered
nonempty spans, Unicode trimming, typed offsets and fresh versus reused scratch
and fallible API agreement. This bounded exhaustive generator adds no dependency
and has no timing assertion or random seed.
