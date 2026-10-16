//! Glossary term matching and the glossary readers: OpenNLP's
//! `opennlp.tools.glossary` (`AhoCorasickGlossaryMatcher`,
//! `TermAnalyzingGlossaryMatcher`, `CompositeGlossaryMatcher`,
//! `GlossaryAnnotator`) and `opennlp.tools.formats.glossary`
//! (`CsvGlossaryReader`, `TbxGlossaryReader`).
//!
//! - [`TextMatcher`]: terms found by their characters with an
//!   Aho-Corasick automaton over code points, kept when both ends are on
//!   given [`Boundaries`] (UAX #29 word boundaries for OpenNLP's matcher,
//!   token edges in the step), overlaps resolved leftmost, then longest,
//!   then first registered.
//! - [`TokenMatcher`]: terms found as runs of tokens whose values (the
//!   token text of a normalized view, or stems) equal a term's analyzed
//!   tokens, whatever separates them; built on turbo-matcher's
//!   [`PhraseMatcher`](turbo_matcher::PhraseMatcher).
//! - [`GlossaryMatcher`]: either of them, or a composite of several by
//!   priority.
//! - [`GlossaryAnnotator`]: the step, which reads a text view and its
//!   tokens (and `turbo:stems` when a token matcher matches by them) and
//!   writes [`GlossaryTerms`] (`turbo:glossary-terms`).
//! - [`DelimitedReader`] and [`TbxReader`]: comma or tab separated lists
//!   and TBX 2 or 3 termbases, read into a [`Glossary`] with each term's
//!   byte span in the file; [`write_tbx`] writes a glossary back as TBX.
//!
//! # Why two matchers
//!
//! OpenNLP's character matcher compares the separators inside a term, so
//! "hot dog" does not match "hot  dog" or "hot-dog"; that is a property
//! of characters, not tokens, and is what the JVM goldens check, so it is
//! an automaton over code points. Its failure links let one pass over
//! the text find every occurrence. The token matcher only compares token
//! values, which is what turbo-matcher's phrase trie does, so it uses it
//! rather than a second automaton. `examples/glossary_bench.rs` measures
//! both on the same glossary and text; the numbers are in
//! `analysis/glossary/README.md`.
//!
//! # Where this differs from OpenNLP
//!
//! - Offsets are UTF-8 bytes; OpenNLP's are UTF-16 units. A match never
//!   cuts a character in either, so the spans are the same characters.
//! - OpenNLP folds text through an `OffsetAwareNormalizer` inside the
//!   matcher and checks word boundaries in the original text. Here the
//!   normalized text is a view, the step matches over the view and its
//!   tokens, and `Document::to_source` maps the spans back; a caller that
//!   wants OpenNLP's exact behaviour scans the view
//!   ([`TextMatcher::scan`]), maps each hit through the view's alignment,
//!   checks the source boundaries and calls [`select`], as the goldens
//!   test does.
//! - The step's character matches must start and end at token edges or
//!   at the ends of the text, where OpenNLP uses UAX #29 word boundaries.
//!   With the UAX #29 word tokenizer the two differ only inside runs of
//!   punctuation and symbols, which are not tokens: OpenNLP finds "C+"
//!   in "C++", the step does not. [`TextMatcher::find`] takes any
//!   [`Boundaries`], so a caller can match at UAX #29 boundaries, as the
//!   goldens test does.
//! - OpenNLP's `CsvGlossaryReader` takes a UTF-16 `char` as its
//!   delimiter; [`Delimiter`] takes any character but the quote and the
//!   line breaks.
//! - The TBX reader reads UTF-8 only, and refuses an XML declaration
//!   naming another encoding, which the JDK's parser would decode. It
//!   refuses an internal entity whose replacement text holds markup,
//!   which the JDK parses as elements; it does not expand parameter
//!   entity references in the internal subset nor apply attribute
//!   defaults declared there; and it refuses an external parameter
//!   entity declaration, where OpenNLP checks only general entities.
//!
//! # What looks wrong in OpenNLP
//!
//! - `Span.intersects`, which the selection and the composite use, counts
//!   an empty span at either end of another span as intersecting it,
//!   where two spans that only touch do not intersect. An empty match can
//!   come only from a normalizer that inserts text; it is then dropped
//!   when a match ends or starts at its offset. [`intersects`] reproduces
//!   this, and differs there from [`Span::overlaps`](turbo_analysis::Span::overlaps).
//! - `TbxGlossaryReader` at the goldens' OpenNLP commit trims terms with
//!   `String.trim()`, which removes control characters below U+0021 but
//!   no other whitespace (a no-break space stays); the addons branch it
//!   was ported from trims Unicode whitespace, as this crate does.
//! - `CsvGlossaryReader` calls a line of spaces a record of one field and
//!   refuses the file, though it skips empty lines; it is reproduced.

mod annotate;
mod boundaries;
mod error;
mod matcher;
mod read;
mod select;
mod text;
mod token;

pub use annotate::GlossaryAnnotator;
pub use boundaries::Boundaries;
pub use error::{BuildError, MatchError, ReadError};
pub use matcher::{GlossaryMatcher, MatchInput};
pub use read::{DelimitedReader, Delimiter, FirstRecord, GlossaryFile, TbxDialect, TbxReader, WriteError, write_tbx};
pub use select::{Candidate, GlossaryMatch, intersects, merge, select};
pub use text::{Case, TextMatcher};
pub use token::{TokenMatcher, TokenValue};
pub use turbo_glossary_layers::{EntryError, EntryIx, Glossary, GlossaryEntry, GlossaryTerms};
