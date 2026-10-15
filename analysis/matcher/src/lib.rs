//! One rule engine over a document's token layers: spaCy's `Matcher`,
//! `PhraseMatcher` and `SpanRuler`, as typed Rust.
//!
//! - [`Matcher`] runs token patterns. A pattern is a list of [`Tok`], each
//!   a set of tests one token must pass (its text, lowercase form, lemma,
//!   tag or entity kind against a literal or a set, a regular expression,
//!   a fuzzy match, its length, its [`Flag`]s) and an [`Op`]: one token,
//!   `?`, `*`, `+`, `!` or a counted repeat.
//! - [`PhraseMatcher`] finds a large list of multi-token phrases by one
//!   attribute, compiled into an Aho-Corasick automaton over token
//!   symbols: one pass over the tokens whatever the number of phrases.
//! - [`Ruler`] runs both, merges their labels and settles overlaps by an
//!   [`Overlap`] policy; [`RulerAnnotator`] is the ruler as an analysis
//!   step, writing `turbo:matches` (turbo-matcher-layers) under the
//!   instance name the pipeline gives it.
//!
//! Patterns are Rust values built in code; there is no pattern language
//! read at run time. Matches are runs of tokens, so their text spans come
//! from the tokens' spans and map back to the source like any other span.
//!
//! # What a match is
//!
//! A pattern matches every run of tokens `[start, end)` whose tokens it
//! accepts, as spaCy's matcher does: `a b*` over `a b b` matches `a`,
//! `a b` and `a b b`. A rule's match over one run is reported once,
//! whichever of its patterns or paths reached it, and an empty run is
//! never a match. [`Greedy`] keeps, per rule, the leftmost or the longest
//! of overlapping matches, by spaCy's own sort and sweep.
//!
//! # Where it differs from spaCy
//!
//! - **Order.** Matches come in order of start, end and label. spaCy
//!   reports them in the order its state machine finds them, with
//!   filtered rules after the others.
//! - **Alignments.** [`Matcher::align`] chooses, where a pattern can take
//!   a run of tokens more than one way, the way a backtracking regular
//!   expression with greedy operators would. spaCy's choice follows from
//!   its state machine's history and is not one rule; the two agree on
//!   every alignment its tests fix except where the tests say otherwise
//!   (`tests/spacy_matcher.rs` lists them).
//! - **Attributes.** [`Attr::Norm`] is a case and compatibility fold, not
//!   spaCy's per-language norm table; [`Flag::LikeNum`] reads digits, and
//!   number words only through `turbo:numbers`. There are no `LIKE_URL`,
//!   `LIKE_EMAIL`, `MORPH` or dependency tests; extension attributes are
//!   [`Attr::Column`]s.
//! - **Callbacks.** There are none: a caller reads the matches.

mod annotate;
mod attr;
mod fuzzy;
mod matcher;
mod pattern;
mod phrase;
mod ruler;
mod seq;
mod vocab;

pub use annotate::RulerAnnotator;
pub use attr::{Attr, Flag};
pub use matcher::{Greedy, Match, MatchError, Matcher, MatcherBuilder};
pub use pattern::{Cmp, Op, Pattern, PatternError, Tok};
pub use phrase::{PhraseError, PhraseMatcher, PhraseMatcherBuilder, PhraseOn};
pub use ruler::{Overlap, Ruler};
pub use seq::{Sequence, SequenceError};
