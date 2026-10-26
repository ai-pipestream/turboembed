//! Rule-based matching over a document's token layers, after spaCy's
//! `Matcher`, `PhraseMatcher` and `SpanRuler` (MIT).
//!
//! - [`Matcher`]: token patterns built in Rust ([`Tok`]), each token a
//!   set of tests on its text, lowercase, shape, affixes, a normalized
//!   form, lemma, stem, part of speech, entity, flags (punctuation, case,
//!   stopword, number...) and length, with the operators `!`, `?`, `*`,
//!   `+` and repeats `{n,m}`. Patterns compile to one automaton, simulated
//!   state set by state set, so no pattern can make matching exponential.
//! - [`PhraseMatcher`]: a large list of phrases over one token attribute,
//!   as a trie over token ids.
//! - [`SpanRuler`]: both, under labels, with phrases over as many fields
//!   as wanted, as an analysis step that writes `turbo:matches`
//!   ([`Matches`]) under a named instance, keeping every match or
//!   resolving overlaps with `filter_spans` or spaCy's own token-length
//!   filter.
//! - [`Matcher::align`]: which pattern token took each token of a match,
//!   spaCy's `with_alignments`, as a pass over a match already found.
//!
//! A match is a range of tokens ([`IxRange<Tokens>`]); its bytes are the
//! tokens' ([`Input::span`]), in the view the tokens were made in.
//! Patterns read the layers the [`Input`] is given; a pattern reading one
//! it lacks fails the call with [`MatchError::Missing`].
//!
//! [`IxRange<Tokens>`]: turbo_analysis::IxRange
//! [`Tokens`]: turbo_analysis_layers::Tokens
//!
//! # What may surprise
//!
//! - **Every start and every end.** A pattern `a b*` over "a b b" matches
//!   "a", "a b" and "a b b", as spaCy does; [`Greedy`] keeps the first or
//!   longest per label, and the ruler's [`Policy`] resolves overlaps
//!   across labels.
//! - **Values are compared as written.** `Tok::attr(Attr::Lower,
//!   "Boston")` matches nothing. Phrases are different: a phrase matcher
//!   over [`Attr::Lower`] lowercases the phrase's tokens, as spaCy reads
//!   a pattern document's attributes.
//! - **Phrases are pre-tokenized.** A phrase is a slice of tokens; it
//!   must be split as the document's tokenizer splits text, or it will
//!   not line up with the tokens.

mod align;
mod attr;
mod entries;
mod error;
mod hash;
mod input;
mod matcher;
mod nfa;
mod pattern;
mod phrase;
mod predicates;
mod ruler;
pub mod spacy;
mod vocab;

pub use attr::{Attr, Cmp, Field, Flag, Iob};
pub use error::{MatchError, PatternError};
pub use input::Input;
pub use matcher::{Greedy, Match, Matcher, PatternId};
pub use pattern::{Test, TextTest, Tok};
pub use phrase::{PhraseId, PhraseMatch, PhraseMatcher};
pub use predicates::{SetRelation, ValueTest};
pub use ruler::{Policy, SpanRuler};
pub use turbo_analysis_layers::Matches;
pub use vocab::Label;

mod dependency;
pub use dependency::{DependencyError, DependencyMatch, DependencyMatcher, DependencyNode, DependencyOp, MatchLimits};
