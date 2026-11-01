//! Name finders that need no model archive: `DictionaryNameFinder`,
//! `RegexNameFinder` and the patterns of `RegexNameFinderFactory`, the
//! merge of several finders' names, and CRF annotators decoding
//! caller-provided numerical scores with explicit provenance.
//!
//! Each finder reads a text view and its `turbo:tokens` and writes
//! `turbo:entities`: token ranges, UTF-8 byte spans of the view, entity
//! types and scores. [`MergeEntities`] reads several named instances of
//! `turbo:entities` and writes one without overlaps. On the token level
//! the finders are [`TokenNameFinder`]s, which take a sentence's tokens as
//! strings.
//!
//! # Modules
//!
//! - [`dictionary`]: [`DictionaryNameFinder`] over a [`Dictionary`], our
//!   line format with one typed entry per line.
//! - [`regex_finder`]: [`RegexNameFinder`], its factory functions and
//!   [`Pattern`].
//! - [`builtin`]: [`BuiltIn`], the factory's five patterns, matched by
//!   hand-written scanners over the joined UTF-8 bytes.
//! - [`merge`](mod@merge): `dropOverlappingSpans` and [`MergeEntities`].
//! - [`consensus`](mod@consensus): [`ConsensusEntityMerger`], the votes of
//!   several recognizers merged under an [`EntityMergePolicy`], with their
//!   evidence.
//! - [`annotate`]: the two finders as analysis steps.
//! - [`crf`]: [`CrfNameAnnotator`], decoding caller-provided numerical CRF
//!   scores through the typed entity schema of `turbo-sequence`.
//! - [`calibrated`][]: [`CalibratedCrfNameAnnotator`](calibrated::CalibratedCrfNameAnnotator)
//!   and its calibration identity, span marginals with an abstaining policy.
//! - [`rules`]: deterministic projection of [`Temporals`](turbo_analysis_layers::Temporals),
//!   [`Times`](turbo_analysis_layers::Times), [`Money`](turbo_analysis_layers::Money),
//!   and percentage [`Quantities`](turbo_analysis_layers::Quantities) into
//!   [`Entities`](turbo_analysis_layers::Entities) (`"date"`, `"time"`,
//!   `"money"`, `"percent"`), preserving source provenance for
//!   [`ConsensusEntityMerger`].
//!
//! # Regular expressions
//!
//! This crate uses no external `regex` engine. The five built-in patterns
//! are matched directly over the joined UTF-8 bytes by hand-written
//! scanners with no regex compilation or stand-in buffer ([`builtin`]
//! explains how, including `EMAIL`'s lookbehind and `URL`'s word
//! boundaries). A caller's own pattern is a `turbo_matcher` token
//! pattern: each `Tok` is the tests one token must pass, so a match is a
//! run of whole tokens, found by the matcher's automaton in time linear
//! in the text.
//!
//! # Behaviour worth knowing
//!
//! - `RegexNameFinder` returns names type by type, in the order the
//!   finder's map iterates, not in text order. The factory keeps the
//!   types in the order they were first given.
//! - A built-in's match counts only when it starts and ends at a token
//!   boundary of the space-joined tokens. A match that starts with the
//!   separating space starts at the next token; one that ends with it
//!   ends at the token before. A match dropped because it ends inside a
//!   token still consumes its text, so a shorter match inside it is not
//!   found.
//! - `URL` starts with `\b`, so its `/` and `~/` forms need a word
//!   character just before the slash or tilde: `~/a.com` at the start of
//!   a sentence or after a space is never found, and in `x/a.com` the
//!   match starts inside the token and is dropped.
//! - The classes of `DEGREES_MIN_SEC_LAT_LON` are written like
//!   `[N|n|S|s]`, so they also match `|`.
//! - The built-in scanners read the joined text as bytes with the ASCII
//!   classes `\d`, `\w`, `\s` and `\b`; the text between the tokens is
//!   ASCII spaces, so a match that is kept lies within the tokens.
//! - `DictionaryNameFinder` keeps the longest entry from each token and
//!   resumes after it, so of two overlapping entries only the one that
//!   starts first is found, even when the other is longer. Of entries
//!   that fold to the same tokens the first is kept.
//! - `dropOverlappingSpans` counts an empty span at the end of a kept
//!   span as inside it, and drops it.
//! - Names from these finders carry probability 0, which becomes the
//!   entity's score.
//!
//! # Not reproduced
//!
//! - `RegexNameFinder.find(String)`, over the characters of a text
//!   rather than tokens.
//! - Patterns over characters inside a token: a `Tok` pattern matches
//!   whole tokens.

pub mod annotate;
pub mod calibrated;
pub mod crf;
pub use crf::{CrfNameAnnotator, CrfSentenceScores};
pub mod builtin;
pub mod consensus;
pub mod dictionary;
pub mod error;
pub mod merge;
pub mod name;
pub mod regex_finder;
pub mod rules;

pub use builtin::BuiltIn;
pub use consensus::{Consensus, ConsensusEntityMerger, EntityMergePolicy, Vote, consensus};
pub use dictionary::{Dictionary, DictionaryNameFinder};
pub use error::Error;
pub use merge::{FilterEntities, MergeEntities, drop_overlapping, merge};
pub use name::{Name, TokenNameFinder};
pub use regex_finder::{Pattern, RegexNameFinder, Test, Tok, TokensPattern};
pub use rules::{
    DATE_ENTITY, DateTimeEntityKind, MONEY_ENTITY, PERCENT_ENTITY, RULE_ENTITY_KINDS, RuleEntitiesAnnotator,
    RuleEntityProjector, RuleEntitySource, TIME_ENTITY, TimeEntityFilter, money_to_entities, quantities_to_entities,
    rule_entities, span_token_range, temporals_to_entities, times_to_entities, times_to_entities_with,
};
pub use turbo_analysis::FilterRule;
