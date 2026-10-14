//! Constituency parse trees, Penn Treebank bracket I/O, and language head rules (`opennlp.tools.parser`).
//!
//! This crate ports OpenNLP 3.x's constituency parse representation, zero-regex Penn Treebank
//! scanner, and English and AnCora Spanish head rules:
//!
//! - [`Parse`] (`opennlp.tools.parser.Parse`): constituency tree node with shared sentence text,
//!   UTF-8 byte [`Span`](turbo_analysis::Span) offsets, head constituent tracking, derivation
//!   tracking, single-pass cursor bracket parsing (`Parse::parse`, `Parse::parse_with_options`),
//!   and bracket rendering (`Parse::show`, `Parse::show_code_bank`).
//! - [`ParseSampleStream`] (`opennlp.tools.parser.ParseSampleStream`): line-by-line Penn Treebank
//!   parse stream, plus [`ParserPosSample`] (`PosSampleStream`) and [`ParserChunkSample`]
//!   (`ChunkSampleStream`) conversions.
//! - [`HeadRules`] and [`GapLabeler`] (`opennlp.tools.parser.HeadRules`,
//!   `opennlp.tools.parser.GapLabeler`):
//!   - [`EnglishHeadRules`] (`opennlp.tools.parser.lang.en.HeadRules`): Collins/Magerman English
//!     head rules with `NP`/`NX` priority passes and subject/object trace extraction gap labeling.
//!   - [`AncoraSpanishHeadRules`] (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`): AnCora
//!     Spanish head rules using a 62-state `u64` bitmask automaton ([`TagPattern`]) with both
//!     `# opennlp-ancora-head-rules 2` and legacy format import.

pub mod head_rules;
pub mod parse;
pub mod stream;

pub use head_rules::{
    ANCORA_FORMAT_HEADER, AncoraSpanishHeadRules, EnglishHeadRules, GapLabeler, HeadRule, HeadRules, HeadRulesError,
    MAX_PATTERN_STATES, MAX_TAGS_PER_RULE, TagPattern,
};
pub use parse::{
    BRACKET_LCB, BRACKET_LRB, BRACKET_LSB, BRACKET_RCB, BRACKET_RRB, BRACKET_RSB, COMPLETE, CONT, Constituent,
    INC_NODE, INCOMPLETE, OTHER, Parse, ParseError, START, TOK_NODE, TOP_NODE, decode_token, encode_token,
    space_untokenized_parens,
};
pub use stream::{ParseSampleStream, ParserChunkSample, ParserPosSample};
