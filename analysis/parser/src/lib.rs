//! Constituency parse trees, Penn Treebank bracket I/O, and language head rules (`opennlp.tools.parser`).
//!
//! This crate ports OpenNLP 3.x's constituency parse representation, its Penn Treebank reader
//! and writer, and its English and AnCora Spanish head rules, matching OpenNLP at the commit the
//! goldens name (`analysis/goldens-tool/src/recipe.rs`):
//!
//! - [`Parse`] (`opennlp.tools.parser.Parse`): constituency tree node with shared sentence text,
//!   UTF-8 byte [`Span`](turbo_analysis::Span) offsets, its [`Head`], derivation tracking, the
//!   bracket reader of `Parse.parseParse` (`Parse::parse`, `Parse::parse_with_options`) and the
//!   bracket writer of `Parse.show` (`Parse::show`, `Parse::to_string_penn_treebank`).
//! - [`ParseSampleStream`] (`opennlp.tools.parser.ParseSampleStream`): line-by-line Penn Treebank
//!   parse stream, plus [`ParserPosSample`] (`PosSampleStream`) and [`ParserChunkSample`]
//!   (`ChunkSampleStream`) conversions.
//! - [`HeadRules`] and [`GapLabeler`] (`opennlp.tools.parser.HeadRules`,
//!   `opennlp.tools.parser.GapLabeler`):
//!   - [`EnglishHeadRules`] (`opennlp.tools.parser.lang.en.HeadRules`): Collins/Magerman English
//!     head rules with `NP`/`NX` priority passes and subject/object trace extraction gap labeling.
//!   - [`AncoraSpanishHeadRules`] (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`): AnCora
//!     Spanish head rules, whose tags are Java regular expressions, matched by [`TagPattern`]
//!     without backtracking.

pub mod head_rules;
pub mod parse;
pub mod stream;

pub use head_rules::{
    AncoraSpanishHeadRules, EnglishHeadRules, GapLabeler, HeadRule, HeadRules, HeadRulesError, MAX_PATTERN_ATOMS,
    MAX_TAGS_PER_RULE, TagPattern,
};
pub use parse::{
    BRACKET_LCB, BRACKET_LRB, BRACKET_LSB, BRACKET_RCB, BRACKET_RRB, BRACKET_RSB, COMPLETE, CONT, Constituent, Head,
    INC_NODE, INCOMPLETE, OTHER, Parse, ParseError, START, TOK_NODE, TOP_NODE, TokenSource, decode_token, encode_token,
    space_untokenized_parens,
};
pub use stream::{ParseSampleStream, ParserChunkSample, ParserPosSample};
