//! Constituency parse trees, Penn Treebank bracket I/O, language head rules, and bottom-up
//! constituency parsers (`opennlp.tools.parser`).
//!
//! This crate ports OpenNLP 3.x's constituency parse representation, Penn Treebank reader and
//! writer, English and AnCora Spanish head rules, and the `CHUNKING` and `TREEINSERT` bottom-up
//! constituency parsers, trainers, model I/O, evaluator, cross-validator, and pipeline annotator:
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
//! - [`ParserModel`], [`ParserType`], [`ParserEventType`], [`HeadRulesArtifact`], and
//!   [`ParserChunkerModel`] (`opennlp.tools.parser.ParserModel`, `ParserType`, `ParserEventTypeEnum`):
//!   ZIP `.bin` model packaging and loading for both `CHUNKING` and `TREEINSERT` parsers.
//! - [`Parser`], [`ParserEngine`], and [`ParserFactory`] (`opennlp.tools.parser.Parser`,
//!   `AbstractBottomUpParser`, `ParserFactory`): shared K-best POS and chunker beam search and
//!   bottom-up derivation search.
//! - [`chunking`] (`opennlp.tools.parser.chunking`): Ratnaparkhi (1999) two-stage (`build` + `check`)
//!   constituency parser, context generators, event stream, and trainer.
//! - [`treeinsert`] (`opennlp.tools.parser.treeinsert`): Adwait Ratnaparkhi / OpenNLP three-stage
//!   (`build` + `attach` + `check`) right-frontier tree-insertion parser, context generators, event
//!   stream, and trainer.
//! - [`ParserEvaluator`] and [`ParserCrossValidator`] (`opennlp.tools.parser.ParserEvaluator`,
//!   `ParserCrossValidator`): Parseval constituent F-measure evaluation and K-fold cross-validation.
//! - [`ParserAnnotator`] and [`ParserTool`] (`opennlp.tools.parser.ParserAnnotator`,
//!   `opennlp.tools.cmdline.parser.ParserTool`): pipeline [`Annotator`](turbo_analysis::Annotator)
//!   producing [`Phrases`](turbo_analysis_layers::Phrases) (`opennlp:phrases`) and command-line
//!   line parser.

pub mod annotator;
pub mod bottom_up;
pub mod chunking;
pub mod context;
pub mod evaluator;
pub mod head_rules;
pub mod model;
pub mod parse;
pub mod stream;
pub mod treeinsert;

pub use annotator::{ParserAnnotator, ParserTool};
pub use bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser, ParserEngine, ParserFactory, build_dictionary,
    collapse_punctuation,
};
pub use chunking::ParserTrainingParameters;
pub use context::{ChunkContextGenerator, Cons, ParserChunkerSequenceValidator};
pub use evaluator::{ParserCrossValidator, ParserEvaluator};
pub use head_rules::{
    AncoraSpanishHeadRules, EnglishHeadRules, GapLabeler, HeadRule, HeadRules, HeadRulesError, MAX_PATTERN_ATOMS,
    MAX_TAGS_PER_RULE, TagPattern,
};
pub use model::{HeadRulesArtifact, ParserChunkerModel, ParserEventType, ParserModel, ParserType};
pub use parse::{
    BRACKET_LCB, BRACKET_LRB, BRACKET_LSB, BRACKET_RCB, BRACKET_RRB, BRACKET_RSB, COMPLETE, CONT, Constituent, Head,
    INC_NODE, INCOMPLETE, OTHER, Parse, ParseError, START, TOK_NODE, TOP_NODE, decode_token, encode_token,
    space_untokenized_parens,
};
pub use stream::{ParseSampleStream, ParserChunkSample, ParserPosSample};
