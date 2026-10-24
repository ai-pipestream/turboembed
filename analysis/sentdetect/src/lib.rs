//! Sentence and paragraph detection, ported from OpenNLP 3.x onto the
//! analysis API.
//!
//! - [`NewlineDetector`] writes one sentence per non-blank line, as
//!   OpenNLP's `NewlineSentenceDetector` does.
//! - [`ParagraphDetector`] writes the paragraphs of a view, runs of
//!   non-blank lines, as `turbo:structure` blocks.
//! - [`SentenceDetector`] splits a view into sentences with an OpenNLP
//!   maxent model ([`SentenceModel`]), as `SentenceDetectorME` does, and
//!   writes a probability per sentence.
//! - [`RuleSentenceDetector`] splits a view into sentences with the UAX #29
//!   sentence rules tailored per language ([`rules`]), for the languages
//!   that have no model.
//!
//! The maxent and rule detectors can consult a classified
//! [`AbbreviationInventory`] ([`inventory`]) on top of their own data.
//!
//! The newline, paragraph and rule detectors need no model. Each detector takes
//! OpenNLP's whitespace mode as a parameter, never a global, and works on
//! the bytes of the view it reads: every offset it writes is a UTF-8 byte
//! offset into that view, on a character boundary. Each has a plain
//! function underneath ([`newline_spans`], [`paragraph_spans`],
//! [`SentenceDetector::detect`], [`RuleSplitter::split`]) for callers
//! outside a pipeline. docs/segmentation.md describes them all.

pub mod eos;
pub mod error;
pub mod features;
pub mod inventory;
pub mod me;
pub mod newline;
pub mod paragraph;
pub mod rules;

mod abbreviation;
pub mod context;
mod scan;
mod units;

pub use eos::{EndOfSentence, Position, Scanned};
pub use error::ModelError;
pub use features::Features;
pub use inventory::{
    Abbreviation, AbbreviationClass, AbbreviationClasses, AbbreviationError, AbbreviationInventories,
    AbbreviationInventory, AbbreviationPreset, ContextRule, Verdict,
};
pub use me::{Candidate, Decision, Scratch, SentenceDetector, SentenceModel};
pub use newline::{NewlineDetector, newline_spans};
pub use paragraph::{ParagraphDetector, paragraph_spans};
pub use rules::{
    LineBreaks, RuleScratch, RuleSentenceDetector, RuleSplitter, Rules, UNPUNCTUATED_SCRIPTS, UnpunctuatedSplitter,
    has_unpunctuated_script, is_unpunctuated_script, needs_unpunctuated_fallback,
};
pub use turbo_unicode::WhitespaceMode;

use turbo_analysis::Error;

/// Refuses a text whose offsets would not fit in a `u32`.
fn check_len(annotator: &'static str, text: &str) -> Result<(), Error> {
    if u32::try_from(text.len()).is_ok() {
        Ok(())
    } else {
        Err(Error::Annotator {
            annotator,
            detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
        })
    }
}

/// The configuration hash of a detector whose one option is the
/// whitespace mode: FNV-1a over a number per mode, fixed so that stored
/// provenance stays comparable across builds.
fn mode_hash(mode: WhitespaceMode) -> u64 {
    let x: u64 = match mode {
        WhitespaceMode::Legacy => 1,
        WhitespaceMode::Unicode => 2,
    };
    (0xcbf2_9ce4_8422_2325 ^ x).wrapping_mul(0x0100_0000_01b3)
}

/// `h` with the eight bytes of `x` folded in, FNV-1a: a configuration
/// hash extended with an abbreviation inventory's hash.
fn fold_hash(mut h: u64, x: u64) -> u64 {
    for b in x.to_le_bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}
