//! The MultiCoNER v2 kept-corpus reader (issue #1533).
//!
//! MultiCoNER v2 (SemEval-2023 Task 2) is a kept corpus here under CC BY 4.0
//! (docs/licences.md). Its released CoNLL files hold one document after an
//! `# id` header and one sentence per block of lines, an empty line between
//! sentences, and a token, the separator ` _ _ `, then its IOB tag per line:
//!
//! ```text
//! doris _ _ B-Artist
//! day _ _ I-Artist
//! included _ _ O
//! ```
//!
//! A tag is `O`, or `B-` or `I-` with one of the 36 fine-grained types
//! ([`MULTICONER_TYPES`]), kept as written. The reader follows the module's
//! shared IOB conventions and uses its typed errors, not a JVM oracle.
//!
//! # What is pinned here
//!
//! - A row has exactly one ` _ _ ` separator, a nonempty token and a
//!   nonempty tag with no whitespace. Any other row is refused at its line.
//! - IOB is strict: a name starts at `B-` and grows through `I-` tags of
//!   the same type until `O` or the next `B-`. An orphaned or mismatched
//!   continuation is refused at its line.
//! - The first sample after an `# id` header clears adaptive data.
//! - Tokens are the source's own bytes at their UTF-8 spans, including
//!   multi-byte and supplementary characters.

use super::{Sentence, Tag, bio_names};
use crate::error::FormatError;
use crate::sample::NameSample;
use crate::source::Lines;

/// The 36 fine-grained name types of MultiCoNER v2, in the release loader's
/// order. Coarse mappings are not substituted for these names.
pub const MULTICONER_TYPES: [&str; 36] = [
    "Facility",
    "OtherLOC",
    "HumanSettlement",
    "Station",
    "VisualWork",
    "MusicalWork",
    "WrittenWork",
    "ArtWork",
    "Software",
    "OtherCW",
    "MusicalGRP",
    "PublicCorp",
    "PrivateCorp",
    "OtherCorp",
    "AerospaceManufacturer",
    "SportsGRP",
    "CarManufacturer",
    "TechCORP",
    "ORG",
    "Scientist",
    "Artist",
    "Athlete",
    "Politician",
    "Cleric",
    "SportsManager",
    "OtherPER",
    "Clothing",
    "Vehicle",
    "Food",
    "Drink",
    "OtherPROD",
    "Medication/Vaccine",
    "MedicalProcedure",
    "AnatomicalStructure",
    "Symptom",
    "Disease",
];

/// Reads name samples from a MultiCoNER v2 CoNLL file.
#[derive(Clone, Debug)]
pub struct MulticonerReader<'a> {
    lines: Lines<'a>,
    /// Whether the next sentence starts a new document: set when an `# id`
    /// header ends the sentence before it, so the flag survives until the
    /// iterator is called again.
    pending_clear: bool,
}

impl<'a> MulticonerReader<'a> {
    /// A reader of the released CoNLL `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str) -> Result<MulticonerReader<'a>, FormatError> {
        Ok(MulticonerReader { lines: Lines::new(source)?, pending_clear: false })
    }
}

fn kind(text: &str, line: u32) -> Result<&str, FormatError> {
    MULTICONER_TYPES
        .iter()
        .copied()
        .find(|known| *known == text)
        .ok_or_else(|| FormatError::UnknownNameType { line, kind: text.to_owned() })
}

impl<'a> Iterator for MulticonerReader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut sentence = Sentence::default();
            let mut clear = std::mem::take(&mut self.pending_clear);
            let mut at_end = false;
            loop {
                let Some(line) = self.lines.next() else {
                    at_end = true;
                    break;
                };
                if line.text().is_empty() {
                    break;
                }
                if line.text().starts_with("# id") {
                    if sentence.tokens.is_empty() {
                        clear = true;
                        continue;
                    }
                    self.pending_clear = true;
                    break;
                }
                sentence.cover(&line);
                let Some(separator) = line.text().find(" _ _ ") else {
                    return Some(Err(FormatError::MulticonerFields {
                        line: line.number,
                        text: line.text().to_owned(),
                    }));
                };
                if line.text()[separator + 5..].contains(" _ _ ") {
                    return Some(Err(FormatError::MulticonerFields {
                        line: line.number,
                        text: line.text().to_owned(),
                    }));
                }
                let token = line.piece.slice(0, separator);
                let tag = line.piece.slice(separator + 5, line.text().len());
                if token.text.is_empty() || tag.text.is_empty() || tag.text.contains(char::is_whitespace) {
                    return Some(Err(FormatError::MulticonerFields {
                        line: line.number,
                        text: line.text().to_owned(),
                    }));
                }
                sentence.tokens.push(token.field());
                sentence.tags.push(Tag { text: tag.text, line: line.number });
            }
            if !sentence.tokens.is_empty() {
                sentence.clear_adaptive_data = clear;
                let names = bio_names(&sentence.tags, |tag| tag, kind);
                return sentence.sample(names);
            }
            if at_end {
                return None;
            }
        }
    }
}
