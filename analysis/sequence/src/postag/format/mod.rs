//! `POSTagFormat` and `POSTagFormatMapper`: which tag set a model's
//! outcomes are in, and the conversion between Penn Treebank tags and
//! Universal Dependencies part-of-speech tags that `POSTaggerME` applies
//! to the tags it returns.
//!
//! The tables are OpenNLP's, after the Universal Dependencies conversion
//! table for English Penn tags. Penn to UD is many to one; UD to Penn
//! picks one Penn tag per UD tag, so a round trip loses information.
//!
//! What OpenNLP does here that may surprise:
//!
//! - `POSTaggerME` asks for UD tags unless told otherwise, so with its
//!   default constructor a Penn model, such as the OpenNLP 1.5 English
//!   ones, returns UD tags.
//! - The Penn table has no opening quote, so a Penn model's `` `` ``
//!   becomes `?`, as does `-LCB-`, `-RCB-` or any tag outside the table.
//! - The format is guessed by counting the outcomes that are keys of each
//!   table; `SYM` is in both. A tie, including a model with no outcome in
//!   either table, is [`PosTagFormat::Unknown`], and a tagger whose model
//!   is of unknown format returns its outcomes unchanged.
//! - Asking for [`PosTagFormat::Unknown`] converts like asking for the
//!   other format: a Penn model returns UD tags and a UD model Penn tags.
//! - When converting from Penn, Java logs a warning about ambiguity for
//!   the UD tags `NOUN`, `PART`, `PROPN`, `PUNCT` and `VERB`, which are not
//!   Penn tags and convert to `?`. Those warnings are not reproduced.

use std::fmt;

/// `POSTagFormat`: the tag set of a part-of-speech model, or the one a
/// caller asks the tagger for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PosTagFormat {
    /// Universal Dependencies part-of-speech tags.
    Ud,
    /// Penn Treebank tags.
    Penn,
    /// Whatever the model's outcomes are, unchanged.
    Custom,
    /// Neither, as far as the guess can tell.
    Unknown,
}

impl PosTagFormat {
    /// The Java constant's name.
    pub fn as_str(self) -> &'static str {
        match self {
            PosTagFormat::Ud => "UD",
            PosTagFormat::Penn => "PENN",
            PosTagFormat::Custom => "CUSTOM",
            PosTagFormat::Unknown => "UNKNOWN",
        }
    }

    /// The format named by a Java constant's name.
    pub fn parse(name: &str) -> Option<PosTagFormat> {
        match name {
            "UD" => Some(PosTagFormat::Ud),
            "PENN" => Some(PosTagFormat::Penn),
            "CUSTOM" => Some(PosTagFormat::Custom),
            "UNKNOWN" => Some(PosTagFormat::Unknown),
            _ => None,
        }
    }
}

impl fmt::Display for PosTagFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The format `POSTaggerME` returns tags in when the caller names none.
pub const DEFAULT_TAG_FORMAT: PosTagFormat = PosTagFormat::Ud;

mod penn;
mod ud;

pub use penn::PennTag;
pub use ud::UdTag;

/// What [`TagFormatMapper::convert_tag`] returns for a tag outside the
/// table it converts with.
pub const UNMAPPED: &str = "?";

/// `POSTagFormatMapper`: the format guessed from a model's outcomes, and
/// the conversion out of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagFormatMapper {
    guessed: PosTagFormat,
}

impl TagFormatMapper {
    /// Guesses the format by majority: [`PosTagFormat::Ud`] when more
    /// outcomes are UD tags than Penn tags, [`PosTagFormat::Penn`] for the
    /// reverse, [`PosTagFormat::Unknown`] on a tie. An outcome counts for
    /// each table it is a key of; listed twice, it counts twice.
    pub fn guess<'a>(outcomes: impl IntoIterator<Item = &'a str>) -> TagFormatMapper {
        let (mut ud, mut penn) = (0usize, 0usize);
        for outcome in outcomes {
            ud += usize::from(UdTag::parse(outcome).is_some());
            penn += usize::from(PennTag::parse(outcome).is_some());
        }
        let guessed = match ud.cmp(&penn) {
            std::cmp::Ordering::Greater => PosTagFormat::Ud,
            std::cmp::Ordering::Less => PosTagFormat::Penn,
            std::cmp::Ordering::Equal => PosTagFormat::Unknown,
        };
        TagFormatMapper { guessed }
    }

    /// `POSTagFormatMapper.NoOp`: converts nothing and reports
    /// [`PosTagFormat::Custom`].
    pub fn no_op() -> TagFormatMapper {
        TagFormatMapper { guessed: PosTagFormat::Custom }
    }

    /// The mapper `POSTaggerME` makes when asked for `requested`: none
    /// when it returns the outcomes unchanged, which is when `requested`
    /// is [`PosTagFormat::Custom`] or the format guessed from `outcomes`.
    pub fn for_tagger<'a>(
        requested: PosTagFormat,
        outcomes: impl IntoIterator<Item = &'a str>,
    ) -> Option<TagFormatMapper> {
        if requested == PosTagFormat::Custom {
            return None;
        }
        let mapper = TagFormatMapper::guess(outcomes);
        (mapper.guessed != requested).then_some(mapper)
    }

    /// `getGuessedFormat`.
    pub fn guessed_format(&self) -> PosTagFormat {
        self.guessed
    }

    /// `convertTag`: from a UD model a UD tag becomes its Penn tag, from a
    /// Penn model a Penn tag its UD tag, and any other tag [`UNMAPPED`].
    /// For a model of unknown or custom format the tag is returned as it
    /// is.
    pub fn convert_tag<'a>(&self, tag: &'a str) -> &'a str {
        match self.guessed {
            PosTagFormat::Ud => UdTag::parse(tag).map_or(UNMAPPED, |t| t.to_penn().as_str()),
            PosTagFormat::Penn => PennTag::parse(tag).map_or(UNMAPPED, |t| t.to_ud().as_str()),
            PosTagFormat::Custom | PosTagFormat::Unknown => tag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for t in UdTag::ALL {
            assert_eq!(UdTag::parse(t.as_str()), Some(t));
        }
        for t in PennTag::ALL {
            assert_eq!(PennTag::parse(t.as_str()), Some(t));
        }
        for f in [PosTagFormat::Ud, PosTagFormat::Penn, PosTagFormat::Custom, PosTagFormat::Unknown] {
            assert_eq!(PosTagFormat::parse(f.as_str()), Some(f));
        }
        assert_eq!(PennTag::parse("``"), None);
        assert_eq!(UdTag::parse("noun"), None);
    }

    #[test]
    fn taggers_convert_only_out_of_the_guessed_format() {
        let penn = ["NN", "VB", "DT"];
        assert_eq!(TagFormatMapper::for_tagger(PosTagFormat::Penn, penn), None);
        assert_eq!(TagFormatMapper::for_tagger(PosTagFormat::Custom, penn), None);
        let m = TagFormatMapper::for_tagger(PosTagFormat::Ud, penn).unwrap();
        assert_eq!(m.convert_tag("NN"), "NOUN");
        assert_eq!(m.convert_tag("``"), UNMAPPED);
        assert_eq!(TagFormatMapper::no_op().convert_tag("``"), "``");
    }
}
