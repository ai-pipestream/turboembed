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

/// The 17 Universal Dependencies part-of-speech tags, the keys of
/// OpenNLP's UD to Penn table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UdTag {
    Adj,
    Adp,
    Adv,
    Aux,
    Cconj,
    Det,
    Intj,
    Noun,
    Num,
    Part,
    Pron,
    Propn,
    Punct,
    Sconj,
    Sym,
    Verb,
    X,
}

impl UdTag {
    pub const ALL: [UdTag; 17] = [
        UdTag::Adj,
        UdTag::Adp,
        UdTag::Adv,
        UdTag::Aux,
        UdTag::Cconj,
        UdTag::Det,
        UdTag::Intj,
        UdTag::Noun,
        UdTag::Num,
        UdTag::Part,
        UdTag::Pron,
        UdTag::Propn,
        UdTag::Punct,
        UdTag::Sconj,
        UdTag::Sym,
        UdTag::Verb,
        UdTag::X,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            UdTag::Adj => "ADJ",
            UdTag::Adp => "ADP",
            UdTag::Adv => "ADV",
            UdTag::Aux => "AUX",
            UdTag::Cconj => "CCONJ",
            UdTag::Det => "DET",
            UdTag::Intj => "INTJ",
            UdTag::Noun => "NOUN",
            UdTag::Num => "NUM",
            UdTag::Part => "PART",
            UdTag::Pron => "PRON",
            UdTag::Propn => "PROPN",
            UdTag::Punct => "PUNCT",
            UdTag::Sconj => "SCONJ",
            UdTag::Sym => "SYM",
            UdTag::Verb => "VERB",
            UdTag::X => "X",
        }
    }

    /// The tag a string names, matching case exactly.
    pub fn parse(tag: &str) -> Option<UdTag> {
        Some(match tag {
            "ADJ" => UdTag::Adj,
            "ADP" => UdTag::Adp,
            "ADV" => UdTag::Adv,
            "AUX" => UdTag::Aux,
            "CCONJ" => UdTag::Cconj,
            "DET" => UdTag::Det,
            "INTJ" => UdTag::Intj,
            "NOUN" => UdTag::Noun,
            "NUM" => UdTag::Num,
            "PART" => UdTag::Part,
            "PRON" => UdTag::Pron,
            "PROPN" => UdTag::Propn,
            "PUNCT" => UdTag::Punct,
            "SCONJ" => UdTag::Sconj,
            "SYM" => UdTag::Sym,
            "VERB" => UdTag::Verb,
            "X" => UdTag::X,
            _ => return None,
        })
    }

    /// OpenNLP's UD to Penn table. `AUX` becomes `MD`, `PART` `RP` (never
    /// `TO`), `PUNCT` `.`, and every verb `VB`.
    pub fn to_penn(self) -> PennTag {
        match self {
            UdTag::Adj => PennTag::Jj,
            UdTag::Adp => PennTag::In,
            UdTag::Adv => PennTag::Rb,
            UdTag::Aux => PennTag::Md,
            UdTag::Cconj => PennTag::Cc,
            UdTag::Det => PennTag::Dt,
            UdTag::Intj => PennTag::Uh,
            UdTag::Noun => PennTag::Nn,
            UdTag::Num => PennTag::Cd,
            UdTag::Part => PennTag::Rp,
            UdTag::Pron => PennTag::Prp,
            UdTag::Propn => PennTag::Nnp,
            UdTag::Punct => PennTag::Period,
            UdTag::Sconj => PennTag::In,
            UdTag::Sym => PennTag::Sym,
            UdTag::Verb => PennTag::Vb,
            UdTag::X => PennTag::Fw,
        }
    }
}

impl fmt::Display for UdTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The 47 Penn Treebank tags of OpenNLP's Penn to UD table. The table
/// leaves out some Penn tags, among them the opening quote `` `` `` (an
/// outcome of the OpenNLP 1.5 English models) and the curly brackets
/// `-LCB-` and `-RCB-`; those are not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PennTag {
    /// `#`
    Hash,
    /// `$`
    Dollar,
    /// `''`, the closing quote.
    CloseQuote,
    /// `,`
    Comma,
    /// `-LRB-`
    LeftRoundBracket,
    /// `-RRB-`
    RightRoundBracket,
    /// `.`
    Period,
    /// `:`
    Colon,
    Afx,
    Cc,
    Cd,
    Dt,
    Ex,
    Fw,
    Hyph,
    In,
    Jj,
    Jjr,
    Jjs,
    Ls,
    Md,
    Nil,
    Nn,
    Nnp,
    Nnps,
    Nns,
    Pdt,
    Pos,
    Prp,
    /// `PRP$`
    PrpPossessive,
    Rb,
    Rbr,
    Rbs,
    Rp,
    Sym,
    To,
    Uh,
    Vb,
    Vbd,
    Vbg,
    Vbn,
    Vbp,
    Vbz,
    Wdt,
    Wp,
    /// `WP$`
    WpPossessive,
    Wrb,
}

impl PennTag {
    pub const ALL: [PennTag; 47] = [
        PennTag::Hash,
        PennTag::Dollar,
        PennTag::CloseQuote,
        PennTag::Comma,
        PennTag::LeftRoundBracket,
        PennTag::RightRoundBracket,
        PennTag::Period,
        PennTag::Colon,
        PennTag::Afx,
        PennTag::Cc,
        PennTag::Cd,
        PennTag::Dt,
        PennTag::Ex,
        PennTag::Fw,
        PennTag::Hyph,
        PennTag::In,
        PennTag::Jj,
        PennTag::Jjr,
        PennTag::Jjs,
        PennTag::Ls,
        PennTag::Md,
        PennTag::Nil,
        PennTag::Nn,
        PennTag::Nnp,
        PennTag::Nnps,
        PennTag::Nns,
        PennTag::Pdt,
        PennTag::Pos,
        PennTag::Prp,
        PennTag::PrpPossessive,
        PennTag::Rb,
        PennTag::Rbr,
        PennTag::Rbs,
        PennTag::Rp,
        PennTag::Sym,
        PennTag::To,
        PennTag::Uh,
        PennTag::Vb,
        PennTag::Vbd,
        PennTag::Vbg,
        PennTag::Vbn,
        PennTag::Vbp,
        PennTag::Vbz,
        PennTag::Wdt,
        PennTag::Wp,
        PennTag::WpPossessive,
        PennTag::Wrb,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PennTag::Hash => "#",
            PennTag::Dollar => "$",
            PennTag::CloseQuote => "''",
            PennTag::Comma => ",",
            PennTag::LeftRoundBracket => "-LRB-",
            PennTag::RightRoundBracket => "-RRB-",
            PennTag::Period => ".",
            PennTag::Colon => ":",
            PennTag::Afx => "AFX",
            PennTag::Cc => "CC",
            PennTag::Cd => "CD",
            PennTag::Dt => "DT",
            PennTag::Ex => "EX",
            PennTag::Fw => "FW",
            PennTag::Hyph => "HYPH",
            PennTag::In => "IN",
            PennTag::Jj => "JJ",
            PennTag::Jjr => "JJR",
            PennTag::Jjs => "JJS",
            PennTag::Ls => "LS",
            PennTag::Md => "MD",
            PennTag::Nil => "NIL",
            PennTag::Nn => "NN",
            PennTag::Nnp => "NNP",
            PennTag::Nnps => "NNPS",
            PennTag::Nns => "NNS",
            PennTag::Pdt => "PDT",
            PennTag::Pos => "POS",
            PennTag::Prp => "PRP",
            PennTag::PrpPossessive => "PRP$",
            PennTag::Rb => "RB",
            PennTag::Rbr => "RBR",
            PennTag::Rbs => "RBS",
            PennTag::Rp => "RP",
            PennTag::Sym => "SYM",
            PennTag::To => "TO",
            PennTag::Uh => "UH",
            PennTag::Vb => "VB",
            PennTag::Vbd => "VBD",
            PennTag::Vbg => "VBG",
            PennTag::Vbn => "VBN",
            PennTag::Vbp => "VBP",
            PennTag::Vbz => "VBZ",
            PennTag::Wdt => "WDT",
            PennTag::Wp => "WP",
            PennTag::WpPossessive => "WP$",
            PennTag::Wrb => "WRB",
        }
    }

    /// The tag a string names, matching case exactly.
    pub fn parse(tag: &str) -> Option<PennTag> {
        Some(match tag {
            "#" => PennTag::Hash,
            "$" => PennTag::Dollar,
            "''" => PennTag::CloseQuote,
            "," => PennTag::Comma,
            "-LRB-" => PennTag::LeftRoundBracket,
            "-RRB-" => PennTag::RightRoundBracket,
            "." => PennTag::Period,
            ":" => PennTag::Colon,
            "AFX" => PennTag::Afx,
            "CC" => PennTag::Cc,
            "CD" => PennTag::Cd,
            "DT" => PennTag::Dt,
            "EX" => PennTag::Ex,
            "FW" => PennTag::Fw,
            "HYPH" => PennTag::Hyph,
            "IN" => PennTag::In,
            "JJ" => PennTag::Jj,
            "JJR" => PennTag::Jjr,
            "JJS" => PennTag::Jjs,
            "LS" => PennTag::Ls,
            "MD" => PennTag::Md,
            "NIL" => PennTag::Nil,
            "NN" => PennTag::Nn,
            "NNP" => PennTag::Nnp,
            "NNPS" => PennTag::Nnps,
            "NNS" => PennTag::Nns,
            "PDT" => PennTag::Pdt,
            "POS" => PennTag::Pos,
            "PRP" => PennTag::Prp,
            "PRP$" => PennTag::PrpPossessive,
            "RB" => PennTag::Rb,
            "RBR" => PennTag::Rbr,
            "RBS" => PennTag::Rbs,
            "RP" => PennTag::Rp,
            "SYM" => PennTag::Sym,
            "TO" => PennTag::To,
            "UH" => PennTag::Uh,
            "VB" => PennTag::Vb,
            "VBD" => PennTag::Vbd,
            "VBG" => PennTag::Vbg,
            "VBN" => PennTag::Vbn,
            "VBP" => PennTag::Vbp,
            "VBZ" => PennTag::Vbz,
            "WDT" => PennTag::Wdt,
            "WP" => PennTag::Wp,
            "WP$" => PennTag::WpPossessive,
            "WRB" => PennTag::Wrb,
            _ => return None,
        })
    }

    /// OpenNLP's Penn to UD table. Modals (`MD`) become `VERB`, not `AUX`;
    /// `PRP$` and `WP$` become `DET`; `RP` becomes `ADP`.
    pub fn to_ud(self) -> UdTag {
        match self {
            PennTag::Hash | PennTag::Dollar | PennTag::Sym => UdTag::Sym,
            PennTag::CloseQuote
            | PennTag::Comma
            | PennTag::LeftRoundBracket
            | PennTag::RightRoundBracket
            | PennTag::Period
            | PennTag::Colon
            | PennTag::Hyph => UdTag::Punct,
            PennTag::Afx | PennTag::Jj | PennTag::Jjr | PennTag::Jjs => UdTag::Adj,
            PennTag::Cc => UdTag::Cconj,
            PennTag::Cd => UdTag::Num,
            PennTag::Dt | PennTag::Pdt | PennTag::PrpPossessive | PennTag::Wdt | PennTag::WpPossessive => UdTag::Det,
            PennTag::Ex | PennTag::Prp | PennTag::Wp => UdTag::Pron,
            PennTag::Fw | PennTag::Ls | PennTag::Nil => UdTag::X,
            PennTag::In | PennTag::Rp => UdTag::Adp,
            PennTag::Md | PennTag::Vb | PennTag::Vbd | PennTag::Vbg | PennTag::Vbn | PennTag::Vbp | PennTag::Vbz => {
                UdTag::Verb
            }
            PennTag::Nn | PennTag::Nns => UdTag::Noun,
            PennTag::Nnp | PennTag::Nnps => UdTag::Propn,
            PennTag::Pos | PennTag::To => UdTag::Part,
            PennTag::Rb | PennTag::Rbr | PennTag::Rbs | PennTag::Wrb => UdTag::Adv,
            PennTag::Uh => UdTag::Intj,
        }
    }
}

impl fmt::Display for PennTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

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
