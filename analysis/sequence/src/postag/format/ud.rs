use std::fmt;

use super::PennTag;
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
