use std::fmt;

use super::UdTag;

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
