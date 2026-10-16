//! The affix file's directive names and which of them the reader parses,
//! ignores or does not support.

pub(crate) const SET: &str = "SET";
pub(crate) const FLAG: &str = "FLAG";
pub(crate) const ALIAS: &str = "AF";
pub(crate) const PREFIX: &str = "PFX";
pub(crate) const SUFFIX: &str = "SFX";
pub(crate) const INPUT_CONVERSION: &str = "ICONV";
pub(crate) const OUTPUT_CONVERSION: &str = "OCONV";
pub(crate) const IGNORE: &str = "IGNORE";
pub(crate) const KEEP_CASE: &str = "KEEPCASE";
pub(crate) const COMPLEX_PREFIXES: &str = "COMPLEXPREFIXES";
pub(crate) const COMPOUND_RULE: &str = "COMPOUNDRULE";
pub(crate) const COMPOUND_PATTERN: &str = "CHECKCOMPOUNDPATTERN";
pub(crate) const MORPHOLOGY_ALIAS: &str = "AM";
pub(crate) const BREAK: &str = "BREAK";
pub(crate) const LANGUAGE: &str = "LANG";
pub(crate) const WARNING: &str = "WARN";
pub(crate) const FORBID_WARNING: &str = "FORBIDWARN";
pub(crate) const CHECK_SHARPS: &str = "CHECKSHARPS";
pub(crate) const COMPOUND_ROOT: &str = "COMPOUNDROOT";
pub(crate) const FORCE_UPPER_CASE: &str = "FORCEUCASE";
pub(crate) const COMPOUND_MORE_SUFFIXES: &str = "COMPOUNDMORESUFFIXES";
pub(crate) const SIMPLIFIED_TRIPLE: &str = "SIMPLIFIEDTRIPLE";
pub(crate) const SYLLABLE_NUMBER: &str = "SYLLABLENUM";
pub(crate) const LEMMA_PRESENT: &str = "LEMMA_PRESENT";
pub(crate) const COMPOUND_SYLLABLE: &str = "COMPOUNDSYLLABLE";
pub(crate) const CHECK_COMPOUND_REP: &str = "CHECKCOMPOUNDREP";
pub(crate) const REPLACEMENT: &str = "REP";

/// Whether the reader reads the directive's fields.
pub(crate) fn is_parsed(name: &str) -> bool {
    matches!(
        name,
        SET | FLAG
            | ALIAS
            | PREFIX
            | SUFFIX
            | "COMPOUNDFLAG"
            | "COMPOUNDBEGIN"
            | "COMPOUNDMIDDLE"
            | "COMPOUNDEND"
            | "COMPOUNDPERMITFLAG"
            | "COMPOUNDFORBIDFLAG"
            | "NEEDAFFIX"
            | "PSEUDOROOT"
            | "ONLYINCOMPOUND"
            | "FORBIDDENWORD"
            | "CIRCUMFIX"
            | "COMPOUNDMIN"
            | "COMPOUNDWORDMAX"
            | "CHECKCOMPOUNDDUP"
            | "CHECKCOMPOUNDCASE"
            | "CHECKCOMPOUNDTRIPLE"
            | "FULLSTRIP"
            | INPUT_CONVERSION
            | OUTPUT_CONVERSION
            | IGNORE
            | MORPHOLOGY_ALIAS
            | COMPLEX_PREFIXES
            | KEEP_CASE
            | WARNING
            | FORBID_WARNING
            | LANGUAGE
            | CHECK_SHARPS
            | COMPOUND_RULE
            | COMPOUND_ROOT
            | FORCE_UPPER_CASE
            | COMPOUND_MORE_SUFFIXES
            | SIMPLIFIED_TRIPLE
            | CHECK_COMPOUND_REP
            | COMPOUND_PATTERN
            | COMPOUND_SYLLABLE
            | SYLLABLE_NUMBER
            | LEMMA_PRESENT
            | REPLACEMENT
            | BREAK
    )
}

/// Whether the directive is metadata, a suggestion or command line setting
/// or one of the old compound names, which stemming does not use. `REP` is
/// read only when `CHECKCOMPOUNDREP` is declared.
pub(crate) fn is_ignored(name: &str) -> bool {
    matches!(
        name,
        "NAME"
            | "HOME"
            | "VERSION"
            | "KEY"
            | "TRY"
            | REPLACEMENT
            | "MAP"
            | "PHONE"
            | "NOSUGGEST"
            | "MAXCPDSUGS"
            | "MAXNGRAMSUGS"
            | "MAXDIFF"
            | "ONLYMAXDIFF"
            | "NOSPLITSUGS"
            | "SUGSWITHDOTS"
            | "SUBSTANDARD"
            | "WORDCHARS"
            | "NONGRAMSUGGEST"
            | "CHECKNUM"
            | "COMPOUNDFIRST"
            | "COMPOUNDLAST"
            | "ONLYROOT"
            | "HU_KOTOHANGZO"
            | "GENERATE"
    )
}

/// Whether the directive's one value is a flag.
pub(crate) fn takes_one_flag(name: &str) -> bool {
    matches!(
        name,
        "COMPOUNDFLAG"
            | "COMPOUNDBEGIN"
            | "COMPOUNDMIDDLE"
            | "COMPOUNDEND"
            | "COMPOUNDPERMITFLAG"
            | "COMPOUNDFORBIDFLAG"
            | "NEEDAFFIX"
            | "PSEUDOROOT"
            | "ONLYINCOMPOUND"
            | "FORBIDDENWORD"
            | "CIRCUMFIX"
            | KEEP_CASE
            | WARNING
            | COMPOUND_ROOT
            | FORCE_UPPER_CASE
            | LEMMA_PRESENT
    )
}
