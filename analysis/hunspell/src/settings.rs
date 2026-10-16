//! The affix file's flag declarations and switches, and its `BREAK` and
//! `CHECKCOMPOUNDPATTERN` tables.

use crate::affix::Affix;
use crate::error::LoadError;
use crate::flag::{Flag, FlagSet};

/// The flags the affix file gives a meaning, each [`Flag::NONE`] when the
/// file does not declare it, and its switches and limits.
#[derive(Clone, Debug)]
pub(crate) struct Settings {
    /// `COMPOUNDFLAG`: the entry may be any part of a compound.
    pub(crate) compound_flag: Flag,
    /// `COMPOUNDBEGIN`, `COMPOUNDMIDDLE`, `COMPOUNDEND`: the entry may be
    /// the first, an inner or the last part.
    pub(crate) compound_begin: Flag,
    pub(crate) compound_middle: Flag,
    pub(crate) compound_end: Flag,
    /// `COMPOUNDPERMITFLAG`: an affix carrying it may face another part.
    pub(crate) compound_permit: Flag,
    /// `COMPOUNDFORBIDFLAG`: an affix carrying it keeps its word out of
    /// compounds, an entry carrying it may only be the last part.
    pub(crate) compound_forbid: Flag,
    /// `NEEDAFFIX` (or `PSEUDOROOT`): the entry is a word only with an affix.
    pub(crate) need_affix: Flag,
    /// `ONLYINCOMPOUND`: the entry or affix appears only inside compounds.
    pub(crate) only_in_compound: Flag,
    /// `FORBIDDENWORD`: the entry is listed to be rejected.
    pub(crate) forbidden_word: Flag,
    /// `CIRCUMFIX`: an affix carrying it needs a carrying affix of the
    /// other kind.
    pub(crate) circumfix: Flag,
    /// `KEEPCASE`: the entry is accepted only in its own case.
    pub(crate) keep_case: Flag,
    /// `WARN`: the entry is rejected under `FORBIDWARN`.
    pub(crate) warning: Flag,
    /// `COMPOUNDROOT`: the entry is itself a compound and counts twice
    /// toward `COMPOUNDWORDMAX`.
    pub(crate) compound_root: Flag,
    /// `FORCEUCASE`: a compound ending in the entry must be capitalized.
    pub(crate) force_upper_case: Flag,
    /// `COMPOUNDMIN`: the fewest characters of a part, at least 1.
    pub(crate) compound_min: usize,
    /// `COMPOUNDWORDMAX`: the most parts, 0 for no limit.
    pub(crate) compound_word_max: usize,
    /// `COMPOUNDSYLLABLE`: the most syllables of a compound longer than
    /// `COMPOUNDWORDMAX` (0 for none), and the vowels that count them.
    pub(crate) max_compound_syllables: usize,
    pub(crate) compound_vowels: Box<str>,
    pub(crate) check_compound_dup: bool,
    pub(crate) check_compound_case: bool,
    pub(crate) check_compound_triple: bool,
    pub(crate) check_compound_rep: bool,
    pub(crate) compound_more_suffixes: bool,
    pub(crate) simplified_triple: bool,
    pub(crate) syllable_number: bool,
    pub(crate) full_strip: bool,
    pub(crate) complex_prefixes: bool,
    pub(crate) forbid_warn: bool,
    pub(crate) check_sharps: bool,
    /// `LANG`, empty without one.
    pub(crate) language: Box<str>,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            compound_flag: Flag::NONE,
            compound_begin: Flag::NONE,
            compound_middle: Flag::NONE,
            compound_end: Flag::NONE,
            compound_permit: Flag::NONE,
            compound_forbid: Flag::NONE,
            need_affix: Flag::NONE,
            only_in_compound: Flag::NONE,
            forbidden_word: Flag::NONE,
            circumfix: Flag::NONE,
            keep_case: Flag::NONE,
            warning: Flag::NONE,
            compound_root: Flag::NONE,
            force_upper_case: Flag::NONE,
            compound_min: 3,
            compound_word_max: 0,
            max_compound_syllables: 0,
            compound_vowels: "".into(),
            check_compound_dup: false,
            check_compound_case: false,
            check_compound_triple: false,
            check_compound_rep: false,
            compound_more_suffixes: false,
            simplified_triple: false,
            syllable_number: false,
            full_strip: false,
            complex_prefixes: false,
            forbid_warn: false,
            check_sharps: false,
            language: "".into(),
        }
    }
}

/// One `BREAK` separator, optionally bound to the start (`^`) or the end
/// (`$`) of the word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WordBreak {
    pub(crate) separator: Box<str>,
    pub(crate) at_start: bool,
    pub(crate) at_end: bool,
}

impl WordBreak {
    /// A separator with its anchors; anchors alone are an error.
    pub(crate) fn parse(declaration: &str, line: usize) -> Result<WordBreak, LoadError> {
        let at_start = declaration.starts_with('^');
        let at_end = declaration.ends_with('$');
        let start = usize::from(at_start);
        let end = declaration.len() - usize::from(at_end);
        if end <= start {
            return Err(LoadError::InvalidBreak { line });
        }
        Ok(WordBreak { separator: declaration[start..end].into(), at_start, at_end })
    }

    /// The separators without a `BREAK` table: a hyphen inside the word, at
    /// its start and at its end.
    pub(crate) fn defaults() -> Vec<WordBreak> {
        [(false, false), (true, false), (false, true)]
            .into_iter()
            .map(|(at_start, at_end)| WordBreak { separator: "-".into(), at_start, at_end })
            .collect()
    }
}

/// One `CHECKCOMPOUNDPATTERN` line: the end of a left part and the start of
/// a right part (each optionally requiring a flag) that may not meet, and
/// optionally the text the junction is written as instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompoundPattern {
    pub(crate) end: Box<str>,
    pub(crate) end_flag: Flag,
    pub(crate) begin: Box<str>,
    pub(crate) begin_flag: Flag,
    /// The left end was written `0`: the left part may carry no affix that
    /// changes it.
    pub(crate) unaffixed: bool,
    pub(crate) replacement: Option<Box<str>>,
}

/// One side of a junction: the part as written, its entry's flags and the
/// affixes undone on it.
pub(crate) struct Side<'a> {
    pub(crate) text: &'a str,
    pub(crate) flags: &'a FlagSet,
    pub(crate) affixes: &'a [&'a Affix],
}

impl CompoundPattern {
    /// Whether the pattern describes the junction of `left` and `right`.
    pub(crate) fn matches(&self, left: &Side<'_>, right: &Side<'_>) -> bool {
        if !left.text.ends_with(&*self.end)
            || !right.text.starts_with(&*self.begin)
            || !has_boundary_flag(left, self.end_flag)
            || !has_boundary_flag(right, self.begin_flag)
        {
            return false;
        }
        !self.unaffixed || left.affixes.iter().all(|a| a.is_identity())
    }
}

/// Whether the entry or one of the affixes carries `required`, or nothing
/// is required.
fn has_boundary_flag(side: &Side<'_>, required: Flag) -> bool {
    required == Flag::NONE || side.flags.contains(required) || side.affixes.iter().any(|a| a.allows(required))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaks_parse_their_anchors() {
        assert_eq!(
            WordBreak::parse("^-", 1).unwrap(),
            WordBreak { separator: "-".into(), at_start: true, at_end: false }
        );
        assert_eq!(WordBreak::parse("--$", 1).unwrap().separator.as_ref(), "--");
        assert!(matches!(WordBreak::parse("^$", 3), Err(LoadError::InvalidBreak { line: 3 })));
        assert!(matches!(WordBreak::parse("$", 3), Err(LoadError::InvalidBreak { line: 3 })));
    }
}
