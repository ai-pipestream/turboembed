//! What the declared flags allow: whether an entry stands on its own,
//! takes an affix, may be a compound part, and how compounds are counted.
//! Each takes one entry's flags; the stemmer asks entry by entry, so a
//! reading never combines flags of different homonyms.

use crate::affix::Affix;
use crate::flag::{Flag, FlagSet};
use crate::text::{is_title, is_upper, utf16_len};

use super::case::{CaseType, case_type};
use super::{Dictionary, Homonyms};

/// A part's place in a compound, which decides the positional flag that
/// admits it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CompoundPosition {
    Begin,
    Middle,
    End,
}

/// The flags Hungarian dictionaries reserve for a part before a hyphen.
const HUNGARIAN_HYPHEN_FLAGS: [Flag; 3] = [Flag::of('F'), Flag::of('G'), Flag::of('H')];

impl Dictionary {
    /// Whether the first listed homonym is forbidden, which forbids the
    /// spelling: a dictionary can list a word and then a forbidden
    /// compound-only homonym.
    pub(crate) fn first_forbidden(&self, homonyms: Homonyms<'_>) -> bool {
        homonyms.first().is_some_and(|e| self.flags(e).contains(self.settings.forbidden_word))
    }

    /// Whether the first listed homonym carries `COMPOUNDFORBIDFLAG`, which
    /// keeps the spelling, affixed readings included, out of every part
    /// but the last.
    pub(crate) fn forbids_compound_start(&self, homonyms: Homonyms<'_>) -> bool {
        homonyms.first().is_some_and(|e| self.flags(e).contains(self.settings.compound_forbid))
    }

    /// Whether the entry is a word on its own: not a virtual stem
    /// (`NEEDAFFIX`), compound-only or forbidden.
    pub(crate) fn valid_standalone(&self, flags: &FlagSet) -> bool {
        let s = &self.settings;
        !flags.contains(s.need_affix) && !flags.contains(s.only_in_compound) && !flags.contains(s.forbidden_word)
    }

    /// Whether the entry takes the affix of `flag` outside compounds.
    pub(crate) fn supports(&self, flags: &FlagSet, flag: Flag) -> bool {
        let s = &self.settings;
        flags.contains(flag) && !flags.contains(s.only_in_compound) && !flags.contains(s.forbidden_word)
    }

    /// Whether the entry carries the affix's flag and the forbidden flag,
    /// which forbids the affixed spelling.
    pub(crate) fn forbids_affixed(&self, flags: &FlagSet, flag: Flag) -> bool {
        flags.contains(flag) && flags.contains(self.settings.forbidden_word)
    }

    /// Whether the entry allows both rules: it carries the prefix's flag and
    /// the suffix's (or the prefix's continuation allows the suffix), or it
    /// carries the suffix's and the suffix allows the prefix.
    pub(crate) fn licenses_cross_product(&self, flags: &FlagSet, prefix: &Affix, suffix: &Affix) -> bool {
        (flags.contains(prefix.flag) && (flags.contains(suffix.flag) || prefix.allows(suffix.flag)))
            || (flags.contains(suffix.flag) && suffix.allows(prefix.flag))
    }

    /// [`Dictionary::licenses_cross_product`] for a word outside compounds.
    pub(crate) fn supports_cross_product(&self, flags: &FlagSet, prefix: &Affix, suffix: &Affix) -> bool {
        let s = &self.settings;
        self.licenses_cross_product(flags, prefix, suffix)
            && !flags.contains(s.only_in_compound)
            && !flags.contains(s.forbidden_word)
    }

    /// Whether the affix leaves a virtual stem: a further affix must join.
    pub(crate) fn needs_further_affix(&self, affix: &Affix) -> bool {
        affix.allows(self.settings.need_affix)
    }

    /// Whether the affix applies only inside compounds.
    pub(crate) fn compound_only(&self, affix: &Affix) -> bool {
        affix.allows(self.settings.only_in_compound)
    }

    /// Whether the affix is half of a circumfix and needs a marked affix of
    /// the other kind.
    pub(crate) fn circumfix_only(&self, affix: &Affix) -> bool {
        affix.allows(self.settings.circumfix)
    }

    /// Whether the affix keeps its word out of compounds.
    pub(crate) fn forbids_in_compound(&self, affix: &Affix) -> bool {
        affix.allows(self.settings.compound_forbid)
    }

    /// Whether the affix may face another part (`COMPOUNDPERMITFLAG`).
    pub(crate) fn permits_inside(&self, affix: &Affix) -> bool {
        affix.allows(self.settings.compound_permit)
    }

    /// Whether any compound flag or rule is declared.
    pub(crate) fn compounds_declared(&self) -> bool {
        self.positional_compounds_declared() || !self.compound_rules.is_empty()
    }

    /// Whether `COMPOUNDFLAG` or a positional flag is declared.
    pub(crate) fn positional_compounds_declared(&self) -> bool {
        let s = &self.settings;
        [s.compound_flag, s.compound_begin, s.compound_middle, s.compound_end].iter().any(|&f| f != Flag::NONE)
    }

    fn positional_flag(&self, position: CompoundPosition) -> Flag {
        match position {
            CompoundPosition::Begin => self.settings.compound_begin,
            CompoundPosition::Middle => self.settings.compound_middle,
            CompoundPosition::End => self.settings.compound_end,
        }
    }

    /// Whether a `COMPOUNDFORBIDFLAG` entry is kept from the position: it may
    /// only be the last part.
    fn forbidden_at(&self, flags: &FlagSet, position: CompoundPosition) -> bool {
        position != CompoundPosition::End && flags.contains(self.settings.compound_forbid)
    }

    /// Whether the entry may stand unaffixed at the position. A
    /// `NEEDAFFIX` entry still needs an affix there.
    pub(crate) fn may_stand(&self, flags: &FlagSet, position: CompoundPosition) -> bool {
        let s = &self.settings;
        (flags.contains(s.compound_flag) || flags.contains(self.positional_flag(position)))
            && !flags.contains(s.forbidden_word)
            && !flags.contains(s.need_affix)
            && !self.forbidden_at(flags, position)
    }

    /// Whether the entry, with the affix of `affix_flag` undone, may stand
    /// at the position: the entry carries the flag and either the affix
    /// admits the position or the entry does.
    pub(crate) fn supports_part(
        &self,
        flags: &FlagSet,
        affix_flag: Flag,
        position: CompoundPosition,
        affix_admits: bool,
    ) -> bool {
        let s = &self.settings;
        flags.contains(affix_flag)
            && !flags.contains(s.forbidden_word)
            && !self.forbidden_at(flags, position)
            && (affix_admits || flags.contains(s.compound_flag) || flags.contains(self.positional_flag(position)))
    }

    /// Whether the affix's continuation classes carry `COMPOUNDFLAG` or the
    /// position's flag, so its form may stand at the position.
    pub(crate) fn affix_admits(&self, affix: &Affix, position: CompoundPosition) -> bool {
        affix.allows(self.settings.compound_flag) || affix.allows(self.positional_flag(position))
    }

    /// [`Dictionary::licenses_cross_product`] for a compound part.
    pub(crate) fn supports_compound_cross_product(
        &self,
        flags: &FlagSet,
        position: CompoundPosition,
        prefix: &Affix,
        suffix: &Affix,
        additional: Option<&Affix>,
    ) -> bool {
        let position_allowed = flags.contains(self.settings.compound_flag)
            || flags.contains(self.positional_flag(position))
            || self.affix_admits(prefix, position)
            || self.affix_admits(suffix, position)
            || additional.is_some_and(|a| self.affix_admits(a, position));
        self.licenses_cross_product(flags, prefix, suffix)
            && !flags.contains(self.settings.forbidden_word)
            && !self.forbidden_at(flags, position)
            && position_allowed
    }

    /// Whether the entry and affixes may be a part of a `COMPOUNDRULE`
    /// compound.
    pub(crate) fn accepts_rule_part(&self, flags: &FlagSet, last: bool, affixes: &[&Affix]) -> bool {
        let s = &self.settings;
        if flags.contains(s.forbidden_word)
            || (!last && flags.contains(s.compound_forbid))
            || (affixes.is_empty() && flags.contains(s.need_affix))
        {
            return false;
        }
        if affixes.iter().any(|a| self.forbids_in_compound(a) || self.circumfix_only(a)) {
            return false;
        }
        affixes.len() != 1 || !self.needs_further_affix(affixes[0])
    }

    /// Whether the entry opens a compound before a hyphen under the
    /// Hungarian rule: it carries `F`, `G` or `H` and is not forbidden.
    pub(crate) fn opens_hyphenated_compound(&self, flags: &FlagSet) -> bool {
        !flags.contains(self.settings.forbidden_word) && HUNGARIAN_HYPHEN_FLAGS.iter().any(|&f| flags.contains(f))
    }

    /// Whether a compound ending in the entry may be written as `word`:
    /// `FORCEUCASE` asks for an upper or title case first character.
    pub(crate) fn compound_case_allowed(&self, flags: &FlagSet, word: &str) -> bool {
        !flags.contains(self.settings.force_upper_case)
            || word.chars().next().is_some_and(|c| is_upper(c) || is_title(c))
    }

    /// Whether the reading keeps to the case and warning restrictions:
    /// `FORBIDWARN` rejects a `WARN` entry or affix, a forbidden affix is
    /// rejected, and a `KEEPCASE` entry or affix must be written in its own
    /// case (`surface` equal to `variant`), which `CHECKSHARPS` relaxes for
    /// a sharp s written `SS` or capitalized.
    pub(crate) fn accepts_case(&self, flags: &FlagSet, surface: &str, variant: &str, affixes: &[&Affix]) -> bool {
        let s = &self.settings;
        if s.forbid_warn && flags.contains(s.warning) {
            return false;
        }
        let mut preserve_case = flags.contains(s.keep_case);
        for a in affixes {
            if a.allows(s.forbidden_word) || (s.forbid_warn && a.allows(s.warning)) {
                return false;
            }
            preserve_case |= a.allows(s.keep_case);
        }
        if !preserve_case || surface == variant {
            return true;
        }
        s.check_sharps
            && variant.contains('ß')
            && (surface.contains("SS")
                || (case_type(surface) != CaseType::Upper && surface.chars().next().is_some_and(is_upper)))
    }

    /// How many words a part counts toward `COMPOUNDWORDMAX`: two for a
    /// `COMPOUNDROOT` entry, and in Hungarian one more for each prefix of
    /// more than one syllable.
    pub(crate) fn compound_units(&self, flags: &FlagSet, affixes: &[&Affix]) -> usize {
        let mut units = if flags.contains(self.settings.compound_root) { 2 } else { 1 };
        if self.hungarian {
            units += affixes.iter().filter(|a| !a.is_suffix() && self.count_syllables(&a.add) > 1).count();
        }
        units
    }

    /// Whether a compound of `units` words and `syllables` syllables keeps
    /// to `COMPOUNDWORDMAX` and `COMPOUNDSYLLABLE`.
    pub(crate) fn compound_size_allowed(&self, syllables: i64, units: usize) -> bool {
        let s = &self.settings;
        if s.compound_word_max == 0 || units <= s.compound_word_max {
            return true;
        }
        s.max_compound_syllables != 0 && syllables <= s.max_compound_syllables as i64
    }

    /// A part's syllables toward `COMPOUNDSYLLABLE`: the last part's vowels,
    /// and in Hungarian every part's, adjusted for the last part's suffix.
    pub(crate) fn compound_syllables(&self, part: &str, flags: &FlagSet, affixes: &[&Affix], last: bool) -> i64 {
        if !self.hungarian {
            return if last { self.count_syllables(part) as i64 } else { 0 };
        }
        let mut result = self.count_syllables(part) as i64;
        if !last {
            return result;
        }
        let suffix = affixes.iter().rev().find(|a| a.is_suffix());
        if affixes.is_empty() && flags.contains(Flag::of('I')) && !flags.contains(Flag::of('J')) {
            return result - 1;
        }
        if let Some(suffix) = suffix {
            let add = &*suffix.add;
            if suffix.continuation.is_empty() {
                result -= self.count_syllables(add) as i64;
            } else if add.ends_with('i') && !add.ends_with("yi") && !add.ends_with("ti") {
                result -= 1;
            }
            if self.settings.syllable_number {
                if suffix.flag == Flag::of('c') {
                    result += 2;
                } else if suffix.flag == Flag::of('J')
                    || (suffix.flag == Flag::of('I') && flags.contains(Flag::of('J')))
                {
                    result += 1;
                }
            }
        }
        result
    }

    /// The characters of `text` that `COMPOUNDSYLLABLE` lists as vowels.
    fn count_syllables(&self, text: &str) -> usize {
        let vowels = &*self.settings.compound_vowels;
        if vowels.is_empty() {
            return 0;
        }
        text.chars().filter(|&c| vowels.contains(c)).count()
    }

    /// Whether a `CHECKCOMPOUNDREP` replacement turns `word` into a word
    /// `recognized` accepts, which rules out reading it as a compound.
    /// A replacement shortens the word by at most its longest pattern, so a
    /// word longer than that and [`Dictionary::may_read`] allows is not
    /// tried: no replacement can give a word with a reading.
    pub(crate) fn rejects_compound_replacement(&self, word: &str, recognized: impl FnMut(&str) -> bool) -> bool {
        self.settings.check_compound_rep
            && self.may_read(utf16_len(word).saturating_sub(self.replacements.longest_pattern()))
            && self.replacements.any_replacement(word, recognized)
    }
}
