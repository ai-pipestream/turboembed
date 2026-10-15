//! Reading the decoded affix file: OpenNLP's `parseAffix`.
//!
//! The tables (`FLAG`, `AF`, `ICONV`, `OCONV`, `AM`, `COMPOUNDRULE`,
//! `CHECKCOMPOUNDPATTERN`, `REP`, `BREAK`) and `IGNORE` are read first, in
//! that order, each over the whole file, so a later declaration applies to
//! earlier lines; then one pass reads the `PFX` and `SFX` blocks and the
//! other directives line by line. The first error in that order is the one
//! reported, as in Java.

use std::sync::Arc;

use crate::affix::{Affix, AffixKind, Fields};
use crate::compound_rule::CompoundRule;
use crate::condition::Condition;
use crate::conversion::{Conversion, ConversionTable};
use crate::error::LoadError;
use crate::flag::{Flag, FlagMode, FlagSet};
use crate::read::{NO_MATERIAL, directive, table};
use crate::settings::{CompoundPattern, Settings, WordBreak};
use crate::text::{is_count, parse_int, remove_ignored, split_fields, split_lines};

/// What the affix file declares.
pub(crate) struct AffixFile {
    pub(crate) flag_mode: FlagMode,
    /// The `AF` table: flag runs a word list or affix rule names by their
    /// position from 1.
    pub(crate) flag_aliases: Vec<Vec<Flag>>,
    /// The `AM` table: morphological fields named by their position from 1.
    pub(crate) morphology_aliases: Vec<Fields>,
    pub(crate) prefixes: Vec<Affix>,
    pub(crate) suffixes: Vec<Affix>,
    pub(crate) settings: Settings,
    /// `IGNORE`: characters removed from words, affixes and input.
    pub(crate) ignored: Box<str>,
    pub(crate) input_conversion: Conversion,
    pub(crate) output_conversion: Conversion,
    pub(crate) replacements: Conversion,
    pub(crate) compound_rules: Vec<CompoundRule>,
    pub(crate) compound_patterns: Vec<CompoundPattern>,
    pub(crate) word_breaks: Vec<WordBreak>,
}

/// Reads the affix file, its byte order mark already removed.
pub(crate) fn parse(content: &str) -> Result<AffixFile, LoadError> {
    let lines: Vec<Vec<&str>> = split_lines(content).into_iter().map(split_fields).collect();
    let flag_mode = read_flag_mode(&lines)?;
    let flag_aliases = read_flag_aliases(&lines, flag_mode)?;
    let input_conversion = Conversion::parse(&lines, ConversionTable::Input)?;
    let output_conversion = Conversion::parse(&lines, ConversionTable::Output)?;
    let morphology_aliases = read_morphology_aliases(&lines)?;
    let compound_rules = read_compound_rules(&lines, flag_mode)?;
    let compound_patterns = read_compound_patterns(&lines, flag_mode)?;
    let replacements = Conversion::parse(&lines, ConversionTable::Replacement)?;
    let word_breaks = read_word_breaks(&lines)?;
    let mut ignored: Box<str> = "".into();
    for (i, fields) in lines.iter().enumerate() {
        if fields.first() == Some(&directive::IGNORE) {
            if fields.len() != 2 {
                return Err(LoadError::InvalidIgnore { line: i + 1 });
            }
            ignored = fields[1].into();
        }
    }
    let mut file = AffixFile {
        flag_mode,
        flag_aliases,
        morphology_aliases,
        prefixes: Vec::new(),
        suffixes: Vec::new(),
        settings: Settings::default(),
        ignored,
        input_conversion,
        output_conversion,
        replacements,
        compound_rules,
        compound_patterns,
        word_breaks,
    };
    let mut i = 0;
    while i < lines.len() {
        let fields = &lines[i];
        match fields.first() {
            None => i += 1,
            Some(name) if name.starts_with('#') => i += 1,
            Some(&directive::PREFIX | &directive::SUFFIX) => i = read_affix_block(&lines, i, &mut file)?,
            Some(_) => {
                let line = i + 1;
                if !read_flag_declaration(fields, line, &mut file)? && !read_compound_setting(fields, line, &mut file)?
                {
                    read_general_setting(fields, line, &mut file.settings)?;
                }
                i += 1;
            }
        }
    }
    Ok(file)
}

/// The `FLAG` declaration; there may be at most one.
fn read_flag_mode(lines: &[Vec<&str>]) -> Result<FlagMode, LoadError> {
    let mut mode = None;
    for (i, fields) in lines.iter().enumerate() {
        if fields.first() != Some(&directive::FLAG) {
            continue;
        }
        let line = i + 1;
        if mode.is_some() {
            return Err(LoadError::MultipleFlagModes { line });
        }
        let Some(&name) = fields.get(1) else {
            return Err(LoadError::FlagModeMissing { line });
        };
        mode = Some(match name {
            "long" => FlagMode::Long,
            "num" => FlagMode::Num,
            "UTF-8" => FlagMode::Char,
            _ => return Err(LoadError::UnsupportedFlagMode { mode: name.into(), line }),
        });
    }
    Ok(mode.unwrap_or_default())
}

/// The `AF` table. Its first line holds the count; each later one a flag
/// run in its second field.
fn read_flag_aliases(lines: &[Vec<&str>], mode: FlagMode) -> Result<Vec<Vec<Flag>>, LoadError> {
    let mut aliases = Vec::new();
    let mut expected: Option<i32> = None;
    for (i, fields) in lines.iter().enumerate() {
        if fields.first() != Some(&directive::ALIAS) {
            continue;
        }
        let line = i + 1;
        if fields.len() < 2 {
            return Err(LoadError::AliasValueMissing { line });
        }
        match expected {
            None => {
                let n = parse_value(fields, line)?;
                if n < 0 {
                    return Err(LoadError::NegativeAliasCount { line });
                }
                expected = Some(n);
            }
            Some(_) => aliases.push(mode.parse_run(fields[1], line)?),
        }
    }
    match expected {
        Some(n) if aliases.len() != n as usize => Err(LoadError::AliasCount { expected: n, found: aliases.len() }),
        _ => Ok(aliases),
    }
}

/// The `AM` table: each entry's fields after the directive.
fn read_morphology_aliases(lines: &[Vec<&str>]) -> Result<Vec<Fields>, LoadError> {
    let Some(entries) = table::read(lines, directive::MORPHOLOGY_ALIAS, 2, usize::MAX)? else {
        return Ok(Vec::new());
    };
    Ok(entries.iter().map(|e| fields_of(&e.fields[1..])).collect())
}

fn fields_of(fields: &[&str]) -> Fields {
    fields.iter().map(|&f| Box::<str>::from(f)).collect::<Vec<_>>().into()
}

fn read_compound_rules(lines: &[Vec<&str>], mode: FlagMode) -> Result<Vec<CompoundRule>, LoadError> {
    let Some(entries) = table::read(lines, directive::COMPOUND_RULE, 2, 2)? else {
        return Ok(Vec::new());
    };
    entries.iter().map(|e| CompoundRule::parse(e.fields[1], e.line, mode)).collect()
}

/// `CHECKCOMPOUNDPATTERN`: `end[/flag] begin[/flag] [replacement]`; an
/// end of `0` asks for a left part without affixes.
fn read_compound_patterns(lines: &[Vec<&str>], mode: FlagMode) -> Result<Vec<CompoundPattern>, LoadError> {
    let Some(entries) = table::read(lines, directive::COMPOUND_PATTERN, 3, 4)? else {
        return Ok(Vec::new());
    };
    let mut patterns = Vec::with_capacity(entries.len());
    for e in &entries {
        let (left, left_flag) = split_flag(e.fields[1]);
        let (right, right_flag) = split_flag(e.fields[2]);
        let end_flag = match left_flag {
            Some(text) => mode.parse_one(text, e.line)?,
            None => Flag::NONE,
        };
        let begin_flag = match right_flag {
            Some(text) => mode.parse_one(text, e.line)?,
            None => Flag::NONE,
        };
        let unaffixed = left == NO_MATERIAL;
        patterns.push(CompoundPattern {
            end: if unaffixed { "".into() } else { left.into() },
            end_flag,
            begin: right.into(),
            begin_flag,
            unaffixed,
            replacement: e.fields.get(3).map(|&r| r.into()),
        });
    }
    Ok(patterns)
}

/// A field and the flag text after its first `/`.
fn split_flag(field: &str) -> (&str, Option<&str>) {
    match field.find('/') {
        Some(slash) => (&field[..slash], Some(&field[slash + 1..])),
        None => (field, None),
    }
}

fn read_word_breaks(lines: &[Vec<&str>]) -> Result<Vec<WordBreak>, LoadError> {
    let Some(entries) = table::read(lines, directive::BREAK, 2, 2)? else {
        return Ok(WordBreak::defaults());
    };
    entries.iter().map(|e| WordBreak::parse(e.fields[1], e.line)).collect()
}

/// `Integer.parseInt` of a directive's second field.
fn parse_value(fields: &[&str], line: usize) -> Result<i32, LoadError> {
    let Some(&value) = fields.get(1) else {
        return Err(LoadError::ValueMissing { directive: fields[0].into(), line });
    };
    parse_int(value).ok_or_else(|| LoadError::MalformedValue { directive: fields[0].into(), line })
}

/// A flag run that may be an `AF` alias: digits name an alias when the
/// file has an `AF` table.
pub(crate) fn parse_aliased_flags(
    text: &str,
    mode: FlagMode,
    aliases: &[Vec<Flag>],
    line: usize,
) -> Result<Vec<Flag>, LoadError> {
    if !aliases.is_empty() && is_count(text) {
        let alias = parse_int(text).ok_or_else(|| LoadError::MalformedFlagAlias { alias: text.into(), line })?;
        if alias < 1 || alias as usize > aliases.len() {
            return Err(LoadError::FlagAliasOutOfRange { alias, line, size: aliases.len() });
        }
        return Ok(aliases[alias as usize - 1].clone());
    }
    mode.parse_run(text, line)
}

/// Morphological fields, or with an `AM` table a single alias number.
pub(crate) fn parse_morphology(fields: &[&str], aliases: &[Fields], line: usize) -> Result<Option<Fields>, LoadError> {
    if fields.is_empty() {
        return Ok(None);
    }
    if aliases.is_empty() {
        return Ok(Some(fields_of(fields)));
    }
    let index = match fields {
        [one] => parse_int(one),
        _ => None,
    };
    match index {
        Some(n) if n >= 1 && n as usize <= aliases.len() => Ok(Some(Arc::clone(&aliases[n as usize - 1]))),
        _ => Err(LoadError::InvalidMorphologyAlias { line }),
    }
}

/// A `PFX` or `SFX` block: the header `PFX flag Y|N count` and `count` rule
/// lines `PFX flag strip affix[/flags] [condition [fields...]]`. Returns the
/// index of the line after the block.
fn read_affix_block(lines: &[Vec<&str>], index: usize, file: &mut AffixFile) -> Result<usize, LoadError> {
    let header = &lines[index];
    let header_line = index + 1;
    if header.len() < 4 {
        return Err(LoadError::MalformedAffixHeader { line: header_line });
    }
    let kind = if header[0] == directive::SUFFIX { AffixKind::Suffix } else { AffixKind::Prefix };
    let flag = file.flag_mode.parse_one(header[1], header_line)?;
    let cross_product = match header[2] {
        "Y" => true,
        "N" => false,
        _ => return Err(LoadError::InvalidCrossProduct { line: header_line }),
    };
    let count = parse_int(header[3]).ok_or(LoadError::MalformedRuleCount { line: header_line })?;
    if count < 0 {
        return Err(LoadError::NegativeRuleCount { line: header_line });
    }
    let mut i = index + 1;
    for _ in 0..count {
        let line = i + 1;
        let Some(fields) = lines.get(i) else {
            return Err(LoadError::TruncatedAffixBlock { line });
        };
        if fields.len() < 4 || fields[0] != header[0] {
            return Err(LoadError::MalformedAffixRule { line });
        }
        if file.flag_mode.parse_one(fields[1], line)? != flag {
            return Err(LoadError::RuleFlagMismatch { line });
        }
        let strip = if fields[2] == NO_MATERIAL { "".into() } else { remove_ignored(fields[2], &file.ignored).into() };
        let (mut add, continuation) = match fields[3].find('/') {
            Some(slash) => (
                &fields[3][..slash],
                parse_aliased_flags(&fields[3][slash + 1..], file.flag_mode, &file.flag_aliases, line)?,
            ),
            None => (fields[3], Vec::new()),
        };
        if add == NO_MATERIAL {
            add = "";
        }
        let add: Box<str> = remove_ignored(add, &file.ignored).into();
        let has_condition = fields.len() > 4;
        let morphology =
            parse_morphology(&fields[if has_condition { 5 } else { 4 }..], &file.morphology_aliases, line)?;
        let condition = Condition::parse(if has_condition { fields[4] } else { "." }, kind == AffixKind::Suffix, line)?;
        let affix = Affix {
            flag,
            flag_text: fields[1].into(),
            cross_product,
            strip,
            add,
            condition,
            continuation: FlagSet::new(continuation),
            kind,
            morphology,
        };
        match kind {
            AffixKind::Suffix => file.suffixes.push(affix),
            AffixKind::Prefix => file.prefixes.push(affix),
        }
        i += 1;
    }
    Ok(i)
}

/// A directive whose value is one flag. Returns whether the line was one.
fn read_flag_declaration(fields: &[&str], line: usize, file: &mut AffixFile) -> Result<bool, LoadError> {
    let name = fields[0];
    if !directive::takes_one_flag(name) {
        return Ok(false);
    }
    let Some(&text) = fields.get(1) else {
        return Err(LoadError::FlagMissing { directive: name.into(), line });
    };
    let flag = file.flag_mode.parse_one(text, line)?;
    let s = &mut file.settings;
    match name {
        "COMPOUNDFLAG" => s.compound_flag = flag,
        "COMPOUNDBEGIN" => s.compound_begin = flag,
        "COMPOUNDMIDDLE" => s.compound_middle = flag,
        "COMPOUNDEND" => s.compound_end = flag,
        "COMPOUNDPERMITFLAG" => s.compound_permit = flag,
        "COMPOUNDFORBIDFLAG" => s.compound_forbid = flag,
        // PSEUDOROOT is the directive's name before Hunspell renamed it.
        "NEEDAFFIX" | "PSEUDOROOT" => s.need_affix = flag,
        "ONLYINCOMPOUND" => s.only_in_compound = flag,
        "CIRCUMFIX" => s.circumfix = flag,
        "FORBIDDENWORD" => s.forbidden_word = flag,
        directive::KEEP_CASE => s.keep_case = flag,
        directive::WARNING => s.warning = flag,
        directive::COMPOUND_ROOT => s.compound_root = flag,
        directive::FORCE_UPPER_CASE => s.force_upper_case = flag,
        // LEMMA_PRESENT is read and has no effect.
        _ => {}
    }
    Ok(true)
}

/// A compound limit or switch. Returns whether the line was one.
fn read_compound_setting(fields: &[&str], line: usize, file: &mut AffixFile) -> Result<bool, LoadError> {
    let s = &mut file.settings;
    match fields[0] {
        "COMPOUNDMIN" => {
            let min = parse_value(fields, line)?;
            if min < 0 {
                return Err(LoadError::NegativeCompoundMin { line });
            }
            if min > i32::MAX / 2 {
                return Err(LoadError::CompoundMinTooLarge { line });
            }
            s.compound_min = min.max(1) as usize;
        }
        "COMPOUNDWORDMAX" => {
            let max = parse_value(fields, line)?;
            if max < 0 {
                return Err(LoadError::NegativeCompoundWordMax { line });
            }
            s.compound_word_max = max as usize;
        }
        directive::COMPOUND_SYLLABLE => {
            let max = parse_value(fields, line)?;
            if fields.len() != 3 || max < 0 {
                return Err(LoadError::InvalidCompoundSyllable { line });
            }
            s.max_compound_syllables = max as usize;
            s.compound_vowels = fields[2].into();
        }
        directive::SYLLABLE_NUMBER => {
            if fields.len() != 2 {
                return Err(LoadError::InvalidSyllableNum { line });
            }
            s.syllable_number = true;
        }
        "CHECKCOMPOUNDDUP" => s.check_compound_dup = true,
        "CHECKCOMPOUNDCASE" => s.check_compound_case = true,
        "CHECKCOMPOUNDTRIPLE" => s.check_compound_triple = true,
        directive::CHECK_COMPOUND_REP => s.check_compound_rep = true,
        directive::COMPOUND_MORE_SUFFIXES => s.compound_more_suffixes = true,
        directive::SIMPLIFIED_TRIPLE => s.simplified_triple = true,
        _ => return Ok(false),
    }
    Ok(true)
}

/// A switch outside compounding, or `LANG`. Every other directive was read
/// before this pass or has no effect on stemming.
fn read_general_setting(fields: &[&str], line: usize, s: &mut Settings) -> Result<(), LoadError> {
    match fields[0] {
        "FULLSTRIP" => s.full_strip = true,
        directive::COMPLEX_PREFIXES => s.complex_prefixes = true,
        directive::FORBID_WARNING => s.forbid_warn = true,
        directive::CHECK_SHARPS => s.check_sharps = true,
        directive::LANGUAGE => {
            if fields.len() != 2 {
                return Err(LoadError::InvalidLanguage { line });
            }
            s.language = fields[1].into();
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_err(text: &str) -> String {
        match parse(text) {
            Ok(_) => panic!("{text:?} parsed"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn blocks_and_settings_are_read() {
        let file = parse(
            "FLAG long\nCOMPOUNDFLAG Cc\nCOMPOUNDMIN 0\nLANG hu_HU\nSFX Aa Y 2\nSFX Aa y ies/Bb [^aeiou]y po:pl\nSFX Aa 0 s .\nPFX Bb N 1\nPFX Bb 0 re .\n",
        )
        .unwrap();
        assert_eq!(file.flag_mode, FlagMode::Long);
        assert_eq!(file.settings.compound_min, 1);
        assert_eq!(&*file.settings.language, "hu_HU");
        assert_eq!(file.suffixes.len(), 2);
        let s = &file.suffixes[0];
        assert_eq!((s.strip(), s.affix(), s.cross_product), ("y", "ies", true));
        assert!(s.allows(Flag(0x42_0062)));
        assert_eq!(s.fields(), [Box::<str>::from("po:pl")]);
        assert_eq!(file.prefixes[0].affix(), "re");
        assert!(!file.prefixes[0].cross_product);
    }

    #[test]
    fn errors_name_their_line() {
        assert_eq!(parse_err("FLAG long\nFLAG num\n"), "multiple FLAG directives at line 2");
        assert_eq!(parse_err("FLAG chars\n"), "unsupported FLAG mode 'chars' at line 1");
        assert_eq!(parse_err("SFX A Y 2\nSFX A 0 s ."), "affix block truncated at line 3");
        assert_eq!(parse_err("SFX A Y 2\nSFX A 0 s .\n"), "malformed affix rule at line 3");
        assert_eq!(parse_err("SFX A Y 1\nPFX A 0 s .\n"), "malformed affix rule at line 2");
        assert_eq!(parse_err("SFX A Y 1\nSFX B 0 s .\n"), "affix rule flag does not match header at line 2");
        assert_eq!(parse_err("SFX A X 1\n"), "invalid cross-product marker at line 1");
        assert_eq!(parse_err("SFX A Y x\n"), "malformed affix rule count at line 1");
        assert_eq!(parse_err("AF 2\nAF A\n"), "AF header specifies 2 aliases but found 1");
        assert_eq!(
            parse_err("AF 1\nAF A\nSFX A Y 1\nSFX A 0 s/2 .\n"),
            "flag alias 2 at line 4 is outside the AF table of 1 aliases"
        );
        assert_eq!(parse_err("AM 1\nAM po:n\nSFX A Y 1\nSFX A 0 s . x\n"), "invalid AM alias at line 4");
        assert_eq!(parse_err("SFX A Y 1\nSFX A 0 s [ab\n"), "unterminated character class at line 2");
        assert_eq!(parse_err("COMPOUNDMIN\n"), "COMPOUNDMIN line without a value at line 1");
        assert_eq!(parse_err("COMPOUNDMIN x\n"), "malformed COMPOUNDMIN at line 1");
        assert_eq!(parse_err("COMPOUNDMIN 1073741824\n"), "COMPOUNDMIN exceeds 1073741823 at line 1");
        assert_eq!(parse_err("NEEDAFFIX\n"), "NEEDAFFIX line without a flag at line 1");
        assert_eq!(parse_err("IGNORE a b\n"), "invalid IGNORE at line 1");
        assert_eq!(parse_err("LANG\n"), "invalid LANG at line 1");
        assert_eq!(parse_err("ICONV 1\n"), "ICONV header specifies 1 entries but found 0");
    }

    #[test]
    fn aliases_expand() {
        let file = parse("AF 1\nAF AB\nAM 1\nAM st:x po:y\nSFX A Y 1\nSFX A 0 s/1 . 1\n").unwrap();
        let s = &file.suffixes[0];
        assert_eq!(s.continuation.as_slice(), [Flag::of('A'), Flag::of('B')]);
        assert_eq!(s.fields().len(), 2);
    }
}
