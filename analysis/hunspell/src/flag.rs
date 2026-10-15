//! Flags: how the affix file writes them (`FLAG`), and the flag sets of
//! entries and affix rules.

use crate::error::LoadError;
use crate::text::{parse_int, trim};

/// One flag. Its value depends on the file's [`FlagMode`]: a code point, a
/// pair of UTF-16 units packed into 32 bits, or a number. The value 0 is
/// what a declaration the file leaves out holds ([`Flag::NONE`]); no entry
/// carries it, even one whose flags include a NUL character.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Flag(pub(crate) u32);

impl Flag {
    /// The flag of an undeclared directive, which no set contains.
    pub const NONE: Flag = Flag(0);

    /// The flag's value.
    pub fn value(self) -> u32 {
        self.0
    }

    /// A flag of the default mode: the code point `c`.
    pub(crate) const fn of(c: char) -> Flag {
        Flag(c as u32)
    }
}

/// The flags of a dictionary entry or of an affix rule's continuation
/// classes, in the order the file lists them. Order matters only to
/// equality, which tells homonyms with the same flags apart from others.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct FlagSet(Box<[Flag]>);

impl FlagSet {
    pub(crate) fn new(flags: Vec<Flag>) -> FlagSet {
        FlagSet(flags.into_boxed_slice())
    }

    /// Whether the set holds `flag`; never for [`Flag::NONE`].
    #[inline]
    pub fn contains(&self, flag: Flag) -> bool {
        flag != Flag::NONE && self.0.contains(&flag)
    }

    pub fn as_slice(&self) -> &[Flag] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// The `FLAG` declaration: how a run of flags is written.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FlagMode {
    /// One flag per code point; also what `FLAG UTF-8` declares. A
    /// variation selector (U+FE00 to U+FE0F) is dropped from a run.
    #[default]
    Char,
    /// `FLAG long`: each two UTF-16 units are one flag.
    Long,
    /// `FLAG num`: decimal numbers from 1 to 65535 separated by commas.
    Num,
}

/// The largest flag `FLAG num` allows: the whole unsigned 16-bit range,
/// which published dictionaries use although the manual names 65000.
const MAX_NUMERIC_FLAG: i32 = 65_535;

impl FlagMode {
    /// Parses a flag run (without its `/`). An empty run has no flags.
    pub(crate) fn parse_run(self, text: &str, line: usize) -> Result<Vec<Flag>, LoadError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            FlagMode::Num => text
                .split(',')
                .map(|part| {
                    let value = trim(part);
                    let n = parse_int(value).ok_or(LoadError::MalformedNumericFlag { line })?;
                    if !(1..=MAX_NUMERIC_FLAG).contains(&n) {
                        return Err(LoadError::NumericFlagRange { line, value: value.into() });
                    }
                    Ok(Flag(n as u32))
                })
                .collect(),
            FlagMode::Long => {
                let units: Vec<u16> = text.encode_utf16().collect();
                if !units.len().is_multiple_of(2) {
                    return Err(LoadError::OddLongFlags { line });
                }
                Ok(units.as_chunks::<2>().0.iter().map(|p| Flag((u32::from(p[0]) << 16) | u32::from(p[1]))).collect())
            }
            FlagMode::Char => {
                Ok(text.chars().filter(|c| !('\u{FE00}'..='\u{FE0F}').contains(c)).map(Flag::of).collect())
            }
        }
    }

    /// Parses a field that holds exactly one flag.
    pub(crate) fn parse_one(self, text: &str, line: usize) -> Result<Flag, LoadError> {
        match self.parse_run(text, line)?.as_slice() {
            [flag] => Ok(*flag),
            _ => Err(LoadError::ExpectedOneFlag { line }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_parse_in_each_mode() {
        assert_eq!(
            FlagMode::Char.parse_run("AB\u{FE0F}\u{1F600}", 1).unwrap(),
            [Flag::of('A'), Flag::of('B'), Flag::of('\u{1F600}')]
        );
        assert_eq!(FlagMode::Long.parse_run("AaBb", 1).unwrap(), [Flag(0x41_0061), Flag(0x42_0062)]);
        assert_eq!(FlagMode::Long.parse_run("\u{1F600}", 1).unwrap(), [Flag(0xD83D_DE00)]);
        assert!(matches!(FlagMode::Long.parse_run("abc", 3), Err(LoadError::OddLongFlags { line: 3 })));
        assert_eq!(FlagMode::Num.parse_run("1, 65535", 1).unwrap(), [Flag(1), Flag(65535)]);
        assert!(matches!(FlagMode::Num.parse_run("0", 1), Err(LoadError::NumericFlagRange { .. })));
        assert!(matches!(FlagMode::Num.parse_run("1,,2", 1), Err(LoadError::MalformedNumericFlag { .. })));
        assert!(FlagMode::Char.parse_run("", 1).unwrap().is_empty());
        assert!(matches!(FlagMode::Char.parse_one("AB", 2), Err(LoadError::ExpectedOneFlag { line: 2 })));
    }

    #[test]
    fn no_set_holds_the_absent_flag() {
        let set = FlagSet::new(vec![Flag(0), Flag(5)]);
        assert!(!set.contains(Flag::NONE));
        assert!(set.contains(Flag(5)));
    }
}
