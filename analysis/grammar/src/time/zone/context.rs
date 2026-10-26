use super::{Region, ZoneMatch};
use crate::engine::{Lexed, NumeralStyle};
use crate::time::spec::{DateSpec, Range, TimeValue};
use turbo_unicode::whitespace::WhitespaceMode;

/// Zone suffixes disambiguate clock ranges only when the source has a
/// range cue or explicit clock notation. A unit such as `pt` alone must
/// not turn font sizes, scores or page numbers into times.
pub(crate) fn for_clock(text: &str, start: u32, end: u32, value: TimeValue, region: Region) -> Option<ZoneMatch> {
    if !super::has_clock(value) {
        return None;
    }
    let zone = super::at(text, end as usize, region)?;
    let prefix =
        text.get(..start as usize)?.trim_end_matches(|c: char| WhitespaceMode::Unicode.is_whitespace(c as u32));
    if prefix
        .chars()
        .next_back()
        .is_some_and(|c| turbo_unicode::category::general_category(c) == turbo_unicode::category::GeneralCategory::Sc)
    {
        return None;
    }
    let previous = prefix.rsplit(|c: char| !c.is_ascii_alphabetic()).next().unwrap_or_default();
    if ["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF", "INR"]
        .iter()
        .any(|code| previous.eq_ignore_ascii_case(code))
    {
        return None;
    }
    if let TimeValue::Range { range: Range::Between { start: a, end: b, .. }, .. } = value
        && a.date == DateSpec::None
        && b.date == DateSpec::None
    {
        let source = text.get(start as usize..end as usize)?;
        let mut lex = Lexed::default();
        lex.lex(
            source,
            &NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false },
        );
        let cue =
            !lex.is_empty() && matches!(lex.word(0), "from" | "between" | "de" | "desde" | "entre" | "tussen" | "van");
        let explicit =
            source.contains(':') || (0..lex.len()).any(|i| matches!(lex.word(i), "am" | "pm" | "h" | "uur" | "horas"));
        if !cue && !explicit {
            return None;
        }
    }
    Some(zone)
}
