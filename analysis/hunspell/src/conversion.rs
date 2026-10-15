//! Conversion tables: `ICONV`, `OCONV` and the `REP` table that
//! `CHECKCOMPOUNDREP` checks compounds against. OpenNLP's
//! `HunspellConversion`.

use crate::error::LoadError;
use crate::read::table;
use crate::text::upper_unit;

/// One conversion: text and its replacement, optionally bound to the start
/// or the end of the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rule {
    pub(crate) from: Box<str>,
    pub(crate) to: Box<str>,
    pub(crate) initial: bool,
    pub(crate) terminal: bool,
}

/// A table of conversions, applied without rescanning replaced text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Conversion {
    rules: Vec<Rule>,
    /// The positions of the rules, by the first byte of their text and
    /// then in table order; `starts[b]` is where the rules of byte `b`
    /// begin. A text is tried only against the rules of its byte.
    order: Vec<u32>,
    starts: Vec<u32>,
}

/// Which table a conversion was read from; `REP` marks its anchors with `^`
/// and `$` and may have more than three fields, the others mark an anchor
/// with `_`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConversionTable {
    Input,
    Output,
    Replacement,
}

impl ConversionTable {
    fn tag(self) -> &'static str {
        match self {
            ConversionTable::Input => crate::read::directive::INPUT_CONVERSION,
            ConversionTable::Output => crate::read::directive::OUTPUT_CONVERSION,
            ConversionTable::Replacement => crate::read::directive::REPLACEMENT,
        }
    }
}

impl Conversion {
    /// Reads the table from the affix file's fields. In both fields `_`
    /// stands for a space, and a replacement of `0` is empty.
    pub(crate) fn parse(lines: &[Vec<&str>], which: ConversionTable) -> Result<Conversion, LoadError> {
        let tag = which.tag();
        let replacement = which == ConversionTable::Replacement;
        let max = if replacement { usize::MAX } else { 3 };
        let Some(entries) = table::read(lines, tag, 3, max)? else {
            return Ok(Conversion::default());
        };
        let (start_mark, end_mark) = if replacement { ('^', '$') } else { ('_', '_') };
        let mut rules = Vec::with_capacity(entries.len());
        for entry in entries {
            let pattern = entry.fields[1];
            let initial = pattern.starts_with(start_mark);
            let terminal = pattern.ends_with(end_mark);
            let start = usize::from(initial);
            let end = pattern.len() - usize::from(terminal);
            if end <= start {
                return Err(LoadError::EmptyConversionPattern { tag, line: entry.line });
            }
            let to = entry.fields[2];
            rules.push(Rule {
                from: pattern[start..end].replace('_', " ").into(),
                to: if to == crate::read::NO_MATERIAL { "".into() } else { to.replace('_', " ").into() },
                initial,
                terminal,
            });
        }
        let mut table = Conversion { rules, ..Conversion::default() };
        table.index();
        Ok(table)
    }

    /// Groups the rules by the first byte of their text. Called again after
    /// [`Conversion::push_phonetic`].
    pub(crate) fn index(&mut self) {
        let first = |r: &Rule| r.from.as_bytes()[0];
        self.order = (0..self.rules.len() as u32).collect();
        self.order.sort_by_key(|&i| (first(&self.rules[i as usize]), i));
        self.starts = vec![0; 257];
        for r in &self.rules {
            self.starts[usize::from(first(r)) + 1] += 1;
        }
        for b in 0..256 {
            self.starts[b + 1] += self.starts[b];
        }
    }

    /// The rules whose text starts with the byte at `offset`, in table
    /// order.
    #[inline]
    fn rules_at(&self, input: &str, offset: usize) -> impl Iterator<Item = &Rule> {
        debug_assert_eq!(self.order.len(), self.rules.len(), "the table is indexed");
        let b = usize::from(input.as_bytes()[offset]);
        self.order[self.starts[b] as usize..self.starts[b + 1] as usize].iter().map(|&i| &self.rules[i as usize])
    }

    /// Adds a conversion from `from` to `to`, and from `from` with its first
    /// character upper case when that differs: a `ph:` field of the word
    /// list.
    pub(crate) fn push_phonetic(&mut self, from: &str, to: &str) {
        self.rules.push(Rule { from: from.into(), to: to.into(), initial: false, terminal: false });
        let first = from.chars().next().expect("a ph: pattern is not empty");
        let upper = upper_unit(first);
        if upper != first {
            let mut s = String::with_capacity(from.len() + 2);
            s.push(upper);
            s.push_str(&from[first.len_utf8()..]);
            self.rules.push(Rule { from: s.into(), to: to.into(), initial: false, terminal: false });
        }
    }

    /// Whether the rule matches `input` at byte `offset`.
    fn matches_at(rule: &Rule, input: &str, offset: usize) -> bool {
        (!rule.initial || offset == 0)
            && (!rule.terminal || offset + rule.from.len() == input.len())
            && input[offset..].starts_with(&*rule.from)
    }

    /// Replaces the longest matching pattern at each position, the first
    /// listed of equally long ones, and moves past the replaced text; where
    /// none matches, moves one character.
    pub(crate) fn apply<'a>(&self, input: &'a str) -> std::borrow::Cow<'a, str> {
        if self.rules.is_empty() || input.is_empty() {
            return input.into();
        }
        let mut out = String::new();
        let mut changed = false;
        let mut offset = 0;
        while offset < input.len() {
            let mut selected: Option<&Rule> = None;
            for rule in self.rules_at(input, offset) {
                if Self::matches_at(rule, input, offset) && selected.is_none_or(|s| rule.from.len() > s.from.len()) {
                    selected = Some(rule);
                }
            }
            match selected {
                Some(rule) => {
                    if !changed {
                        out.reserve(input.len());
                        out.push_str(&input[..offset]);
                        changed = true;
                    }
                    out.push_str(&rule.to);
                    offset += rule.from.len();
                }
                None => {
                    let c = input[offset..].chars().next().expect("offset is inside the input");
                    if changed {
                        out.push(c);
                    }
                    offset += c.len_utf8();
                }
            }
        }
        if changed { out.into() } else { input.into() }
    }

    /// Whether one replacement at one position, not several together, turns
    /// `input` into a different text `accepted` accepts.
    pub(crate) fn any_replacement(&self, input: &str, mut accepted: impl FnMut(&str) -> bool) -> bool {
        let mut buf = String::new();
        let mut offset = 0;
        if self.rules.is_empty() {
            return false;
        }
        while offset < input.len() {
            for rule in self.rules_at(input, offset) {
                if Self::matches_at(rule, input, offset) {
                    buf.clear();
                    buf.push_str(&input[..offset]);
                    buf.push_str(&rule.to);
                    buf.push_str(&input[offset + rule.from.len()..]);
                    if buf != input && accepted(&buf) {
                        return true;
                    }
                }
            }
            offset += input[offset..].chars().next().map_or(1, char::len_utf8);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<Vec<&str>> {
        text.lines().map(|l| l.split_whitespace().collect()).collect()
    }

    #[test]
    fn the_longest_match_wins_and_is_not_rescanned() {
        let l = lines("ICONV 3\nICONV a b\nICONV ab x\nICONV _c 0\n");
        let c = Conversion::parse(&l, ConversionTable::Input).unwrap();
        assert_eq!(c.apply("abab"), "xx");
        assert_eq!(c.apply("aab"), "bx");
        assert_eq!(c.apply("cac"), "bc");
        assert!(matches!(c.apply("zz"), std::borrow::Cow::Borrowed("zz")));
    }

    #[test]
    fn anchors_and_spaces() {
        let l = lines("REP 2\nREP ^a_b x\nREP c$ d\n");
        let c = Conversion::parse(&l, ConversionTable::Replacement).unwrap();
        assert!(c.any_replacement("a bc", |s| s == "xc"));
        assert!(c.any_replacement("a bc", |s| s == "a bd"));
        assert!(!c.any_replacement("ca b", |_| true));
        let bad = lines("ICONV 1\nICONV _ x\n");
        assert!(matches!(
            Conversion::parse(&bad, ConversionTable::Input),
            Err(LoadError::EmptyConversionPattern { tag: "ICONV", line: 2 })
        ));
    }
}
