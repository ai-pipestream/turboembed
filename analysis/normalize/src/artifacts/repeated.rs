//! Lines that recur in a document: the running headers, footers and page
//! labels that text extracted from paged documents carries once per page.
//!
//! This follows the duplicate-line rule of the Gopher quality filters
//! (Rae et al. 2021), which measure how much of a document is lines that
//! occur more than once, and reports the lines themselves. Lines are
//! compared after trimming `White_Space` at both ends, with every inner
//! run of `White_Space` read as one space and every run of decimal digits
//! (general category Nd) read as one digit, so "Page 3 of 12" and
//! "Page 4 of  12" are the same line. Only lines with at least one
//! alphabetic character take part: a line of digits and punctuation (a
//! table cell, a bare page number, a list marker) would otherwise match
//! every other such line once digits are folded.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, DefaultHasher, Hash, Hasher};
use std::ops::Range;

use turbo_normalize_layers::ArtifactKind;
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::props;

use super::Artifact;
use super::lines::trimmed_lines;

/// The options of the repeated-line rule.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RepeatedLines {
    /// How many times a line must occur to be reported. At least 2.
    pub min_occurrences: u32,
}

impl Default for RepeatedLines {
    /// Three occurrences: a line on two pages may be a coincidence, a line
    /// on three is a header or a footer.
    fn default() -> RepeatedLines {
        RepeatedLines { min_occurrences: 3 }
    }
}

/// One unit of a line as it is compared: a character, a run of digits, a
/// run of whitespace.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Unit {
    Char(char),
    Digits,
    Space,
}

/// The units of a trimmed line.
fn units(line: &str) -> impl Iterator<Item = Unit> + '_ {
    let mut chars = line.chars().peekable();
    std::iter::from_fn(move || {
        let c = chars.next()?;
        let unit = classify(c);
        if unit != Unit::Char(c) {
            while chars.next_if(|&n| classify(n) == unit).is_some() {}
        }
        Some(unit)
    })
}

fn classify(c: char) -> Unit {
    if props::white_space(c) {
        Unit::Space
    } else if general_category(c) == GeneralCategory::Nd {
        Unit::Digits
    } else {
        Unit::Char(c)
    }
}

/// Reports every occurrence of every line that occurs at least
/// `min_occurrences` times, in text order.
pub(crate) fn scan(text: &str, options: RepeatedLines, out: &mut Vec<Artifact>) {
    let min = options.min_occurrences.max(2) as usize;
    // Lines are grouped by a hash of their units, then each group is split
    // into classes of truly equal lines, so a hash collision cannot join
    // two different lines. The hasher has fixed keys: the result does not
    // depend on the process.
    let mut groups: HashMap<u64, Vec<Range<usize>>, BuildHasherDefault<DefaultHasher>> = HashMap::default();
    for line in trimmed_lines(text).filter(|r| text[r.clone()].chars().any(props::alphabetic)) {
        let mut h = DefaultHasher::new();
        units(&text[line.clone()]).for_each(|u| u.hash(&mut h));
        groups.entry(h.finish()).or_default().push(line);
    }
    let start = out.len();
    for lines in groups.into_values().filter(|g| g.len() >= min) {
        report_classes(text, lines, min, out);
    }
    out[start..].sort_unstable_by_key(|a| a.span.start);
}

/// Splits one hash group into classes of equal lines and reports the
/// classes with enough occurrences.
fn report_classes(text: &str, mut lines: Vec<Range<usize>>, min: usize, out: &mut Vec<Artifact>) {
    while lines.len() >= min {
        let first = lines[0].clone();
        let (same, rest): (Vec<_>, Vec<_>) =
            lines.into_iter().partition(|r| units(&text[r.clone()]).eq(units(&text[first.clone()])));
        if same.len() >= min {
            out.extend(same.into_iter().map(|r| Artifact::new(r.start, r.end, ArtifactKind::RepeatedLine)));
        }
        lines = rest;
    }
}
