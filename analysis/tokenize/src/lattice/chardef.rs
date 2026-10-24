//! `char.def`: the character categories and which code points belong to
//! them, read as OpenNLP's `MecabDictionary.readCharacterDefinition` and
//! `CategoryTable` read them.
//!
//! A line is cut at its first `#`, trimmed of Unicode `White_Space` and
//! split at whitespace in the configured mode. A line whose first field
//! starts with `0x` (in either case) maps a code point, or a range
//! `0xA..0xB`, to one or more categories, the first of which supplies the
//! unknown word settings; any other line defines a category: its name,
//! INVOKE (`0` or `1`), GROUP (`0` or `1`) and LENGTH (0 to 15). At most
//! 17 categories, and `DEFAULT` must be one. A later mapping wins over an
//! earlier one for the code points both cover.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use turbo_unicode::WhitespaceMode;

use super::error::{CharDefProblem, DictionaryError, DictionaryFile};
use super::number::{after_two_units, parse_java_int};
use super::prebuilt::Table;
use crate::lines::{Lines, trim_unicode_whitespace, whitespace_fields};

/// The category every `char.def` must define, which unmapped code points
/// fall into.
pub(crate) const DEFAULT_CATEGORY: &str = "DEFAULT";

/// MeCab's limit on the number of categories.
const MAX_CATEGORIES: usize = 17;
/// MeCab's limit on a category's LENGTH, a 4-bit field.
const MAX_LENGTH: i32 = 15;
/// The last code point.
const MAX_CODE_POINT: i32 = 0x10_FFFF;
/// Code points with a direct table entry: the Basic Multilingual Plane.
const BMP: usize = 0x1_0000;
/// No mapping.
const NONE: u32 = u32::MAX;

/// The index of a category, in the order `char.def` defines them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CategoryId(pub(crate) u8);

/// One category's unknown word settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Category {
    pub(crate) name: String,
    /// Offer unknown word candidates even where the lexicon matched.
    pub(crate) invoke: bool,
    /// Offer the whole run as one candidate.
    pub(crate) group: bool,
    /// Offer the first 1 to `length` characters of the run.
    pub(crate) length: u8,
}

/// The categories of one code point: the first, which decides the unknown
/// word settings, and the set of all, which decides how far a run goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Assignment {
    pub(crate) primary: CategoryId,
    pub(crate) mask: u32,
}

/// The index of an [`Assignment`] in the table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct AssignmentId(u32);

impl AssignmentId {
    /// A placeholder for a byte that does not start a character.
    pub(crate) const NONE: AssignmentId = AssignmentId(u32::MAX);
}

/// The categories and the code point table.
#[derive(Debug, Clone)]
pub(crate) struct CharDef {
    pub(crate) categories: Vec<Category>,
    pub(crate) default: CategoryId,
    pub(super) assignments: Vec<Assignment>,
    /// The assignment of `DEFAULT` alone, which an unmapped code point gets.
    pub(super) fallback: u32,
    /// The assignment of each code point of the Basic Multilingual Plane.
    pub(super) bmp: Table<u32>,
    /// The supplementary ranges, sorted and disjoint: first, last and
    /// assignment.
    pub(super) ranges: Vec<(u32, u32, u32)>,
}

/// `char.def` as read, before its mappings are resolved against the
/// categories (which `MecabDictionary.load` does last, after `unk.def`).
pub(crate) struct CharDefSource {
    categories: Vec<Category>,
    names: HashMap<String, CategoryId>,
    default: CategoryId,
    /// Each mapping line in file order: its first code point and its
    /// category names.
    mappings: Vec<(u32, Vec<String>)>,
    /// The mapping of each BMP code point.
    bmp: Box<[u32]>,
    /// The supplementary part of each mapping that reaches past the BMP:
    /// first, last and mapping.
    supplementary: Vec<(u32, u32, u32)>,
}

impl CharDefSource {
    pub(crate) fn parse(text: &str, mode: WhitespaceMode) -> Result<CharDefSource, DictionaryError> {
        let mut categories: Vec<Category> = Vec::new();
        let mut names: HashMap<String, CategoryId> = HashMap::new();
        let mut mappings = Vec::new();
        let mut bmp = vec![NONE; BMP].into_boxed_slice();
        let mut supplementary = Vec::new();
        let mut fields: Vec<&str> = Vec::new();
        for (index, raw) in Lines::new(text).enumerate() {
            let line = index + 1;
            let problem = |problem| DictionaryError::CharDef { line, problem };
            let uncommented = raw.find('#').map_or(raw, |i| &raw[..i]);
            let trimmed = trim_unicode_whitespace(uncommented);
            if trimmed.is_empty() {
                continue;
            }
            fields.clear();
            fields.extend(whitespace_fields(trimmed, mode));
            // A trimmed line that is not empty can still be all whitespace
            // in legacy mode (U+001C), and then has no field. Java reads
            // fields[0] of an empty array there and throws; this refuses
            // the line as a malformed category.
            let Some(&first) = fields.first() else {
                return Err(problem(CharDefProblem::MalformedCategory));
            };
            let b = first.as_bytes();
            if b.len() >= 2 && b[0] == b'0' && (b[1] == b'x' || b[1] == b'X') {
                let (from, to) = match first.find("..") {
                    Some(sep) => (code_point(&first[..sep], line)?, code_point(&first[sep + 2..], line)?),
                    None => {
                        let cp = code_point(first, line)?;
                        (cp, cp)
                    }
                };
                if fields.len() < 2 {
                    return Err(problem(CharDefProblem::MappingWithoutCategory));
                }
                if from > to {
                    return Err(problem(CharDefProblem::RangeDescends));
                }
                let mapping = u32::try_from(mappings.len()).map_err(|_| DictionaryError::TooLarge)?;
                mappings.push((from, fields[1..].iter().map(|s| s.to_string()).collect()));
                let bmp_end = to.min(BMP as u32 - 1);
                if from <= bmp_end {
                    bmp[from as usize..=bmp_end as usize].fill(mapping);
                }
                if to >= BMP as u32 {
                    supplementary.push((from.max(BMP as u32), to, mapping));
                }
            } else {
                if fields.len() < 4 {
                    return Err(problem(CharDefProblem::MalformedCategory));
                }
                let flag = |f: &str| match f {
                    "0" => Some(false),
                    "1" => Some(true),
                    _ => None,
                };
                let (Some(invoke), Some(group)) = (flag(fields[1]), flag(fields[2])) else {
                    return Err(problem(CharDefProblem::MalformedFlag));
                };
                let length = parse_java_int(trim_unicode_whitespace(fields[3]), 10)
                    .ok_or(DictionaryError::Number { file: DictionaryFile::CharDef, line })?;
                if !(0..=MAX_LENGTH).contains(&length) {
                    return Err(problem(CharDefProblem::Length));
                }
                if names.contains_key(fields[0]) {
                    return Err(problem(CharDefProblem::DuplicateCategory(fields[0].to_string())));
                }
                if categories.len() >= MAX_CATEGORIES {
                    return Err(problem(CharDefProblem::TooManyCategories));
                }
                names.insert(fields[0].to_string(), CategoryId(categories.len() as u8));
                categories.push(Category { name: fields[0].to_string(), invoke, group, length: length as u8 });
            }
        }
        let Some(&default) = names.get(DEFAULT_CATEGORY) else {
            return Err(DictionaryError::NoDefaultCategory);
        };
        Ok(CharDefSource { categories, names, default, mappings, bmp, supplementary })
    }

    /// The id of the category named `name`, if `char.def` defines it.
    pub(crate) fn category(&self, name: &str) -> Option<CategoryId> {
        self.names.get(name).copied()
    }

    /// The number of categories.
    pub(crate) fn len(&self) -> usize {
        self.categories.len()
    }

    /// Resolves every mapping's names to categories, in file order, failing
    /// at the first mapping that names a category `char.def` does not
    /// define, and builds the code point table.
    pub(crate) fn resolve(self) -> Result<CharDef, DictionaryError> {
        let CharDefSource { categories, names, default, mappings, bmp, mut supplementary } = self;
        let mut assignments: Vec<Assignment> = Vec::new();
        let mut index: HashMap<Assignment, u32> = HashMap::new();
        let mut of_mapping: Vec<u32> = Vec::with_capacity(mappings.len());
        for (from, mapped) in &mappings {
            let mut ids = mapped.iter().map(|n| names.get(n).copied().ok_or(n));
            let first = ids.next().expect("a mapping names a category");
            let primary =
                first.map_err(|n| DictionaryError::UndefinedMapping { code_point: *from, category: n.clone() })?;
            let mut mask = 1u32 << primary.0;
            for id in ids {
                let id =
                    id.map_err(|n| DictionaryError::UndefinedMapping { code_point: *from, category: n.clone() })?;
                mask |= 1 << id.0;
            }
            // Mappings with the same first category and the same set behave
            // the same, so they share one assignment.
            let a = Assignment { primary, mask };
            let next = assignments.len() as u32;
            let id = *index.entry(a).or_insert(next);
            if id == next {
                assignments.push(a);
            }
            of_mapping.push(id);
        }
        // The default assignment, for code points no mapping covers.
        let fallback = Assignment { primary: default, mask: 1 << default.0 };
        let next = assignments.len() as u32;
        let fallback_id = *index.entry(fallback).or_insert(next);
        if fallback_id == next {
            assignments.push(fallback);
        }
        let bmp: Box<[u32]> =
            bmp.iter().map(|&m| if m == NONE { fallback_id } else { of_mapping[m as usize] }).collect();
        let ranges = last_covering(&mut supplementary, &of_mapping);
        Ok(CharDef { categories, default, assignments, fallback: fallback_id, bmp: bmp.into(), ranges })
    }
}

/// The disjoint ranges of the supplementary planes, each with the
/// assignment of the last mapping that covers it, adjacent ranges with the
/// same assignment joined. A sweep over the range starts and ends with a
/// heap of the covering mappings, latest on top.
fn last_covering(bounds: &mut [(u32, u32, u32)], of_mapping: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut edges: Vec<u32> = Vec::with_capacity(bounds.len() * 2);
    for &(a, b, _) in bounds.iter() {
        edges.push(a);
        edges.push(b + 1);
    }
    edges.sort_unstable();
    edges.dedup();
    bounds.sort_unstable_by_key(|&(a, _, _)| a);
    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    let mut heap: BinaryHeap<(u32, Reverse<u32>)> = BinaryHeap::new();
    let mut next = 0;
    for w in edges.windows(2) {
        let (start, end) = (w[0], w[1] - 1);
        while next < bounds.len() && bounds[next].0 <= start {
            heap.push((bounds[next].2, Reverse(bounds[next].1)));
            next += 1;
        }
        // Drop the mappings that end before this range.
        let winner = loop {
            match heap.peek() {
                Some(&(_, Reverse(last))) if last < start => {
                    heap.pop();
                }
                Some(&(m, _)) => break Some(m),
                None => break None,
            }
        };
        let Some(mapping) = winner else { continue };
        let a = of_mapping[mapping as usize];
        match out.last_mut() {
            Some(prev) if prev.1 + 1 == start && prev.2 == a => prev.1 = end,
            _ => out.push((start, end, a)),
        }
    }
    out
}

impl CharDef {
    /// The assignment of code point `cp`: its mapping's, or `DEFAULT`'s.
    #[inline]
    pub(crate) fn assignment_of(&self, cp: u32) -> AssignmentId {
        if (cp as usize) < BMP {
            return AssignmentId(self.bmp[cp as usize]);
        }
        let i = self.ranges.partition_point(|&(_, last, _)| last < cp);
        match self.ranges.get(i) {
            Some(&(first, _, a)) if first <= cp => AssignmentId(a),
            _ => AssignmentId(self.fallback),
        }
    }

    #[inline(always)]
    pub(crate) fn assignment(&self, id: AssignmentId) -> Assignment {
        self.assignments[id.0 as usize]
    }

    #[inline(always)]
    pub(crate) fn category(&self, id: CategoryId) -> &Category {
        &self.categories[id.0 as usize]
    }
}

/// A `char.def` code point: `0x` (or any two UTF-16 code units, for the
/// end of a range) and a hexadecimal `int`, after trimming, which must be
/// a Unicode code point. This is `parseCodePoint`, which takes
/// `substring(2)` of the trimmed field without looking at what it skips.
fn code_point(field: &str, line: usize) -> Result<u32, DictionaryError> {
    let malformed = DictionaryError::CharDef { line, problem: CharDefProblem::MalformedCodePoint };
    let digits = after_two_units(trim_unicode_whitespace(field)).ok_or(malformed)?;
    let value = parse_java_int(digits, 16)
        .ok_or(DictionaryError::CharDef { line, problem: CharDefProblem::MalformedCodePoint })?;
    if !(0..=MAX_CODE_POINT).contains(&value) {
        return Err(DictionaryError::CharDef { line, problem: CharDefProblem::CodePointOutOfRange });
    }
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<CharDef, DictionaryError> {
        CharDefSource::parse(text, WhitespaceMode::Unicode)?.resolve()
    }

    fn message(text: &str) -> String {
        parse(text).expect_err("an error").to_string()
    }

    fn primary(def: &CharDef, cp: u32) -> &str {
        &def.category(def.assignment(def.assignment_of(cp)).primary).name
    }

    const BASE: &str = "DEFAULT 0 1 0\nKANJI 0 0 2\nNUM 1 1 0\n";

    #[test]
    fn categories_and_mappings() {
        let def = parse(&format!(
            "{BASE}0x4E00..0x9FFF KANJI # comment\n0x0030..0x0039 NUM\n0x4E00 NUM KANJI\n0X20000..0x2A6DF KANJI\n"
        ))
        .unwrap();
        assert_eq!(def.categories.len(), 3);
        assert_eq!(def.categories[1], Category { name: "KANJI".into(), invoke: false, group: false, length: 2 });
        assert_eq!(primary(&def, 0x4E8C), "KANJI");
        // The later mapping wins, and keeps both categories in its set.
        assert_eq!(primary(&def, 0x4E00), "NUM");
        assert_eq!(def.assignment(def.assignment_of(0x4E00)).mask, 0b110);
        assert_eq!(primary(&def, 0x35), "NUM");
        assert_eq!(primary(&def, 0x41), "DEFAULT");
        assert_eq!(primary(&def, 0x20BB7), "KANJI");
        assert_eq!(primary(&def, 0x2A6E0), "DEFAULT");
        assert_eq!(primary(&def, 0x10FFFF), "DEFAULT");
    }

    #[test]
    fn later_supplementary_mappings_win_where_they_overlap() {
        let def = parse(&format!(
            "{BASE}0x20000..0x20010 KANJI\n0x20005..0x20007 NUM\n0xFFF0..0x20002 NUM\n0x30000 KANJI\n0x30001 KANJI\n"
        ))
        .unwrap();
        let p = |cp| primary(&def, cp).to_string();
        assert_eq!(p(0xFFF5), "NUM");
        assert_eq!(p(0x20000), "NUM");
        assert_eq!(p(0x20002), "NUM");
        assert_eq!(p(0x20003), "KANJI");
        assert_eq!(p(0x20006), "NUM");
        assert_eq!(p(0x20008), "KANJI");
        assert_eq!(p(0x20010), "KANJI");
        assert_eq!(p(0x20011), "DEFAULT");
        assert_eq!(p(0x30000), "KANJI");
        assert_eq!(p(0x30001), "KANJI");
        assert_eq!(p(0x30002), "DEFAULT");
        // Adjacent single code points with the same assignment are one range.
        assert_eq!(def.ranges.iter().filter(|r| r.0 >= 0x30000).count(), 1);
    }

    #[test]
    fn code_points_are_read_as_the_java_reads_them() {
        let def = parse(&format!("{BASE}0x+41 KANJI\n0x0042..ab0043 NUM\n0x\u{ff14}\u{ff14} KANJI\n")).unwrap();
        // "0x+41": substring(2) is "+41", a hexadecimal int.
        assert_eq!(primary(&def, 0x41), "KANJI");
        // The end of a range skips two code units whatever they are.
        assert_eq!(primary(&def, 0x43), "NUM");
        // Fullwidth digits are hexadecimal digits too.
        assert_eq!(primary(&def, 0x44), "KANJI");
    }

    #[test]
    fn hostile_lines() {
        let m = |s: &str| message(&format!("{BASE}{s}\n"));
        assert_eq!(m("0x KANJI"), "malformed code point at char.def line 4");
        assert_eq!(m("0xZZ KANJI"), "malformed code point at char.def line 4");
        assert_eq!(m("0x41.. KANJI"), "malformed code point at char.def line 4");
        assert_eq!(m("0x41..0 KANJI"), "malformed code point at char.def line 4");
        assert_eq!(m("0x110000 KANJI"), "code point out of range at char.def line 4");
        assert_eq!(m("0x-1 KANJI"), "code point out of range at char.def line 4");
        assert_eq!(m("0x80000000 KANJI"), "malformed code point at char.def line 4");
        assert_eq!(m("0x4E00..0x9FFF"), "mapping without category at char.def line 4");
        assert_eq!(m("0x42..0x41 KANJI"), "code point range descends at char.def line 4");
        // The code points are read before the field count is checked.
        assert_eq!(m("0xQ"), "malformed code point at char.def line 4");
        assert_eq!(m("OTHER 0 1"), "malformed category at char.def line 4");
        assert_eq!(m("OTHER 2 1 0"), "malformed category flag at char.def line 4");
        assert_eq!(m("OTHER 0 01 0"), "malformed category flag at char.def line 4");
        assert_eq!(m("OTHER 0 1 x"), "malformed number in char.def line 4");
        assert_eq!(m("OTHER 0 1 16"), "category LENGTH must be between 0 and 15 at char.def line 4");
        assert_eq!(m("OTHER 0 1 -1"), "category LENGTH must be between 0 and 15 at char.def line 4");
        assert_eq!(m("KANJI 0 1 0"), "duplicate char.def category KANJI at char.def line 4");
        assert_eq!(m("0x41 MISSING"), "char.def declaration at U+0041 names the undefined category MISSING");
        assert_eq!(
            m("0x20000..0x20001 KANJI MISSING"),
            "char.def declaration at U+20000 names the undefined category MISSING"
        );
        assert_eq!(message("KANJI 0 0 2\n0x4E00 KANJI\n"), "char.def defines no DEFAULT category");
        assert_eq!(message(""), "char.def defines no DEFAULT category");
        // Eighteen categories.
        let many: String = (0..18).map(|i| format!("C{i} 0 0 0\n")).collect();
        assert_eq!(
            message(&format!("DEFAULT 0 1 0\n{many}")),
            "char.def defines 18 categories; MeCab supports at most 17 at char.def line 18"
        );
    }

    #[test]
    fn a_legacy_separator_alone_on_a_line_is_refused_not_a_panic() {
        let e = CharDefSource::parse("DEFAULT 0 1 0\n\u{1c}\n", WhitespaceMode::Legacy).err().unwrap();
        assert_eq!(e.to_string(), "malformed category at char.def line 2");
    }

    #[test]
    fn the_whole_range_of_code_points_maps_without_overallocating() {
        let def = parse(&format!("{BASE}0x0..0x10FFFF KANJI\n0x0..0x10FFFF NUM\n")).unwrap();
        assert_eq!(primary(&def, 0), "NUM");
        assert_eq!(primary(&def, 0x10FFFF), "NUM");
        assert_eq!(def.ranges.len(), 1);
    }

    #[test]
    fn comments_and_flags() {
        let def = parse("# only a comment\nDEFAULT 1 0 3 extra fields#x\n   \n").unwrap();
        assert_eq!(def.categories[0], Category { name: "DEFAULT".into(), invoke: true, group: false, length: 3 });
    }
}
