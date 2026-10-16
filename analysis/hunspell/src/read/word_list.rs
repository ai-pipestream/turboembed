//! Reading the decoded word list (`.dic`): OpenNLP's `parseWordList`.
//!
//! A line is `word[/flags][ fields...]`, after an optional first line
//! holding the number of entries. A word may hold spaces, so the fields are
//! found first: from a tab, or from a space or tab followed by a two
//! character tag and `:` (`po:noun`), or with an `AM` table a trailing
//! alias number. The flags follow the first `/` not written `\/`.

use crate::affix::Fields;
use crate::error::LoadError;
use crate::flag::Flag;
use crate::hash::FoldMap;
use crate::read::affix_file::{AffixFile, parse_aliased_flags, parse_morphology};
use crate::text::{is_count, remove_ignored, split_fields, split_lines, trim};

/// One line of the word list.
pub(crate) struct RawEntry {
    pub(crate) flags: Vec<Flag>,
    pub(crate) morphology: Option<Fields>,
}

/// The entries of each word, the words in the order of their first line.
pub(crate) struct WordList {
    pub(crate) words: Vec<(Box<str>, Vec<RawEntry>)>,
}

/// The two separators of the word list's fields.
fn is_separator(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

/// Reads the word list, its byte order mark already removed.
pub(crate) fn parse(content: &str, affix: &AffixFile) -> Result<WordList, LoadError> {
    let lines = split_lines(content);
    let start = usize::from(lines.first().is_some_and(|l| is_count(trim(l))));
    // The word, its first line's position and its entries.
    let mut words: FoldMap<Box<str>, (usize, Vec<RawEntry>)> = FoldMap::default();
    for (i, raw) in lines.iter().enumerate().skip(start) {
        let line_number = i + 1;
        let line = trim(raw);
        if line.is_empty() {
            continue;
        }
        let mut morphology = morphology_index(line);
        if morphology.is_none()
            && !affix.morphology_aliases.is_empty()
            && let Some(space) = line.rfind(' ')
            && space > 0
            && is_count(&line[space + 1..])
        {
            morphology = Some(space);
        }
        let entry = match morphology {
            Some(cut) => trim(&line[..cut]),
            None => line,
        };
        let mut word = entry;
        let mut flags = Vec::new();
        if let Some(slash) = unescaped_slash(entry) {
            word = &entry[..slash];
            let run = &entry[slash + 1..];
            let end = run.bytes().position(is_separator).unwrap_or(run.len());
            flags = parse_aliased_flags(&run[..end], affix.flag_mode, &affix.flag_aliases, line_number)?;
        }
        let fields = match morphology {
            Some(cut) => parse_morphology(&split_fields(&line[cut..]), &affix.morphology_aliases, line_number)?,
            None => None,
        };
        let unescaped: Box<str> = if word.contains("\\/") { word.replace("\\/", "/").into() } else { word.into() };
        let key: Box<str> = match remove_ignored(&unescaped, &affix.ignored) {
            std::borrow::Cow::Borrowed(_) => unescaped,
            std::borrow::Cow::Owned(s) => s.into(),
        };
        let next = words.len();
        words
            .entry(key)
            .or_insert_with(|| (next, Vec::with_capacity(1)))
            .1
            .push(RawEntry { flags, morphology: fields });
    }
    let mut ordered: Vec<(usize, Box<str>, Vec<RawEntry>)> =
        words.into_iter().map(|(word, (first, entries))| (first, word, entries)).collect();
    ordered.sort_unstable_by_key(|w| w.0);
    Ok(WordList { words: ordered.into_iter().map(|(_, word, entries)| (word, entries)).collect() })
}

/// The first `/` after the first character that does not follow a
/// backslash.
fn unescaped_slash(entry: &str) -> Option<usize> {
    let b = entry.as_bytes();
    (1..b.len()).find(|&i| b[i] == b'/' && b[i - 1] != b'\\')
}

/// Where the morphological fields of a trimmed line start: at the first
/// space or tab of the separators before a two character tag and `:` (a
/// tag at the start of the line is no field), or at an earlier tab.
///
/// OpenNLP counts the tag's two characters in UTF-16 units, so a character
/// outside the Basic Multilingual Plane is a tag of its own.
fn morphology_index(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut cut = None;
    let mut units = 0;
    for (at, c) in line.char_indices() {
        if c == ':'
            && units >= 4
            && let Some(separator) = third_unit_before(line, at)
        {
            let mut field_start = separator;
            while field_start > 0 && is_separator(bytes[field_start - 1]) {
                field_start -= 1;
            }
            // a tag with no word in front of it is not a field
            cut = (field_start > 0).then_some(field_start);
            break;
        }
        units += c.len_utf16();
    }
    match (line.find('\t'), cut) {
        (Some(tab), Some(c)) if tab < c => Some(tab),
        (Some(tab), None) => Some(tab),
        (_, cut) => cut,
    }
}

/// The byte offset of the separator that is the third UTF-16 unit before
/// byte `at`, if that unit is a space or tab.
fn third_unit_before(line: &str, at: usize) -> Option<usize> {
    let mut units = 0;
    for (pos, c) in line[..at].char_indices().rev() {
        units += c.len_utf16();
        if units >= 3 {
            return (units == 3 && (c == ' ' || c == '\t')).then_some(pos);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::affix_file;

    #[test]
    fn fields_start_at_a_tag_or_a_tab() {
        assert_eq!(morphology_index("walk/A po:verb"), Some(6));
        assert_eq!(morphology_index("po:verb"), None);
        assert_eq!(morphology_index("ice cream  po:noun"), Some(9));
        assert_eq!(morphology_index("a\tb"), Some(1));
        assert_eq!(morphology_index("ab \u{1F600}:x"), Some(2));
        assert_eq!(morphology_index("a\u{1F600}:x"), None);
    }

    #[test]
    fn entries_parse() {
        let affix = affix_file::parse("IGNORE -\n").unwrap();
        let list = parse("3\nwalk/AB po:verb\na\\/b\nwa-lk/C\n", &affix).unwrap();
        let words: Vec<&str> = list.words.iter().map(|w| &*w.0).collect();
        assert_eq!(words, ["walk", "a/b"]);
        assert_eq!(list.words[0].1.len(), 2);
        assert_eq!(list.words[0].1[0].flags, [Flag::of('A'), Flag::of('B')]);
        assert!(list.words[0].1[0].morphology.is_some());
        assert_eq!(list.words[0].1[1].flags, [Flag::of('C')]);
        let bad = parse("x/AB\n", &affix_file::parse("FLAG long\n").unwrap());
        assert!(bad.is_ok());
        let odd = parse("x/ABC\n", &affix_file::parse("FLAG long\n").unwrap());
        assert_eq!(odd.err().unwrap().to_string(), "odd long-flag run at line 1");
    }
}
