//! The stem and the morphological analysis of a reading, from the fields
//! of its entry and affixes: OpenNLP's `morphologicalStems` and
//! `morphologicalAnalyses`.
//!
//! A reading names an entry by its spelling and flags, so every homonym
//! with the same spelling and equal flags contributes, each with its own
//! fields.

use crate::affix::Affix;
use crate::dictionary::{Dictionary, EntryId};
use crate::text::utf16_len;

/// The field that gives an entry's stem.
const STEM_FIELD: &str = "st:";
/// The field of a surface prefix, which the stem starts with.
const SURFACE_PREFIX_FIELD: &str = "sp:";
/// The field of a derivational suffix, which makes the derived form the
/// stem.
const DERIVATIONAL_SUFFIX_FIELD: &str = "ds:";
/// The field naming the flag of an affix rule without fields.
const FLAG_FIELD: &str = "fl:";

/// The value of the first field with `tag`.
fn field_value<'a>(fields: &'a [Box<str>], tag: &str) -> Option<&'a str> {
    fields.iter().find_map(|f| f.strip_prefix(tag))
}

/// The fields an affix contributes: its own, or `fl:` and its flag.
fn push_affix_fields(fields: &mut Vec<String>, affix: &Affix) {
    if affix.fields().is_empty() {
        fields.push(format!("{FLAG_FIELD}{}", affix.flag_text));
    } else {
        fields.extend(affix.fields().iter().map(|f| f.to_string()));
    }
}

/// `text` without the last `units` UTF-16 units, as Java's `substring`
/// cuts it (a cut inside a character drops the whole character).
fn cut_utf16_from_end(text: &str, units: usize) -> &str {
    let mut left = units;
    let mut end = text.len();
    for (at, c) in text.char_indices().rev() {
        if left == 0 {
            break;
        }
        left = left.saturating_sub(c.len_utf16());
        end = at;
    }
    &text[..end]
}

impl Dictionary {
    /// The stems of the homonyms of `root` with the flags of `selected`,
    /// after the affixes in the order they were applied.
    pub(crate) fn morphological_stems(&self, root: &str, selected: EntryId, affixes: &[&Affix]) -> Vec<String> {
        let mut out = Vec::new();
        for &e in self.homonyms(root) {
            if self.same_flags(e, selected) {
                let fields = self.entry(e).morphology.as_deref().unwrap_or(&[]);
                out.push(morphological_stem(root, fields, affixes));
            }
        }
        out
    }

    /// The analyses of the homonyms of `root` with the flags of `selected`:
    /// the prefixes' fields (outermost first), `st:` unless the entry has
    /// one, the entry's fields, and the suffixes' fields. A prefix without
    /// fields contributes its text when no suffix follows and its `fl:`
    /// field otherwise; an entry without fields then also gets that prefix's
    /// `fl:` field. The unaffixed last part of a compound with no entry
    /// fields gives an empty analysis.
    pub(crate) fn morphological_analyses(
        &self,
        root: &str,
        selected: EntryId,
        compound_end: bool,
        affixes: &[&Affix],
    ) -> Vec<String> {
        let mut out = Vec::new();
        let has_suffix = affixes.iter().any(|a| a.is_suffix());
        // As in Java, the prefix found for one homonym is kept for the next.
        let mut prefix_only: Option<&Affix> = None;
        for &e in self.homonyms(root) {
            if !self.same_flags(e, selected) {
                continue;
            }
            let mut fields: Vec<String> = Vec::new();
            for prefix in affixes.iter().rev().filter(|a| !a.is_suffix()) {
                if has_suffix || !prefix.fields().is_empty() {
                    push_affix_fields(&mut fields, prefix);
                } else {
                    fields.push(prefix.affix().to_string());
                }
                if !has_suffix {
                    prefix_only = Some(prefix);
                }
            }
            let entry = self.entry(e).morphology.as_deref().unwrap_or(&[]);
            if compound_end && affixes.is_empty() && entry.is_empty() {
                out.push(String::new());
                continue;
            }
            if field_value(entry, STEM_FIELD).is_none() {
                fields.push(format!("{STEM_FIELD}{root}"));
            }
            fields.extend(entry.iter().map(|f| f.to_string()));
            if entry.is_empty()
                && let Some(prefix) = prefix_only
            {
                fields.push(format!("{FLAG_FIELD}{}", prefix.flag_text));
            }
            for suffix in affixes.iter().filter(|a| a.is_suffix()) {
                push_affix_fields(&mut fields, suffix);
            }
            out.push(fields.join(" "));
        }
        out
    }
}

/// One homonym's stem: its `st:` field (or the root, or with a `ds:` field
/// the root itself), replaced by the derived form at a suffix with `ds:`,
/// after the `sp:` fields of the entry and the affixes.
fn morphological_stem(root: &str, fields: &[Box<str>], affixes: &[&Affix]) -> String {
    let mut result: std::borrow::Cow<'_, str> = field_value(fields, STEM_FIELD).unwrap_or(root).into();
    if field_value(fields, DERIVATIONAL_SUFFIX_FIELD).is_some() {
        result = root.into();
    }
    // The derived form is built from the entry and its suffixes alone; a
    // prefix shows in the stem only through an sp: field.
    let mut derived = String::from(root);
    let mut surface_prefix = String::from(field_value(fields, SURFACE_PREFIX_FIELD).unwrap_or(""));
    for affix in affixes {
        if affix.is_suffix() {
            let keep = cut_utf16_from_end(&derived, utf16_len(affix.strip())).len();
            derived.truncate(keep);
            derived.push_str(affix.affix());
            if field_value(affix.fields(), DERIVATIONAL_SUFFIX_FIELD).is_some() {
                result = derived.clone().into();
            }
        }
        surface_prefix.push_str(field_value(affix.fields(), SURFACE_PREFIX_FIELD).unwrap_or(""));
    }
    surface_prefix.push_str(&result);
    surface_prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_count_utf16_units() {
        assert_eq!(cut_utf16_from_end("walky", 1), "walk");
        assert_eq!(cut_utf16_from_end("a\u{1F600}", 2), "a");
        assert_eq!(cut_utf16_from_end("ab", 0), "ab");
        assert_eq!(cut_utf16_from_end("ab", 5), "");
    }
}
