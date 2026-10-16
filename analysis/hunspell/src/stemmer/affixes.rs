//! Reading a word as an entry with affixes: OpenNLP's `analyze` and the
//! `undo` methods.
//!
//! Undoing a rule cuts its affix text off the word, puts back the text it
//! stripped and checks its condition on the result. The paths are: one
//! suffix, and through a continuation class a second; one prefix, and with
//! `COMPLEXPREFIXES` a second; and a cross product of a prefix with a
//! suffix and a further suffix. A reading counts only when one homonym's
//! flags allow every rule of it.

use std::borrow::Cow;

use crate::affix::Affix;

use super::{Analysis, Stemmer};

/// The first character of a word that is not empty.
#[inline]
pub(crate) fn first_char(word: &str) -> char {
    word.chars().next().expect("the word is not empty")
}

impl Stemmer<'_> {
    /// Adds every reading of one case variant: the word's own entries, then
    /// the suffix paths, then the prefix paths. The order is the order of
    /// the stems.
    pub(crate) fn analyze_variant(&self, word: &str, a: &mut Analysis<'_>) {
        let d = self.dictionary;
        let entries = a.lookup(d, word);
        if !entries.is_empty() {
            if d.first_forbidden(entries) {
                a.results.forbidden = true;
                return;
            }
            for e in entries.iter() {
                if d.valid_standalone(d.flags(e)) {
                    a.add(d, word, e, &[]);
                }
            }
        }
        for suffix in d.suffixes.candidates(word) {
            self.undo_suffix(word, suffix, a);
        }
        for prefix in d.prefixes.candidates(word) {
            self.undo_prefix(word, prefix, a);
        }
    }

    /// One suffix, and through continuation classes a second one under it.
    /// A compound-only or circumfix rule is not undone, since no prefix
    /// accompanies it here; a rule needing a further affix gives no reading
    /// alone.
    fn undo_suffix(&self, word: &str, suffix: &Affix, a: &mut Analysis<'_>) {
        let d = self.dictionary;
        if d.compound_only(suffix) || d.circumfix_only(suffix) {
            return;
        }
        let Some(stem) = self.remove_suffix_allowing_identity(word, suffix) else { return };
        if !d.needs_further_affix(suffix) {
            let entries = a.lookup(d, &stem);
            for e in entries.iter() {
                a.note_forbidden(d, e, suffix.flag);
            }
            for e in entries.iter() {
                if d.supports(d.flags(e), suffix.flag) {
                    a.add(d, &stem, e, &[suffix]);
                }
            }
        }
        if d.settings.complex_prefixes {
            return;
        }
        for inner in d.suffixes.candidates(&stem) {
            self.undo_inner_suffix(&stem, suffix, inner, a);
        }
    }

    /// The inner suffix of two, when its continuation classes allow the
    /// outer one.
    fn undo_inner_suffix(&self, stem: &str, outer: &Affix, inner: &Affix, a: &mut Analysis<'_>) {
        let d = self.dictionary;
        if !inner.allows(outer.flag) || d.compound_only(inner) || d.circumfix_only(inner) {
            return;
        }
        let Some(root) = self.remove_suffix_allowing_identity(stem, inner) else { return };
        for e in a.lookup(d, &root).iter() {
            if d.supports(d.flags(e), inner.flag) {
                a.add(d, &root, e, &[inner, outer]);
            }
        }
    }

    /// One prefix, with `COMPLEXPREFIXES` a second one under it, and for a
    /// cross product rule a suffix too.
    fn undo_prefix(&self, word: &str, prefix: &Affix, a: &mut Analysis<'_>) {
        let d = self.dictionary;
        if d.compound_only(prefix) {
            return;
        }
        let Some(stem) = self.remove_prefix_allowing_identity(word, prefix) else { return };
        if !d.needs_further_affix(prefix) && !d.circumfix_only(prefix) {
            let entries = a.lookup(d, &stem);
            for e in entries.iter() {
                a.note_forbidden(d, e, prefix.flag);
            }
            for e in entries.iter() {
                if d.supports(d.flags(e), prefix.flag) {
                    a.add(d, &stem, e, &[prefix]);
                }
            }
        }
        if d.settings.complex_prefixes {
            for inner in d.prefixes.candidates(&stem) {
                self.undo_inner_prefix(&stem, prefix, inner, a);
            }
        }
        if !prefix.cross_product {
            return;
        }
        for suffix in d.suffixes.candidates(&stem) {
            self.undo_cross_product_suffix(&stem, prefix, suffix, a);
        }
    }

    /// The suffix of a cross product. Both rules must be cross product
    /// rules and agree on circumfixing.
    fn undo_cross_product_suffix(&self, stem: &str, prefix: &Affix, suffix: &Affix, a: &mut Analysis<'_>) {
        let d = self.dictionary;
        if !suffix.cross_product || d.compound_only(suffix) || d.circumfix_only(prefix) != d.circumfix_only(suffix) {
            return;
        }
        let Some(double_stem) = self.remove_suffix(stem, suffix) else { return };
        // either member may satisfy the other's NEEDAFFIX marker; both rule
        // flags must be in one homonym's flags
        if !(d.needs_further_affix(prefix) && d.needs_further_affix(suffix)) {
            for e in a.lookup(d, &double_stem).iter() {
                let flags = d.flags(e);
                if d.licenses_cross_product(flags, prefix, suffix) {
                    a.note_forbidden(d, e, suffix.flag);
                }
                if d.supports_cross_product(flags, prefix, suffix) {
                    a.add(d, &double_stem, e, &[prefix, suffix]);
                }
            }
        }
        if d.settings.complex_prefixes {
            return;
        }
        for inner in d.suffixes.candidates(&double_stem) {
            self.undo_cross_product_inner_suffix(&double_stem, prefix, suffix, inner, a);
        }
    }

    /// The inner suffix after a prefix and an outer suffix.
    fn undo_cross_product_inner_suffix(
        &self,
        stem: &str,
        prefix: &Affix,
        outer: &Affix,
        inner: &Affix,
        a: &mut Analysis<'_>,
    ) {
        let d = self.dictionary;
        if !inner.cross_product
            || !inner.allows(outer.flag)
            || d.compound_only(inner)
            || d.circumfix_only(outer)
            || d.circumfix_only(prefix) != d.circumfix_only(inner)
        {
            return;
        }
        let Some(root) = self.remove_suffix_allowing_identity(stem, inner) else { return };
        for e in a.lookup(d, &root).iter() {
            if d.supports_cross_product(d.flags(e), prefix, inner) {
                a.add(d, &root, e, &[prefix, inner, outer]);
            }
        }
    }

    /// The inner prefix of two under `COMPLEXPREFIXES`, and a suffix after
    /// both.
    fn undo_inner_prefix(&self, stem: &str, outer: &Affix, inner: &Affix, a: &mut Analysis<'_>) {
        let d = self.dictionary;
        if !inner.allows(outer.flag) || d.compound_only(inner) || d.circumfix_only(outer) {
            return;
        }
        let Some(root) = self.remove_prefix_allowing_identity(stem, inner) else { return };
        if !d.circumfix_only(inner) {
            for e in a.lookup(d, &root).iter() {
                if d.supports(d.flags(e), inner.flag) {
                    a.add(d, &root, e, &[inner, outer]);
                }
            }
        }
        if inner.cross_product && outer.cross_product {
            for suffix in d.suffixes.candidates(&root) {
                self.undo_double_prefix_suffix(&root, inner, outer, suffix, a);
            }
        }
    }

    /// The suffix after two prefixes.
    fn undo_double_prefix_suffix(
        &self,
        word: &str,
        inner: &Affix,
        outer: &Affix,
        suffix: &Affix,
        a: &mut Analysis<'_>,
    ) {
        let d = self.dictionary;
        if !suffix.cross_product || d.compound_only(suffix) || d.circumfix_only(inner) != d.circumfix_only(suffix) {
            return;
        }
        let Some(root) = self.remove_suffix_allowing_identity(word, suffix) else { return };
        for e in a.lookup(d, &root).iter() {
            if d.supports_cross_product(d.flags(e), inner, suffix) {
                a.add(d, &root, e, &[inner, outer, suffix]);
            }
        }
    }

    /// Undoes a suffix rule: the word must end in its affix text, the stem
    /// (the rest with the strip text put back) must not be empty and must
    /// meet the condition. A word that is all affix text is undone only
    /// under `FULLSTRIP`. A rule that adds and strips nothing is not undone
    /// here.
    pub(crate) fn remove_suffix<'w>(&self, word: &'w str, suffix: &Affix) -> Option<Cow<'w, str>> {
        let (add, strip) = (&*suffix.add, &*suffix.strip);
        if (add.is_empty() && strip.is_empty())
            || !word.ends_with(add)
            || (word.len() == add.len() && (strip.is_empty() || !self.dictionary.settings.full_strip))
        {
            return None;
        }
        let rest = &word[..word.len() - add.len()];
        let stem: Cow<'w, str> = if strip.is_empty() { rest.into() } else { format!("{rest}{strip}").into() };
        suffix.condition.matches(&stem).then_some(stem)
    }

    /// Undoes a prefix rule, as [`Stemmer::remove_suffix`] a suffix rule.
    pub(crate) fn remove_prefix<'w>(&self, word: &'w str, prefix: &Affix) -> Option<Cow<'w, str>> {
        let (add, strip) = (&*prefix.add, &*prefix.strip);
        if (add.is_empty() && strip.is_empty())
            || !word.starts_with(add)
            || (word.len() == add.len() && (strip.is_empty() || !self.dictionary.settings.full_strip))
        {
            return None;
        }
        let rest = &word[add.len()..];
        let stem: Cow<'w, str> = if strip.is_empty() { rest.into() } else { format!("{strip}{rest}").into() };
        prefix.condition.matches(&stem).then_some(stem)
    }

    /// [`Stemmer::remove_suffix`], and a rule that adds and strips nothing
    /// gives the word itself when it meets the condition.
    pub(crate) fn remove_suffix_allowing_identity<'w>(&self, word: &'w str, suffix: &Affix) -> Option<Cow<'w, str>> {
        if suffix.is_identity() {
            return suffix.condition.matches(word).then_some(word.into());
        }
        self.remove_suffix(word, suffix)
    }

    /// [`Stemmer::remove_prefix`], and a rule that adds and strips nothing
    /// gives the word itself when it meets the condition.
    pub(crate) fn remove_prefix_allowing_identity<'w>(&self, word: &'w str, prefix: &Affix) -> Option<Cow<'w, str>> {
        if prefix.is_identity() {
            return prefix.condition.matches(word).then_some(word.into());
        }
        self.remove_prefix(word, prefix)
    }

    /// Undoes a rule of either kind on a compound part.
    pub(crate) fn remove_affix_in_compound<'w>(&self, part: &'w str, affix: &Affix) -> Option<Cow<'w, str>> {
        if affix.is_suffix() {
            self.remove_suffix_allowing_identity(part, affix)
        } else {
            self.remove_prefix_allowing_identity(part, affix)
        }
    }
}
