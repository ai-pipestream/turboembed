//! The readings of one compound part: OpenNLP's `partReadings`,
//! `ruleParts` and the `collect` methods.
//!
//! A part is read as an entry the compound flags admit at its place, or as
//! an entry with affixes. An affix may face another part only with
//! `COMPOUNDPERMITFLAG`: a suffix belongs on the last part, a prefix on the
//! first. With `COMPOUNDMORESUFFIXES` a second affix that a continuation
//! class allows is undone too.

use crate::affix::Affix;
use crate::dictionary::EntryId;
use crate::dictionary::rules::CompoundPosition;
use crate::text::upper_char;

use super::Stemmer;
use super::affixes::first_char;
use super::results::CompoundPart;

/// What every reading of one part shares.
struct PartContext<'a> {
    /// The part as read.
    part: &'a str,
    /// The part in the case source.
    surface: &'a str,
    position: CompoundPosition,
    first: bool,
    last: bool,
}

impl<'d> Stemmer<'d> {
    /// The readings of a positional compound part. A spelling whose first
    /// homonym carries `COMPOUNDFORBIDFLAG` has none but as the last part;
    /// a part with no reading is read again with its first letter upper
    /// case, for a capitalized entry.
    pub(crate) fn part_readings(
        &self,
        part: &str,
        surface: &str,
        position: CompoundPosition,
        first: bool,
        last: bool,
        moving_rule: bool,
    ) -> Vec<CompoundPart<'d>> {
        let d = self.dictionary;
        let mut readings = Vec::new();
        // every reading is an entry with at most three affixes
        if d.too_long_to_read(part) {
            return readings;
        }
        if !last && !moving_rule && d.forbids_compound_start(d.lookup(part)) {
            return readings;
        }
        let context = PartContext { part, surface, position, first, last };
        self.collect_part_readings(&context, moving_rule, &mut readings);
        if readings.is_empty() && !part.is_empty() {
            let initial = first_char(part);
            let upper = upper_char(initial);
            if upper != initial {
                let capitalized = format!("{upper}{}", &part[initial.len_utf8()..]);
                let context = PartContext { part: &capitalized, ..context };
                self.collect_part_readings(&context, moving_rule, &mut readings);
            }
        }
        readings
    }

    /// The part's entries the position admits (under the Hungarian rule
    /// before a hyphen also an opening entry with `F`, `G` or `H`), then
    /// its affixed readings.
    fn collect_part_readings(&self, c: &PartContext<'_>, moving_rule: bool, readings: &mut Vec<CompoundPart<'d>>) {
        let d = self.dictionary;
        for e in d.lookup(c.part).iter() {
            let flags = d.flags(e);
            if (d.may_stand(flags, c.position) || (moving_rule && !c.last && d.opens_hyphenated_compound(flags)))
                && d.accepts_case(flags, c.surface, c.part, &[])
            {
                readings.push(part(c.part, c.part, e, &[]));
            }
        }
        for suffix in d.suffixes.candidates(c.part) {
            self.collect_affixed(c, suffix, readings);
        }
        for prefix in d.prefixes.candidates(c.part) {
            self.collect_affixed(c, prefix, readings);
            self.collect_cross_readings(c, prefix, readings);
        }
    }

    /// One affix undone on the part, and with `COMPOUNDMORESUFFIXES` a
    /// second of the same kind at the compound's edge.
    fn collect_affixed(&self, c: &PartContext<'_>, affix: &'d Affix, readings: &mut Vec<CompoundPart<'d>>) {
        let d = self.dictionary;
        let suffix = affix.is_suffix();
        let at_edge = if suffix { c.last } else { c.first };
        // a compound-only suffix joins parts and never closes a compound
        if d.circumfix_only(affix)
            || d.forbids_in_compound(affix)
            || (!at_edge && !d.permits_inside(affix))
            || (suffix && c.last && d.compound_only(affix))
        {
            return;
        }
        let Some(stem) = self.remove_affix_in_compound(c.part, affix) else { return };
        if !d.needs_further_affix(affix) {
            let admits = d.affix_admits(affix, c.position);
            for e in d.lookup(&stem).iter() {
                let flags = d.flags(e);
                if d.supports_part(flags, affix.flag, c.position, admits)
                    && d.accepts_case(flags, c.surface, c.part, &[affix])
                {
                    readings.push(part(c.part, &stem, e, &[affix]));
                }
            }
        }
        if d.settings.compound_more_suffixes && suffix != d.settings.complex_prefixes && at_edge {
            let inner_rules = if suffix {
                d.suffixes.candidates(&stem).collect::<Vec<_>>()
            } else {
                d.prefixes.candidates(&stem).collect::<Vec<_>>()
            };
            for inner in inner_rules {
                self.collect_double(c, &stem, inner, affix, readings);
            }
        }
    }

    /// An inner affix whose continuation classes allow the outer one, and
    /// after two prefixes a suffix.
    fn collect_double(
        &self,
        c: &PartContext<'_>,
        intermediate: &str,
        inner: &'d Affix,
        outer: &'d Affix,
        readings: &mut Vec<CompoundPart<'d>>,
    ) {
        let d = self.dictionary;
        if !inner.allows(outer.flag) || d.forbids_in_compound(inner) {
            return;
        }
        let Some(root) = self.remove_affix_in_compound(intermediate, inner) else { return };
        if !d.circumfix_only(inner) {
            let admits = d.affix_admits(inner, c.position) || d.affix_admits(outer, c.position);
            for e in d.lookup(&root).iter() {
                let flags = d.flags(e);
                if d.supports_part(flags, inner.flag, c.position, admits)
                    && d.accepts_case(flags, c.surface, c.part, &[inner, outer])
                {
                    readings.push(part(c.part, &root, e, &[inner, outer]));
                }
            }
        }
        if !inner.is_suffix() && inner.cross_product && outer.cross_product {
            for suffix in d.suffixes.candidates(&root) {
                if !c.last && !d.permits_inside(suffix) {
                    continue;
                }
                if let Some(base) = self.remove_suffix_allowing_identity(&root, suffix) {
                    self.add_cross_double(c, &base, inner, outer, suffix, readings);
                }
            }
        }
    }

    /// A prefix and a suffix undone on the part.
    fn collect_cross_readings(&self, c: &PartContext<'_>, prefix: &'d Affix, readings: &mut Vec<CompoundPart<'d>>) {
        let d = self.dictionary;
        if !prefix.cross_product || d.forbids_in_compound(prefix) || (!c.first && !d.permits_inside(prefix)) {
            return;
        }
        let Some(intermediate) = self.remove_prefix_allowing_identity(c.part, prefix) else { return };
        for suffix in d.suffixes.candidates(&intermediate) {
            self.collect_cross(c, &intermediate, prefix, suffix, readings);
        }
    }

    /// The suffix of a prefix and suffix pair, and with
    /// `COMPOUNDMORESUFFIXES` an inner suffix on the last part.
    fn collect_cross(
        &self,
        c: &PartContext<'_>,
        intermediate: &str,
        prefix: &'d Affix,
        suffix: &'d Affix,
        readings: &mut Vec<CompoundPart<'d>>,
    ) {
        let d = self.dictionary;
        if !suffix.cross_product
            || d.forbids_in_compound(suffix)
            || (!c.last && !d.permits_inside(suffix))
            || d.circumfix_only(prefix) != d.circumfix_only(suffix)
            || (d.needs_further_affix(prefix) && d.needs_further_affix(suffix))
        {
            return;
        }
        let Some(root) = self.remove_suffix_allowing_identity(intermediate, suffix) else { return };
        for e in d.lookup(&root).iter() {
            let flags = d.flags(e);
            if d.supports_compound_cross_product(flags, c.position, prefix, suffix, None)
                && d.accepts_case(flags, c.surface, c.part, &[prefix, suffix])
            {
                readings.push(part(c.part, &root, e, &[prefix, suffix]));
            }
        }
        if c.last && d.settings.compound_more_suffixes && !d.settings.complex_prefixes {
            for inner in d.suffixes.candidates(&root) {
                if let Some(base) = self.remove_suffix_allowing_identity(&root, inner) {
                    self.add_cross_double(c, &base, inner, suffix, prefix, readings);
                }
            }
        }
    }

    /// Two affixes of one kind linked by a continuation class and one of
    /// the other kind (`cross`), all cross product rules.
    fn add_cross_double(
        &self,
        c: &PartContext<'_>,
        root: &str,
        inner: &'d Affix,
        outer: &'d Affix,
        cross: &'d Affix,
        readings: &mut Vec<CompoundPart<'d>>,
    ) {
        let d = self.dictionary;
        if !inner.allows(outer.flag)
            || !inner.cross_product
            || !cross.cross_product
            || d.forbids_in_compound(inner)
            || d.forbids_in_compound(cross)
            || d.circumfix_only(outer)
            || d.circumfix_only(inner) != d.circumfix_only(cross)
        {
            return;
        }
        let (prefix, suffix, applied) = if inner.is_suffix() {
            (cross, inner, [cross, inner, outer])
        } else {
            (inner, cross, [inner, outer, cross])
        };
        for e in d.lookup(root).iter() {
            let flags = d.flags(e);
            if d.supports_compound_cross_product(flags, c.position, prefix, suffix, Some(outer))
                && d.accepts_case(flags, c.surface, c.part, &applied)
            {
                readings.push(part(c.part, root, e, &applied));
            }
        }
    }

    /// The readings of a `COMPOUNDRULE` part: its entries, and on the last
    /// part its entries with one suffix undone.
    pub(crate) fn rule_parts(&self, text: &str, surface: &str, last: bool) -> Vec<CompoundPart<'d>> {
        let mut result = Vec::new();
        if self.dictionary.too_long_to_read(text) {
            return result;
        }
        self.add_rule_part(text, text, surface, last, &[], &mut result);
        if last {
            for suffix in self.dictionary.suffixes.candidates(text) {
                if let Some(root) = self.remove_suffix_allowing_identity(text, suffix) {
                    self.add_rule_part(text, &root, surface, true, &[suffix], &mut result);
                }
            }
        }
        result
    }

    fn add_rule_part(
        &self,
        text: &str,
        root: &str,
        surface: &str,
        last: bool,
        affixes: &[&'d Affix],
        result: &mut Vec<CompoundPart<'d>>,
    ) {
        let d = self.dictionary;
        let entries = d.lookup(root);
        if entries.is_empty() || d.first_forbidden(entries) {
            return;
        }
        for e in entries.iter() {
            let flags = d.flags(e);
            if d.accepts_rule_part(flags, last, affixes)
                && d.accepts_case(flags, surface, text, affixes)
                && affixes.first().is_none_or(|a| flags.contains(a.flag))
            {
                result.push(part(text, root, e, affixes));
            }
        }
    }
}

fn part<'d>(surface: &str, root: &str, entry: EntryId, affixes: &[&'d Affix]) -> CompoundPart<'d> {
    CompoundPart { surface: surface.to_string(), root: root.to_string(), entry, affixes: affixes.to_vec() }
}
