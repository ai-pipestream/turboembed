//! The canonical encoding of a `RedactionPolicy` message, which
//! [`super::PolicyDigest::of_canonical`] hashes. Protobuf bytes are not
//! canonical (field order, map order and unknown fields vary), so they are
//! never digested.
//!
//! The encoding is the UTF-8 domain tag `turbo.pii-policy.v1` as a string
//! (below), then the policy message. All integers are little-endian.
//!
//! - A message is its present fields in ascending field-number order, each
//!   as the field number (`u32`) and the value. Nothing marks its end: a
//!   nested message is length-prefixed.
//! - A field is present when it holds a non-default value: a scalar other
//!   than 0, `false` or the empty string; a set message field (even an
//!   empty message); a non-empty repeated or map field; the set arm of a
//!   oneof, whatever its value. Unknown fields are dropped by decoding and
//!   so never encoded. A field added later is absent in every older policy,
//!   which keeps that policy's digest.
//! - Values: `uint32`, and `int32` and enums by their `i32` bits, as `u32`; `uint64` as
//!   `u64`; `bool` as one byte `1`; `float` as the `u32` of its IEEE 754
//!   bits, with `-0.0` written as `0.0` (and so absent as a field); a
//!   string as its byte length (`u64`) and its bytes; a nested message as
//!   its encoding's byte length (`u64`) and that encoding.
//! - A repeated field is its element count (`u64`) and each element's
//!   value in order: order is part of a policy.
//! - A map field is its entry count (`u64`) and each entry's key and value,
//!   sorted by the key's bytes.

use crate::proto::result as pb;
use pb::{
    pii_default_strategy, pii_kind_ref, pii_map_strategy, pii_rewrite_strategy, pii_strategy_condition, pii_term_source,
};

struct Enc(Vec<u8>);

impl Enc {
    fn tag(&mut self, n: u32) {
        self.0.extend_from_slice(&n.to_le_bytes());
    }

    fn raw_u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn raw_u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn raw_str(&mut self, s: &str) {
        self.raw_u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }

    fn u32(&mut self, n: u32, v: u32) {
        if v != 0 {
            self.tag(n);
            self.raw_u32(v);
        }
    }

    fn enm(&mut self, n: u32, v: i32) {
        self.u32(n, v as u32);
    }

    fn i32(&mut self, n: u32, v: i32) {
        self.u32(n, v as u32);
    }

    fn u64(&mut self, n: u32, v: u64) {
        if v != 0 {
            self.tag(n);
            self.raw_u64(v);
        }
    }

    fn bool(&mut self, n: u32, v: bool) {
        if v {
            self.tag(n);
            self.0.push(1);
        }
    }

    fn f32(&mut self, n: u32, v: f32) {
        let v = if v == 0.0 { 0.0f32 } else { v };
        self.u32(n, v.to_bits());
    }

    fn str(&mut self, n: u32, s: &str) {
        if !s.is_empty() {
            self.tag(n);
            self.raw_str(s);
        }
    }

    /// A nested message's value: its length, then its encoding.
    fn nested(&mut self, f: impl FnOnce(&mut Enc)) {
        let at = self.0.len();
        self.raw_u64(0);
        f(self);
        let len = (self.0.len() - at - 8) as u64;
        self.0[at..at + 8].copy_from_slice(&len.to_le_bytes());
    }

    /// A message field, present when set.
    fn msg<T>(&mut self, n: u32, v: Option<&T>, f: impl FnOnce(&mut Enc, &T)) {
        if let Some(v) = v {
            self.tag(n);
            self.nested(|e| f(e, v));
        }
    }

    fn rep_msg<T>(&mut self, n: u32, items: &[T], f: impl Fn(&mut Enc, &T)) {
        if !items.is_empty() {
            self.tag(n);
            self.raw_u64(items.len() as u64);
            for item in items {
                self.nested(|e| f(e, item));
            }
        }
    }

    fn rep_str(&mut self, n: u32, items: &[String]) {
        if !items.is_empty() {
            self.tag(n);
            self.raw_u64(items.len() as u64);
            for s in items {
                self.raw_str(s);
            }
        }
    }

    fn rep_enum(&mut self, n: u32, items: &[i32]) {
        if !items.is_empty() {
            self.tag(n);
            self.raw_u64(items.len() as u64);
            for &v in items {
                self.raw_u32(v as u32);
            }
        }
    }

    /// A oneof arm holding a message: always present once set.
    fn arm<T>(&mut self, n: u32, v: &T, f: impl FnOnce(&mut Enc, &T)) {
        self.tag(n);
        self.nested(|e| f(e, v));
    }
}

/// The canonical encoding of `p`.
pub fn encode(p: &pb::RedactionPolicy) -> Vec<u8> {
    let mut e = Enc(Vec::with_capacity(256 + 128 * p.kind_rules.len()));
    e.raw_str(super::PolicyDigest::DOMAIN);
    e.str(1, &p.id);
    e.u64(2, p.revision);
    e.rep_str(3, &p.custom_kinds);
    e.rep_msg(4, &p.kind_rules, kind_rule);
    e.msg(5, p.default_strategy.as_ref(), default_strategy);
    e.msg(6, p.allow.as_ref(), allow_list);
    e.rep_msg(7, &p.deny, deny_list);
    e.rep_msg(8, &p.custom_identifiers, custom_identifier);
    e.bool(9, p.redact_and_flag_unknown);
    e.0
}

fn kind_ref(e: &mut Enc, k: &pb::PiiKindRef) {
    match k.kind {
        None => {}
        Some(pii_kind_ref::Kind::Builtin(id)) => {
            e.tag(1);
            e.raw_u32(id as u32);
        }
        Some(pii_kind_ref::Kind::Custom(i)) => {
            e.tag(2);
            e.raw_u32(i);
        }
    }
}

fn kind_rule(e: &mut Enc, r: &pb::PiiKindRule) {
    e.str(1, &r.rule_id);
    e.msg(2, r.kind.as_ref(), kind_ref);
    e.enm(3, r.state);
    e.rep_msg(4, &r.strategies, strategy_entry);
    e.msg(5, r.allow.as_ref(), allow_list);
    e.msg(6, r.labels.as_ref(), |e, l: &pb::PiiLabelOverride| {
        e.enm(1, l.requirement);
        e.u32(2, l.window_words);
        e.rep_str(3, &l.extra_labels);
    });
    e.msg(7, r.release_ceiling.as_ref(), |e, c: &pb::PiiReleaseCeiling| e.rep_enum(1, &c.forms));
}

fn strategy_entry(e: &mut Enc, s: &pb::PiiStrategyEntry) {
    e.rep_msg(1, &s.when, condition);
    e.msg(2, s.strategy.as_ref(), strategy);
}

fn condition(e: &mut Enc, c: &pb::PiiStrategyCondition) {
    use pii_strategy_condition::Test;
    match &c.test {
        None => {}
        Some(Test::ChecksumPassed(p)) => e.arm(1, p, |e, p| e.rep_enum(1, &p.algorithms)),
        Some(Test::LabelPresent(p)) => e.arm(2, p, |_, _| {}),
        Some(Test::Detector(d)) => e.arm(3, d, |e, d| {
            e.str(1, &d.id);
            e.str(2, &d.version);
        }),
        Some(Test::ScoreAtLeast(s)) => e.arm(4, s, |e, s| {
            e.enm(1, s.kind);
            e.f32(2, s.min);
            e.bool(3, s.calibrated);
        }),
    }
}

fn mask(e: &mut Enc, m: &pb::PiiMaskStrategy) {
    e.str(1, &m.mask_char);
    e.bool(2, m.keep_format);
}

fn keyed_token(e: &mut Enc, k: &pb::PiiKeyedTokenStrategy) {
    e.msg(1, k.key.as_ref(), |e, r: &pb::PiiPolicyKeyRef| {
        e.str(1, &r.id);
        e.u32(2, r.version);
    });
    e.u32(2, k.length);
}

fn date(e: &mut Enc, d: &pb::PiiDateStrategy) {
    use pb::pii_date_strategy::Mode;
    match &d.mode {
        None => {}
        Some(Mode::Year(y)) => e.arm(1, y, |_, _| {}),
        Some(Mode::Shift(s)) => e.arm(2, s, |e, s| {
            e.i32(1, s.years);
            e.i32(2, s.months);
            e.i32(3, s.days);
            e.msg(4, s.key.as_ref(), |e, r: &pb::PiiPolicyKeyRef| {
                e.str(1, &r.id);
                e.u32(2, r.version);
            });
            e.u32(5, s.max_days);
            e.str(6, &s.context);
        }),
        Some(Mode::Relative(r)) => e.arm(3, r, |e, r| e.str(1, &r.reference)),
        Some(Mode::SafeHarbor(r)) => e.arm(5, r, |e, r| e.str(1, &r.reference)),
    }
    e.str(4, &d.mask_char);
}

fn dictionary(e: &mut Enc, d: &pb::PiiDictionaryRef) {
    e.str(1, &d.artifact_id);
    e.str(2, &d.sha256);
}

fn strategy(e: &mut Enc, s: &pb::PiiRewriteStrategy) {
    use pii_rewrite_strategy::Strategy as S;
    match &s.strategy {
        None => {}
        Some(S::Mask(m)) => e.arm(1, m, mask),
        Some(S::Format(f)) => e.arm(2, f, |e, f| e.str(1, &f.template)),
        Some(S::Map(m)) => e.arm(3, &**m, map),
        Some(S::Truncate(t)) => e.arm(4, t, |e, t| {
            e.enm(1, t.side);
            e.u32(2, t.count);
            e.str(3, &t.mask_char);
            e.bool(4, t.keep_format);
        }),
        Some(S::LastN(l)) => e.arm(5, l, |e, l| {
            e.u32(1, l.count);
            e.str(2, &l.mask_char);
            e.bool(3, l.keep_format);
        }),
        Some(S::Abbreviate(a)) => e.arm(6, a, |e, a| {
            e.str(1, &a.separator);
            e.str(2, &a.trailing);
            e.bool(3, a.preserve_hyphens);
            e.bool(4, a.preserve_case);
        }),
        Some(S::KeyedToken(k)) => e.arm(7, k, keyed_token),
        Some(S::Sha256(h)) => e.arm(8, h, |e, h| {
            e.u32(1, h.length);
            e.bool(2, h.kind_prefix);
            e.str(3, &h.prefix);
        }),
        Some(S::StaticText(t)) => e.arm(9, t, |e, t| e.str(1, &t.replacement)),
        Some(S::Pseudonym(p)) => e.arm(10, p, |e, p| {
            e.str(1, &p.prefix);
            e.str(2, &p.suffix);
        }),
        Some(S::ZipSafeHarbor(z)) => e.arm(11, z, |e, z| {
            e.str(1, &z.mask_char);
            // A set: the strategy sorts and folds it, so the digest does,
            // and the bundled table spelled out digests as empty.
            let mut prefixes = z.restricted_prefixes.clone();
            prefixes.sort_unstable();
            prefixes.dedup();
            if prefixes.iter().map(String::as_str).eq(turbo_pii::rewrite::HHS_2000_RESTRICTED_PREFIXES) {
                prefixes.clear();
            }
            e.rep_str(2, &prefixes);
        }),
        Some(S::AgeSafeHarbor(a)) => e.arm(12, a, |e, a| e.str(1, &a.mask_char)),
        Some(S::Date(d)) => e.arm(13, d, date),
    }
}

fn map(e: &mut Enc, m: &pb::PiiMapStrategy) {
    match &m.table {
        None => {}
        Some(pii_map_strategy::Table::Entries(t)) => e.arm(1, t, |e, t| {
            if t.entries.is_empty() {
                return;
            }
            let mut entries: Vec<(&String, &String)> = t.entries.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            e.tag(1);
            e.raw_u64(entries.len() as u64);
            for (k, v) in entries {
                e.raw_str(k);
                e.raw_str(v);
            }
        }),
        Some(pii_map_strategy::Table::Dictionary(d)) => e.arm(2, d, dictionary),
    }
    e.bool(3, m.case_sensitive);
    e.enm(4, m.target);
    e.msg(5, m.fallback.as_deref(), strategy);
}

fn default_strategy(e: &mut Enc, d: &pb::PiiDefaultStrategy) {
    match &d.strategy {
        None => {}
        Some(pii_default_strategy::Strategy::Mask(m)) => e.arm(1, m, mask),
        Some(pii_default_strategy::Strategy::KeyedToken(k)) => e.arm(2, k, keyed_token),
    }
}

fn term_source(e: &mut Enc, t: &pb::PiiTermSource) {
    match &t.source {
        None => {}
        Some(pii_term_source::Source::Terms(t)) => e.arm(1, t, |e, t| e.rep_str(1, &t.terms)),
        Some(pii_term_source::Source::Dictionary(d)) => e.arm(2, d, dictionary),
    }
}

fn allow_list(e: &mut Enc, a: &pb::PiiAllowList) {
    e.rep_msg(1, &a.terms, term_source);
    e.bool(2, a.case_sensitive);
    e.rep_msg(3, &a.patterns, |e, p| {
        e.enm(1, p.syntax);
        e.str(2, &p.pattern);
        e.bool(3, p.case_sensitive);
    });
}

fn deny_list(e: &mut Enc, d: &pb::PiiDenyList) {
    e.str(1, &d.rule_id);
    e.msg(2, d.kind.as_ref(), kind_ref);
    e.msg(3, d.terms.as_ref(), term_source);
    e.bool(4, d.case_sensitive);
    e.u32(5, d.max_edits);
    e.bool(6, d.capitalized_only);
}

fn custom_identifier(e: &mut Enc, c: &pb::PiiCustomIdentifier) {
    e.str(1, &c.rule_id);
    e.msg(2, c.kind.as_ref(), kind_ref);
    e.str(3, &c.shape);
    e.bool(4, c.case_sensitive);
}
