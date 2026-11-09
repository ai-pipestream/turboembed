//! `PiiRewriteStrategy` messages to #1902's [`RewriteStrategy`] and back.

use turbo_pii::rewrite::{
    Abbreviator, FormatRedactor, HmacTokenizer, KeepSide, MapReplacer, MapTarget, MaskPolicy, Pseudonymizer,
    RewriteStrategy, Sha256Tokenizer, StaticReplacer, Truncate, ZipSafeHarbor,
};

use super::error::{At, PolicyError};
use super::resolve::{PolicyResolver, check_dictionary, check_key_id};
use crate::proto::result::{
    PiiFormatStrategy, PiiKeepSide, PiiKeyedTokenStrategy, PiiLastNStrategy, PiiMapStrategy, PiiMapTarget,
    PiiMaskStrategy, PiiPseudonymStrategy, PiiRewriteStrategy, PiiSha256Strategy, PiiStaticStrategy, PiiStrategy,
    PiiTruncateStrategy, PiiZipSafeHarborStrategy, pii_map_strategy::Table, pii_rewrite_strategy::Strategy,
};

/// A whole-value mask with `'*'`: the conservative default, and what
/// redact-and-flag puts in place of a strategy it does not know.
pub fn whole_mask() -> RewriteStrategy {
    RewriteStrategy::Mask(MaskPolicy::new(MaskPolicy::DEFAULT_MASK).expect("'*' is ASCII"))
}

fn invalid(at: At, e: impl std::fmt::Display) -> PolicyError {
    PolicyError::Invalid { at, reason: e.to_string() }
}

/// The one character of `s`, or `default` when `s` is empty.
fn mask_char(at: At, s: &str, default: char) -> Result<char, PolicyError> {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (None, _) => Ok(default),
        (Some(c), None) => Ok(c),
        _ => Err(invalid(at, "mask_char is more than one character")),
    }
}

fn count(n: u32) -> usize {
    n as usize
}

pub(crate) fn mask(at: At, m: &PiiMaskStrategy) -> Result<RewriteStrategy, PolicyError> {
    let c = mask_char(at, &m.mask_char, MaskPolicy::DEFAULT_MASK)?;
    let policy = MaskPolicy::new(c).map_err(|e| invalid(at, e))?;
    Ok(RewriteStrategy::Mask(if m.keep_format { policy.keeping_format() } else { policy }))
}

fn truncate(t: Truncate, mask: char, keep_format: bool) -> RewriteStrategy {
    let t = t.with_mask(mask);
    RewriteStrategy::Truncate(if keep_format { t.keeping_format() } else { t })
}

pub(crate) fn keyed_token(
    at: At,
    k: &PiiKeyedTokenStrategy,
    resolver: &dyn PolicyResolver,
) -> Result<RewriteStrategy, PolicyError> {
    let key = k.key.as_ref().ok_or(PolicyError::Missing { at, field: "keyed_token.key" })?;
    check_key_id(at, &key.id)?;
    let bytes = resolver.hmac_key(&key.id, key.version).map_err(|e| PolicyError::Unresolved {
        at,
        field: "keyed_token.key",
        reason: e.reason(),
    })?;
    let length = if k.length == 0 { HmacTokenizer::DEFAULT_LENGTH } else { count(k.length) };
    // TokenizerError names lengths only, never the key.
    HmacTokenizer::with_length(bytes, length).map(RewriteStrategy::Hmac).map_err(|e| invalid(at, e))
}

fn map(
    at: At,
    m: &PiiMapStrategy,
    default: &RewriteStrategy,
    resolver: &dyn PolicyResolver,
) -> Result<RewriteStrategy, PolicyError> {
    let replacer = match &m.table {
        None => return Err(PolicyError::UnknownArm { at, field: "map.table" }),
        Some(Table::Entries(e)) => {
            // Keys equal once trimmed and folded are refused in both
            // tables, so inline and dictionary entries mean the same.
            MapReplacer::new_unique(e.entries.iter().map(|(k, v)| (k.as_str(), v.as_str())), m.case_sensitive)
        }
        Some(Table::Dictionary(d)) => {
            check_dictionary(at, &d.artifact_id, &d.sha256)?;
            let entries = resolver.map_entries(&d.artifact_id, &d.sha256).map_err(|e| PolicyError::Unresolved {
                at,
                field: "map.dictionary",
                reason: format!("dictionary {}: {}", d.artifact_id, e.reason()),
            })?;
            MapReplacer::new_unique(entries.iter().map(|(k, v)| (k.as_str(), v.as_str())), m.case_sensitive)
        }
    }
    .map_err(|e| invalid(at, e))?;
    let target = match PiiMapTarget::try_from(m.target) {
        Ok(PiiMapTarget::NormalizedOrSurface) => MapTarget::NormalizedOrSurface,
        Ok(PiiMapTarget::Normalized) => MapTarget::Normalized,
        Ok(PiiMapTarget::Surface) => MapTarget::Surface,
        Ok(PiiMapTarget::Unspecified) | Err(_) => {
            return Err(PolicyError::Unspecified { at, field: "map.target", value: m.target });
        }
    };
    let fallback = match m.fallback.as_deref() {
        None => default.clone(),
        Some(PiiRewriteStrategy { strategy: Some(Strategy::Map(_)) }) => return Err(PolicyError::NestedMap { at }),
        Some(f) => strategy(at, f, default, resolver)?.ok_or(PolicyError::UnknownArm { at, field: "map.fallback" })?,
    };
    Ok(RewriteStrategy::Map(Box::new(replacer.with_target(target).with_fallback(fallback))))
}

/// The most restricted prefixes a ZIP Safe Harbor table may list: every
/// three-digit area there is.
const MAX_ZIP_PREFIXES: usize = 1000;

fn zip_safe_harbor(at: At, z: &PiiZipSafeHarborStrategy) -> Result<RewriteStrategy, PolicyError> {
    let c = mask_char(at, &z.mask_char, ZipSafeHarbor::DEFAULT_MASK)?;
    let zip = if z.restricted_prefixes.is_empty() {
        ZipSafeHarbor::hhs_2000()
    } else if z.restricted_prefixes.len() > MAX_ZIP_PREFIXES {
        return Err(invalid(at, format_args!("more than {MAX_ZIP_PREFIXES} restricted ZIP prefixes")));
    } else {
        ZipSafeHarbor::with_restricted(&z.restricted_prefixes).map_err(|e| invalid(at, e))?
    };
    Ok(RewriteStrategy::ZipSafeHarbor(zip.with_mask(c)))
}

/// The native strategy of `s`, or `None` when its arm is unset or unknown
/// to this server (the caller decides between refusing and flagging).
/// `default` is what an unset map fallback gets.
pub(crate) fn strategy(
    at: At,
    s: &PiiRewriteStrategy,
    default: &RewriteStrategy,
    resolver: &dyn PolicyResolver,
) -> Result<Option<RewriteStrategy>, PolicyError> {
    let Some(arm) = &s.strategy else { return Ok(None) };
    Ok(Some(match arm {
        Strategy::Mask(m) => mask(at, m)?,
        Strategy::Format(PiiFormatStrategy { template }) => {
            RewriteStrategy::Format(FormatRedactor::new(template.as_str()).map_err(|e| invalid(at, e))?)
        }
        Strategy::Map(m) => map(at, m, default, resolver)?,
        Strategy::Truncate(t) => {
            let side = match PiiKeepSide::try_from(t.side) {
                Ok(PiiKeepSide::Start) => KeepSide::Start,
                Ok(PiiKeepSide::End) => KeepSide::End,
                Ok(PiiKeepSide::Unspecified) | Err(_) => {
                    return Err(PolicyError::Unspecified { at, field: "truncate.side", value: t.side });
                }
            };
            let c = mask_char(at, &t.mask_char, Truncate::DEFAULT_MASK)?;
            truncate(Truncate::new(side, count(t.count)), c, t.keep_format)
        }
        Strategy::LastN(l) => {
            let c = mask_char(at, &l.mask_char, Truncate::DEFAULT_MASK)?;
            truncate(Truncate::last_n(count(l.count)), c, l.keep_format)
        }
        Strategy::Abbreviate(a) => {
            let mut ab = Abbreviator::with_punctuation(a.separator.as_str(), a.trailing.as_str());
            if a.preserve_hyphens {
                ab = ab.preserving_hyphens();
            }
            if a.preserve_case {
                ab = ab.preserving_case();
            }
            RewriteStrategy::Abbreviate(ab)
        }
        Strategy::KeyedToken(k) => keyed_token(at, k, resolver)?,
        Strategy::Sha256(h) => {
            let length = if h.length == 0 { Sha256Tokenizer::DEFAULT_LENGTH } else { count(h.length) };
            let mut t = Sha256Tokenizer::with_length(length).map_err(|e| invalid(at, e))?;
            if h.kind_prefix {
                t = t.with_kind_prefix();
            }
            if !h.prefix.is_empty() {
                t = t.with_prefix(h.prefix.as_str());
            }
            RewriteStrategy::Sha256(t)
        }
        Strategy::StaticText(PiiStaticStrategy { replacement }) => {
            RewriteStrategy::Static(StaticReplacer::new(replacement.as_str()))
        }
        Strategy::Pseudonym(PiiPseudonymStrategy { prefix, suffix }) => {
            RewriteStrategy::Pseudonym(Pseudonymizer::new(prefix.as_str(), suffix.as_str()))
        }
        Strategy::ZipSafeHarbor(z) => zip_safe_harbor(at, z)?,
    }))
}

/// The `PiiStrategy` value an edit written by arm `s` records: exactly one
/// per arm, the same as the converted strategy's
/// [`RewriteStrategy::kind`].
pub fn edit_strategy(s: &Strategy) -> PiiStrategy {
    match s {
        Strategy::Mask(_) => PiiStrategy::Mask,
        Strategy::Format(_) => PiiStrategy::Format,
        Strategy::Map(_) => PiiStrategy::Map,
        Strategy::Truncate(_) | Strategy::LastN(_) => PiiStrategy::Truncate,
        Strategy::Abbreviate(_) => PiiStrategy::Abbreviate,
        Strategy::KeyedToken(_) => PiiStrategy::Hmac,
        Strategy::Sha256(_) => PiiStrategy::Sha256,
        Strategy::StaticText(_) => PiiStrategy::Static,
        Strategy::Pseudonym(_) => PiiStrategy::Pseudonym,
        Strategy::ZipSafeHarbor(_) => PiiStrategy::ZipSafeHarbor,
    }
}

fn char_string(c: char, default: char) -> String {
    if c == default { String::new() } else { c.to_string() }
}

/// The message for `s`, where the native value says all its message does:
/// mask, truncate (a kept end is written as last-N), format, SHA-256,
/// pseudonym and ZIP Safe Harbor. A keyed token holds key bytes and no reference, a map its
/// table, a static replacer per-kind texts and an abbreviator private
/// punctuation, so those give `None`; a strategy that refuses overlaps has
/// no message field and gives `None` too.
pub fn strategy_to_proto(s: &RewriteStrategy) -> Option<PiiRewriteStrategy> {
    if s.refuses_overlaps() {
        return None;
    }
    let arm = match s {
        RewriteStrategy::Mask(m) if m.leading() == 0 && m.trailing() == 0 => Strategy::Mask(PiiMaskStrategy {
            mask_char: char_string(m.mask(), MaskPolicy::DEFAULT_MASK),
            keep_format: m.keeps_format(),
        }),
        RewriteStrategy::Truncate(t) => {
            let mask_char = char_string(t.mask(), Truncate::DEFAULT_MASK);
            let count = u32::try_from(t.count()).ok()?;
            match t.side() {
                KeepSide::End => Strategy::LastN(PiiLastNStrategy { count, mask_char, keep_format: t.keeps_format() }),
                KeepSide::Start => Strategy::Truncate(PiiTruncateStrategy {
                    side: PiiKeepSide::Start as i32,
                    count,
                    mask_char,
                    keep_format: t.keeps_format(),
                }),
            }
        }
        RewriteStrategy::Format(f) => Strategy::Format(PiiFormatStrategy { template: f.template().to_owned() }),
        RewriteStrategy::Sha256(h) => Strategy::Sha256(PiiSha256Strategy {
            length: if h.length() == Sha256Tokenizer::DEFAULT_LENGTH { 0 } else { u32::try_from(h.length()).ok()? },
            kind_prefix: h.has_kind_prefix(),
            prefix: h.prefix().to_owned(),
        }),
        RewriteStrategy::Pseudonym(p) => {
            Strategy::Pseudonym(PiiPseudonymStrategy { prefix: p.prefix.clone(), suffix: p.suffix.clone() })
        }
        RewriteStrategy::ZipSafeHarbor(z) => Strategy::ZipSafeHarbor(PiiZipSafeHarborStrategy {
            mask_char: char_string(z.mask(), ZipSafeHarbor::DEFAULT_MASK),
            restricted_prefixes: if z.is_hhs_2000() { Vec::new() } else { z.restricted().map(str::to_owned).collect() },
        }),
        RewriteStrategy::Mask(_)
        | RewriteStrategy::Hmac(_)
        | RewriteStrategy::Map(_)
        | RewriteStrategy::Static(_)
        | RewriteStrategy::Abbreviate(_) => return None,
    };
    Some(PiiRewriteStrategy { strategy: Some(arm) })
}
