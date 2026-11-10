//! The `turbo-pii:pii` detector a request's [`RedactionPolicy`] asks for:
//! disabled kinds are not detected, allow lists and label settings apply
//! per kind and globally, deny lists and custom identifiers find the
//! policy's custom kinds. Without a policy the detector is unchanged.

use turbo_pii::{CustomExtractor, Pack, PiiDetector, PiiEvidence, PiiKind, PiiKinds, PiiSink, Scanner};

use super::{CustomIdentifier, RedactionPolicy};

/// The name the custom identifier scanner records on its mentions.
pub const IDENTIFIERS_DETECTOR: &str = "turbo-pii:policy-identifiers";

/// Characters trimmed from both ends of a word before a custom identifier
/// pattern is tested against it.
const TRIM: &[char] = &[',', ';', ':', '!', '?', '.', '(', ')', '[', ']', '{', '}', '"', '\'', '<', '>'];

/// A policy's custom identifiers: each whitespace-separated word, trimmed
/// of surrounding punctuation, is tested against every pattern.
struct Identifiers {
    rules: Vec<CustomIdentifier>,
    kinds: Vec<PiiKind>,
    /// The policy digest, the version every mention records.
    version: String,
}

impl CustomExtractor for Identifiers {
    fn name(&self) -> &'static str {
        IDENTIFIERS_DETECTOR
    }

    fn kinds(&self) -> &[PiiKind] {
        &self.kinds
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn scan(&self, text: &str, sink: &mut PiiSink<'_>) {
        let mut at = 0usize;
        for word in text.split_inclusive(char::is_whitespace) {
            let start = at;
            at += word.len();
            let core = word.trim_end_matches(char::is_whitespace);
            let lead = core.len() - core.trim_start_matches(TRIM).len();
            let token = core.trim_matches(TRIM);
            if token.is_empty() {
                continue;
            }
            let s = (start + lead) as u32;
            let span = turbo_analysis::Span::new(s, s + token.len() as u32);
            if let Some(r) = self.rules.iter().find(|r| r.pattern.matches(token)) {
                sink.add_text(span, r.kind, &[PiiEvidence::Shape]);
            }
        }
    }

    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.str(&self.version).usize(self.rules.len());
        for r in &self.rules {
            h.str(&r.rule_id).u32(r.kind.code()).str(r.pattern.as_str());
        }
        h.finish()
    }
}

impl RedactionPolicy {
    /// The kinds this policy turns off.
    pub fn disabled_kinds(&self) -> impl Iterator<Item = PiiKind> + '_ {
        self.rules.iter().filter(|r| !r.enabled).map(|r| r.kind)
    }

    /// The opt-in packs ([`Pack::Dates`], [`Pack::Money`]) this policy
    /// turns on: those with a kind an enabled rule names. Dates and amounts
    /// are in most texts and only some are a person's, so `AllStructured`
    /// leaves them out until a policy asks for them.
    pub fn opted_in_packs(&self) -> impl Iterator<Item = Pack> + '_ {
        [Pack::Dates, Pack::Money].into_iter().filter(|p| {
            let kinds: Vec<PiiKind> = p.scanners().iter().flat_map(|s| s.kinds().iter()).collect();
            self.rules.iter().any(|r| r.enabled && kinds.contains(&r.kind))
        })
    }

    /// The detector of `pack` configured by this policy, with the opt-in
    /// packs the policy turns on ([`RedactionPolicy::opted_in_packs`]).
    pub fn detector(&self, pack: Pack) -> Result<PiiDetector, String> {
        let off: Vec<PiiKind> = self.disabled_kinds().collect();
        let mut scanners = Vec::new();
        let extra = self.opted_in_packs().filter(|p| *p != pack).flat_map(Pack::scanners);
        for s in pack.scanners().into_iter().chain(extra) {
            let kinds: Vec<PiiKind> = s.kinds().iter().filter(|k| !off.contains(k)).collect();
            if kinds.is_empty() {
                continue;
            }
            let kept = PiiKinds::of(&kinds);
            scanners.push(if kept == s.kinds() {
                s
            } else {
                Scanner::with_kinds(s.extractor(), kept).map_err(|e| e.to_string())?
            });
        }
        let mut d = PiiDetector::new(scanners)
            .try_with_custom_kinds(self.custom_kinds.clone())
            .map_err(|e| e.to_string())?
            .with_allow_list(self.allow.clone());
        for r in self.rules.iter().filter(|r| r.enabled) {
            if !r.allow.is_empty() {
                d = d.with_kind_allow_list(r.kind, r.allow.clone()).map_err(|e| e.to_string())?;
            }
            if let Some(l) = &r.labels {
                d = d.with_label_policy(r.kind, l.policy);
                if let Some(w) = l.window_words {
                    d = d.with_label_window(r.kind, w);
                }
                d = d.with_extra_labels(r.kind, l.extra_labels.iter().map(String::as_str));
            }
        }
        for deny in &self.deny {
            if !off.contains(&deny.list.kind()) {
                d = d.with_deny_list(deny.list.clone()).map_err(|e| e.to_string())?;
            }
        }
        let rules: Vec<CustomIdentifier> =
            self.custom_identifiers.iter().filter(|c| !off.contains(&c.kind)).cloned().collect();
        if !rules.is_empty() {
            let mut kinds: Vec<PiiKind> = rules.iter().map(|r| r.kind).collect();
            kinds.dedup();
            let ids = Identifiers { rules, kinds, version: self.digest.to_hex() };
            d = d.try_with_scanner(Scanner::custom(ids)).map_err(|e| e.to_string())?;
        }
        Ok(d)
    }
}
