//! Sequence codecs: how a name finder's outcomes encode spans of tokens.
//!
//! - [`BioCodec`]: `<type>-start` on a span's first token, `<type>-cont`
//!   on the others, `other` outside spans.
//! - [`BilouCodec`]: `<type>-start`, `<type>-cont` and `<type>-last` for a
//!   span of several tokens, `<type>-unit` for a span of one, `other`
//!   outside.
//!
//! A span without a type is written with the type `default`. Each codec
//! has a [`SequenceValidator`] that keeps the beam search from producing
//! outcome sequences the codec cannot decode. All of it follows the
//! classes of `opennlp.tools.namefind` at the commit this crate follows,
//! including how they treat outcome sequences no model should produce
//! (see each method).

use std::collections::HashSet;

use crate::beam::SequenceValidator;

/// The type of a span without one (`SequenceCodec.DEFAULT_PREFIX` without
/// its hyphen).
pub const DEFAULT_TYPE: &str = "default";

/// A span of tokens, `[start, end)`, with an optional type:
/// `opennlp.tools.util.Span` without the probability.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: Option<String>,
}

impl Span {
    pub fn new(start: usize, end: usize, kind: Option<&str>) -> Span {
        assert!(start <= end, "a span ends at or after its start");
        Span { start, end, kind: kind.map(str::to_owned) }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Encodes spans as outcomes and decodes outcomes as spans:
/// `SequenceCodec<String>`.
pub trait SequenceCodec {
    /// The spans the outcomes encode.
    fn decode(&self, outcomes: &[&str]) -> Vec<Span>;

    /// The outcomes of `length` tokens holding the spans.
    ///
    /// # Panics
    ///
    /// If a span reaches past `length`, as Java throws.
    fn encode(&self, spans: &[Span], length: usize) -> Vec<String>;

    /// Whether a model's outcomes are ones this codec can decode.
    fn are_outcomes_compatible(&self, outcomes: &[&str]) -> bool;
}

/// `BioCodec.extractNameType`: the part of an outcome before its last
/// hyphen, when that hyphen is not the first character, everything after
/// it is one or more ASCII letters, digits or underscores, and the
/// outcome holds no line terminator (`\n`, `\r`, U+0085, U+2028, U+2029).
/// This is what the regular expression `(.+)-\w+` of OpenNLP 1.x and 2.x
/// matched.
pub fn name_type(outcome: &str) -> Option<&str> {
    let separator = outcome.rfind('-')?;
    let rest = &outcome[separator + 1..];
    let word = !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    let line_end = outcome.contains(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']);
    (separator > 0 && word && !line_end).then(|| &outcome[..separator])
}

/// Writes `<type>-<suffix>`, with `default` for a span without a type.
fn outcome(span: &Span, suffix: &str) -> String {
    format!("{}-{suffix}", span.kind.as_deref().unwrap_or(DEFAULT_TYPE))
}

/// The BIO codec: `start`, `cont`, `other`.
#[derive(Copy, Clone, Debug, Default)]
pub struct BioCodec;

impl BioCodec {
    pub const START: &'static str = "start";
    pub const CONTINUE: &'static str = "cont";
    pub const OTHER: &'static str = "other";
}

impl SequenceCodec for BioCodec {
    /// An outcome ending in `start` begins a span (ending the one before),
    /// one ending in `cont` extends it and one ending in `other` ends it;
    /// any other outcome is skipped. A span takes its type from the
    /// outcome of its last token ([`name_type`]); a `cont` with no span
    /// open moves the end of no span.
    fn decode(&self, outcomes: &[&str]) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut open: Option<usize> = None;
        let mut end = 0;
        for (i, tag) in outcomes.iter().enumerate() {
            if tag.ends_with(Self::START) {
                if let Some(start) = open {
                    spans.push(Span::new(start, end, name_type(outcomes[i - 1])));
                }
                open = Some(i);
                end = i + 1;
            } else if tag.ends_with(Self::CONTINUE) {
                end = i + 1;
            } else if tag.ends_with(Self::OTHER)
                && let Some(start) = open.take()
            {
                spans.push(Span::new(start, end, name_type(outcomes[i - 1])));
            }
        }
        if let Some(start) = open {
            spans.push(Span::new(start, end, name_type(outcomes[outcomes.len() - 1])));
        }
        spans
    }

    fn encode(&self, spans: &[Span], length: usize) -> Vec<String> {
        let mut outcomes = vec![Self::OTHER.to_owned(); length];
        for span in spans {
            outcomes[span.start] = outcome(span, Self::START);
            for slot in &mut outcomes[span.start + 1..span.end] {
                *slot = outcome(span, Self::CONTINUE);
            }
        }
        outcomes
    }

    /// At least one outcome ending in `start`, every outcome ending in
    /// `cont` with a `start` of the same prefix, and nothing else but
    /// `other`.
    fn are_outcomes_compatible(&self, outcomes: &[&str]) -> bool {
        let mut start = Vec::new();
        let mut cont = Vec::new();
        for o in outcomes {
            if let Some(prefix) = o.strip_suffix(Self::START) {
                start.push(prefix);
            } else if let Some(prefix) = o.strip_suffix(Self::CONTINUE) {
                cont.push(prefix);
            } else if *o != Self::OTHER {
                return false;
            }
        }
        !start.is_empty() && cont.iter().all(|c| start.contains(c))
    }
}

/// The BILOU codec: `start`, `cont`, `last`, `unit`, `other`.
#[derive(Copy, Clone, Debug, Default)]
pub struct BilouCodec;

impl BilouCodec {
    pub const START: &'static str = "start";
    pub const CONTINUE: &'static str = "cont";
    pub const LAST: &'static str = "last";
    pub const UNIT: &'static str = "unit";
    pub const OTHER: &'static str = "other";
}

impl SequenceCodec for BilouCodec {
    /// A `start` opens a span (dropping one still open), `cont` extends
    /// it, `last` closes it including the `last` token, and `unit` is a
    /// span of its own token. A span closed by `last` takes the type of the
    /// outcome before the `last`; a `unit` takes its own.
    fn decode(&self, outcomes: &[&str]) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut open: Option<usize> = None;
        let mut end = 0;
        for (i, tag) in outcomes.iter().enumerate() {
            if tag.ends_with(Self::START) {
                open = Some(i);
                end = i + 1;
            } else if tag.ends_with(Self::CONTINUE) {
                end = i + 1;
            } else if tag.ends_with(Self::LAST) {
                if let Some(start) = open.take() {
                    spans.push(Span::new(start, end + 1, name_type(outcomes[i - 1])));
                }
            } else if tag.ends_with(Self::UNIT) {
                spans.push(Span::new(i, i + 1, name_type(tag)));
            }
        }
        spans
    }

    fn encode(&self, spans: &[Span], length: usize) -> Vec<String> {
        let mut outcomes = vec![Self::OTHER.to_owned(); length];
        for span in spans {
            if span.len() > 1 {
                outcomes[span.start] = outcome(span, Self::START);
                for slot in &mut outcomes[span.start + 1..span.end - 1] {
                    *slot = outcome(span, Self::CONTINUE);
                }
                outcomes[span.end - 1] = outcome(span, Self::LAST);
            } else {
                // A span of one token, or of none: Java writes the unit at
                // `end - 1`, which for an empty span at 0 is an index out
                // of bounds.
                let at = span.end.checked_sub(1).expect("a span at 0 of no tokens has no token to mark");
                outcomes[at] = outcome(span, Self::UNIT);
            }
        }
        outcomes
    }

    /// At least one `start` or `unit` outcome; every `start` prefix with a
    /// `last`, every `cont` prefix with a `start` or a `last`, every `last`
    /// prefix with a `start`; and nothing else but `other`.
    fn are_outcomes_compatible(&self, outcomes: &[&str]) -> bool {
        let (mut start, mut cont, mut last, mut unit) =
            (HashSet::new(), HashSet::new(), HashSet::new(), HashSet::new());
        for o in outcomes {
            if let Some(p) = o.strip_suffix(Self::START) {
                start.insert(p);
            } else if let Some(p) = o.strip_suffix(Self::CONTINUE) {
                cont.insert(p);
            } else if let Some(p) = o.strip_suffix(Self::LAST) {
                last.insert(p);
            } else if let Some(p) = o.strip_suffix(Self::UNIT) {
                unit.insert(p);
            } else if *o != Self::OTHER {
                return false;
            }
        }
        if start.is_empty() && unit.is_empty() {
            return false;
        }
        start.iter().all(|p| last.contains(p))
            && cont.iter().all(|p| start.contains(p) || last.contains(p))
            && last.iter().all(|p| start.contains(p))
    }
}

/// Whether two outcomes name the same type, as the validators compare
/// them: both untyped, or both of the same type.
fn same_type_as_previous(previous: &str, outcome: &str) -> bool {
    let (previous_type, this_type) = (name_type(previous), name_type(outcome));
    match (previous_type, this_type) {
        (None, None) => true,
        (_, None) => false,
        (p, Some(t)) => p == Some(t),
    }
}

/// `NameFinderSequenceValidator`, the BIO codec's: a `cont` must follow a
/// `start` or `cont` of the same type, never begin the sentence or follow
/// an `other`.
#[derive(Copy, Clone, Debug, Default)]
pub struct BioValidator;

impl<T> SequenceValidator<T> for BioValidator {
    fn valid_sequence(&self, _index: usize, _sequence: &[T], prior: &[&str], outcome: &str) -> bool {
        if !outcome.ends_with(BioCodec::CONTINUE) {
            return true;
        }
        let Some(previous) = prior.last() else {
            return false;
        };
        if previous.ends_with(BioCodec::OTHER) {
            return false;
        }
        if previous.ends_with(BioCodec::CONTINUE) || previous.ends_with(BioCodec::START) {
            return same_type_as_previous(previous, outcome);
        }
        true
    }
}

/// `BilouNameFinderSequenceValidator`, the BILOU codec's: `cont` and
/// `last` must follow a `start` or `cont` of the same type, and after a
/// `start` or `cont` only `cont` or `last` may come.
#[derive(Copy, Clone, Debug, Default)]
pub struct BilouValidator;

impl<T> SequenceValidator<T> for BilouValidator {
    fn valid_sequence(&self, _index: usize, _sequence: &[T], prior: &[&str], outcome: &str) -> bool {
        let inside = outcome.ends_with(BilouCodec::CONTINUE) || outcome.ends_with(BilouCodec::LAST);
        if inside {
            let Some(previous) = prior.last() else {
                return false;
            };
            if previous.ends_with(BilouCodec::OTHER) || previous.ends_with(BilouCodec::UNIT) {
                return false;
            }
            if previous.ends_with(BilouCodec::LAST) {
                return false;
            }
            if previous.ends_with(BilouCodec::CONTINUE) || previous.ends_with(BilouCodec::START) {
                return same_type_as_previous(previous, outcome);
            }
        }
        if let Some(previous) = prior.last()
            && (outcome.ends_with(BilouCodec::START)
                || outcome.ends_with(BilouCodec::OTHER)
                || outcome.ends_with(BilouCodec::UNIT))
        {
            return !previous.ends_with(BilouCodec::START) && !previous.ends_with(BilouCodec::CONTINUE);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_types_need_a_word_after_the_last_hyphen() {
        assert_eq!(name_type("person-start"), Some("person"));
        assert_eq!(name_type("a-b-cont"), Some("a-b"));
        assert_eq!(name_type("start"), None);
        assert_eq!(name_type("-start"), None);
        assert_eq!(name_type("x-"), None);
        assert_eq!(name_type("x-st art"), None);
        assert_eq!(name_type("x\n-start"), None);
        assert_eq!(name_type("x-\u{E9}"), None);
    }

    #[test]
    fn bio_round_trips() {
        let spans = [Span::new(0, 2, Some("person")), Span::new(3, 4, None)];
        let encoded = BioCodec.encode(&spans, 5);
        assert_eq!(encoded, ["person-start", "person-cont", "other", "default-start", "other"]);
        let refs: Vec<&str> = encoded.iter().map(String::as_str).collect();
        let decoded = BioCodec.decode(&refs);
        assert_eq!(decoded, [Span::new(0, 2, Some("person")), Span::new(3, 4, Some("default"))]);
    }

    #[test]
    fn bilou_round_trips() {
        let spans = [Span::new(0, 3, Some("loc")), Span::new(4, 5, Some("person"))];
        let encoded = BilouCodec.encode(&spans, 5);
        assert_eq!(encoded, ["loc-start", "loc-cont", "loc-last", "other", "person-unit"]);
        let refs: Vec<&str> = encoded.iter().map(String::as_str).collect();
        assert_eq!(BilouCodec.decode(&refs), spans);
    }

    #[test]
    fn validators_follow_the_codecs() {
        let v = BioValidator;
        let ok = |prior: &[&str], o: &str| SequenceValidator::<&str>::valid_sequence(&v, 0, &[], prior, o);
        assert!(!ok(&[], "person-cont"));
        assert!(!ok(&["other"], "person-cont"));
        assert!(ok(&["person-start"], "person-cont"));
        assert!(!ok(&["loc-start"], "person-cont"));
        assert!(!ok(&["start"], "person-cont"));
        assert!(ok(&["start"], "cont"));
        let v = BilouValidator;
        let ok = |prior: &[&str], o: &str| SequenceValidator::<&str>::valid_sequence(&v, 0, &[], prior, o);
        assert!(!ok(&["person-start"], "other"));
        assert!(ok(&["person-start"], "person-last"));
        assert!(!ok(&["person-last"], "person-last"));
        assert!(ok(&["person-last"], "other"));
    }

    #[test]
    fn compatible_outcomes() {
        assert!(BioCodec.are_outcomes_compatible(&["other", "person-start", "person-cont"]));
        assert!(!BioCodec.are_outcomes_compatible(&["other", "person-cont"]));
        assert!(!BioCodec.are_outcomes_compatible(&["person-start", "x"]));
        assert!(BilouCodec.are_outcomes_compatible(&["other", "a-unit"]));
        assert!(!BilouCodec.are_outcomes_compatible(&["a-start", "a-unit"]));
        assert!(BilouCodec.are_outcomes_compatible(&["a-start", "a-last", "a-cont", "other"]));
    }
}
