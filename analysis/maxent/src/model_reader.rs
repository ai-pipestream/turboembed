//! Reading binary maxent, QN, Perceptron, and NaiveBayes models trained by
//! `turbo-train` into a [`MaxentModel`], while keeping external legacy
//! [`MaxentModel::read`] disabled under `#1511`.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::java_data::{CHUNKED_SIGNATURE, canonical_modified_utf8, decode_modified_utf8};
use crate::java_string::JavaString;
use crate::model::{MaxentModel, ModelKind, outcome_texts};

/// Maximum outcome, pattern and predicate count accepted by the reader.
const MAX_ENTRIES: i32 = 10_000_000;

impl MaxentModel {
    /// Returns the retirement error without decoding external legacy model bytes.
    /// Use [`MaxentModel::from_trained_bytes`] to decode models trained by `turbo-train`.
    pub fn read(_bytes: &[u8]) -> Result<Self> {
        Err(Error::Disabled)
    }

    /// Decodes a [`MaxentModel`] from the native binary bytes emitted by `turbo-train`'s
    /// GIS, L-BFGS (`QN`), Perceptron, or NaiveBayes writers (or a `ModelFile`
    /// archive entry).
    pub fn from_trained_bytes(bytes: &[u8]) -> Result<MaxentModel> {
        let mut r = NativeReader::new(bytes);
        let type_string = r.read_chunked_utf()?;
        let kind = match &*type_string {
            b"TURBO-GIS-1" => ModelKind::Gis,
            b"TURBO-QN-1" => ModelKind::Qn,
            b"TURBO-PERCEPTRON-1" => ModelKind::Perceptron,
            b"TURBO-NAIVE-BAYES-1" => ModelKind::NaiveBayes,
            other => return Err(Error::ModelType(decode_modified_utf8(other).into())),
        };

        let outcome_count = read_count(&mut r, "outcome")?;
        if outcome_count == 0 {
            return Err(Error::Count { what: "outcome", value: 0, offset: r.position() - 4 });
        }
        let mut outcomes: Vec<JavaString> = Vec::with_capacity(capacity(&r, outcome_count));
        for _ in 0..outcome_count {
            outcomes.push(JavaString::from_checked(&r.read_chunked_utf()?));
        }

        let pattern_count = read_count(&mut r, "outcome pattern")?;
        let mut patterns: Vec<Pattern> = Vec::with_capacity(capacity(&r, pattern_count));
        for index in 0..pattern_count {
            patterns.push(Pattern::parse(&r.read_chunked_utf()?, index, outcome_count)?);
        }

        let predicate_count = read_count(&mut r, "predicate")?;
        let described: u64 = patterns.iter().map(|p| u64::from(p.predicates)).sum();
        if described > u64::from(predicate_count) {
            return Err(Error::PatternPredicates { described, listed: predicate_count });
        }
        let mut predicates = HashMap::with_capacity_and_hasher(capacity(&r, predicate_count), Default::default());
        for index in 0..predicate_count {
            let name = r.read_chunked_utf()?;
            predicates.insert(Box::from(&*canonical_modified_utf8(&name)), index);
        }

        let parameters: usize = patterns
            .iter()
            .fold(0usize, |acc, p| acc.saturating_add((p.predicates as usize).saturating_mul(p.outcomes.len())));
        if parameters > r.remaining() / 8 || u32::try_from(parameters).is_err() {
            return Err(Error::Data { offset: r.position(), message: "the data ends inside the parameters" });
        }
        let mut pred_start = Vec::with_capacity(predicate_count as usize + 1);
        let mut pred_outcome = Vec::with_capacity(parameters);
        let mut pred_param = Vec::with_capacity(parameters);
        pred_start.push(0u32);
        for pattern in &patterns {
            for _ in 0..pattern.predicates {
                for &outcome in &pattern.outcomes {
                    pred_outcome.push(outcome);
                    pred_param.push(r.read_double()?);
                }
                pred_start.push(pred_outcome.len() as u32);
            }
        }
        let last = *pred_start.last().expect("pred_start starts with 0");
        pred_start.resize(predicate_count as usize + 1, last);

        let start = match kind {
            ModelKind::Gis => (1.0 / f64::from(outcome_count)).ln(),
            ModelKind::Qn | ModelKind::Perceptron | ModelKind::NaiveBayes => 0.0,
        };
        let (outcome_totals, vocabulary) = if kind == ModelKind::NaiveBayes {
            let mut totals = vec![0.0; outcome_count as usize];
            for (&o, &w) in pred_outcome.iter().zip(&pred_param) {
                totals[o as usize] += w;
            }
            (totals, f64::from(predicate_count))
        } else {
            (Vec::new(), 0.0)
        };
        let outcome_texts = outcome_texts(&outcomes);
        Ok(MaxentModel {
            kind,
            outcomes,
            outcome_texts,
            predicates,
            pred_start,
            pred_outcome,
            pred_param,
            start,
            outcome_totals,
            vocabulary,
        })
    }
}

struct NativeReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> NativeReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.remaining() < len {
            return Err(Error::Data { offset: self.pos, message: "unexpected end of data" });
        }
        let slice = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn read_int(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn read_double(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_utf(&mut self) -> Result<&'a [u8]> {
        let len = self.read_u16()? as usize;
        self.take(len)
    }

    fn read_chunked_utf(&mut self) -> Result<Cow<'a, [u8]>> {
        let first = self.read_utf()?;
        let Some(rest) = first.strip_prefix(CHUNKED_SIGNATURE) else {
            return Ok(Cow::Borrowed(first));
        };
        let count = parse_int(rest)
            .filter(|&n| n >= 0)
            .ok_or(Error::Data { offset: self.pos, message: "invalid chunk count in CHUNKED-MODEL-PARAMS" })?
            as usize;
        let mut out = Vec::new();
        for _ in 0..count {
            out.extend_from_slice(self.read_utf()?);
        }
        Ok(Cow::Owned(out))
    }
}

fn read_count(r: &mut NativeReader<'_>, what: &'static str) -> Result<u32> {
    let offset = r.position();
    let value = r.read_int()?;
    if !(0..=MAX_ENTRIES).contains(&value) {
        return Err(Error::Count { what, value, offset });
    }
    Ok(value as u32)
}

fn capacity(r: &NativeReader<'_>, count: u32) -> usize {
    (count as usize).min(r.remaining() / 2)
}

struct Pattern {
    predicates: u32,
    outcomes: Vec<u32>,
}

impl Pattern {
    fn parse(text: &[u8], index: u32, outcome_count: u32) -> Result<Pattern> {
        let error = |message| Error::OutcomePattern { pattern: index, message };
        let mut tokens = text.split(|&b| b == b' ').filter(|t| !t.is_empty());
        let first = tokens.next().ok_or(error("the pattern is empty"))?;
        let predicates = parse_int(first).ok_or(error("the predicate count is not an integer"))?;
        let mut outcomes = Vec::new();
        for token in tokens {
            let outcome = parse_int(token).ok_or(error("an outcome is not an integer"))?;
            if outcome < 0 || outcome as u32 >= outcome_count {
                return Err(error("an outcome index is outside the model's outcomes"));
            }
            outcomes.push(outcome as u32);
        }
        Ok(Pattern { predicates: predicates.max(0) as u32, outcomes })
    }
}

fn parse_int(token: &[u8]) -> Option<i32> {
    let (negative, digits) = match token.split_first()? {
        (b'-', rest) => (true, rest),
        (b'+', rest) => (false, rest),
        _ => (false, token),
    };
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut value: i64 = 0;
    for &d in digits {
        value = value * 10 + i64::from(d - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    i32::try_from(if negative { -value } else { value }).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Writer(Vec<u8>);

    impl Writer {
        fn utf(&mut self, s: &str) -> &mut Self {
            self.0.extend_from_slice(&(s.len() as u16).to_le_bytes());
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        fn int(&mut self, v: i32) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn double(&mut self, v: f64) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
    }

    fn gis() -> Writer {
        let mut w = Writer::default();
        w.utf("TURBO-GIS-1");
        w.int(2).utf("x").utf("y");
        w.int(2).utf("2 0 1").utf("1 1");
        w.int(3).utf("a").utf("b").utf("c");
        w.double(0.5).double(-0.5).double(1.5).double(0.25).double(2.0);
        w
    }

    #[test]
    fn a_gis_model_is_read_into_columns() {
        let m = MaxentModel::from_trained_bytes(&gis().0).unwrap();
        assert_eq!(m.kind(), ModelKind::Gis);
        assert_eq!(m.outcomes().collect::<Vec<_>>(), [&JavaString::new("x"), &JavaString::new("y")]);
        assert_eq!(m.pred_start, [0, 2, 4, 5]);
        assert_eq!(m.pred_outcome, [0, 1, 0, 1, 1]);
        assert_eq!(m.pred_param, [0.5, -0.5, 1.5, 0.25, 2.0]);
        assert_eq!(m.predicates[&b"c"[..]], 2);
        assert_eq!(m.start, 0.5f64.ln());
    }

    #[test]
    fn qn_perceptron_and_naive_bayes_have_no_gis_header() {
        for (name, kind) in [
            ("TURBO-QN-1", ModelKind::Qn),
            ("TURBO-PERCEPTRON-1", ModelKind::Perceptron),
            ("TURBO-NAIVE-BAYES-1", ModelKind::NaiveBayes),
        ] {
            let mut w = Writer::default();
            w.utf(name).int(1).utf("only").int(1).utf("1 0").int(1).utf("p").double(3.0);
            let m = MaxentModel::from_trained_bytes(&w.0).unwrap();
            assert_eq!(m.kind(), kind);
            assert_eq!(m.pred_param, [3.0]);
            assert_eq!(m.start, 0.0);
            if kind == ModelKind::NaiveBayes {
                assert_eq!(m.outcome_totals, [3.0]);
                assert_eq!(m.vocabulary, 1.0);
            }
        }
    }

    #[test]
    fn unknown_model_types_are_named() {
        let mut w = Writer::default();
        w.utf("Mystery");
        assert!(matches!(MaxentModel::from_trained_bytes(&w.0), Err(Error::ModelType(t)) if &*t == "Mystery"));
        let mut legacy = Writer::default();
        legacy.utf("GIS");
        assert!(matches!(MaxentModel::from_trained_bytes(&legacy.0), Err(Error::ModelType(t)) if &*t == "GIS"));
    }

    #[test]
    fn a_repeated_predicate_name_maps_to_its_last_listing() {
        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(1).utf("o").int(1).utf("2 0").int(2).utf("p").utf("p").double(1.0).double(2.0);
        let m = MaxentModel::from_trained_bytes(&w.0).unwrap();
        assert_eq!(m.num_predicates(), 1);
        assert_eq!(m.pred_param, [1.0, 2.0]);
        assert_eq!(m.predicates[&b"p"[..]], 1);
    }

    #[test]
    fn predicates_the_patterns_do_not_describe_have_no_parameters() {
        let mut w = Writer::default();
        w.utf("TURBO-QN-1")
            .int(2)
            .utf("o")
            .utf("q")
            .int(1)
            .utf("1 0 1")
            .int(2)
            .utf("p")
            .utf("r")
            .double(1.0)
            .double(2.0);
        let m = MaxentModel::from_trained_bytes(&w.0).unwrap();
        assert_eq!(m.pred_start, [0, 2, 2]);
    }

    #[test]
    fn malformed_models_are_refused_with_the_reason() {
        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(-1);
        assert!(matches!(
            MaxentModel::from_trained_bytes(&w.0),
            Err(Error::Count { what: "outcome", value: -1, offset: 12 })
        ));

        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(0);
        assert!(matches!(MaxentModel::from_trained_bytes(&w.0), Err(Error::Count { what: "outcome", value: 0, .. })));

        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(1).utf("o").int(1).utf("1 1");
        assert!(matches!(
            MaxentModel::from_trained_bytes(&w.0),
            Err(Error::OutcomePattern { pattern: 0, message: "an outcome index is outside the model's outcomes" })
        ));

        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(1).utf("o").int(2).utf("1 0").utf("  ");
        assert!(matches!(
            MaxentModel::from_trained_bytes(&w.0),
            Err(Error::OutcomePattern { pattern: 1, message: "the pattern is empty" })
        ));

        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(1).utf("o").int(1).utf("x 0");
        assert!(matches!(MaxentModel::from_trained_bytes(&w.0), Err(Error::OutcomePattern { pattern: 0, .. })));

        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(1).utf("o").int(1).utf("3 0").int(2).utf("p").utf("q");
        assert!(matches!(
            MaxentModel::from_trained_bytes(&w.0),
            Err(Error::PatternPredicates { described: 3, listed: 2 })
        ));

        let mut truncated = gis().0;
        truncated.truncate(truncated.len() - 3);
        assert!(matches!(MaxentModel::from_trained_bytes(&truncated), Err(Error::Data { .. })));

        let mut w = Writer::default();
        w.utf("TURBO-QN-1").int(1).utf("o").int(20_000_000);
        assert!(matches!(MaxentModel::from_trained_bytes(&w.0), Err(Error::Count { what: "outcome pattern", .. })));
    }

    #[test]
    fn integers_follow_parse_int() {
        assert_eq!(parse_int(b"0"), Some(0));
        assert_eq!(parse_int(b"-12"), Some(-12));
        assert_eq!(parse_int(b"+7"), Some(7));
        assert_eq!(parse_int(b"-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int(b"2147483648"), None);
        assert_eq!(parse_int(b"1a"), None);
        assert_eq!(parse_int(b"+"), None);
    }

    #[test]
    fn a_negative_predicate_count_in_a_pattern_covers_no_predicates() {
        let p = Pattern::parse(b"-3 0", 0, 1).unwrap();
        assert_eq!(p.predicates, 0);
        assert_eq!(p.outcomes, [0]);
    }
}
