//! Reading OpenNLP's binary model format into a [`MaxentModel`].
//!
//! The layout, as `GISModelWriter`, `QNModelWriter` and
//! `PerceptronModelWriter` write it through `DataOutputStream`, with every
//! string read by `ModelParameterChunker.readUTF`:
//!
//! 1. The type string: `GIS`, `QN` or `Perceptron`.
//! 2. GIS only: an int and a double that the reader ignores (the writer
//!    always writes 1 and 1.0; they were the correction constant and
//!    parameter of older versions).
//! 3. The outcome count, then each outcome name.
//! 4. The outcome pattern count, then each pattern as a string of
//!    space-separated integers: how many consecutive predicates share the
//!    pattern, then the outcomes those predicates have parameters for.
//! 5. The predicate count, then each predicate name, sorted so that
//!    predicates sharing a pattern are consecutive.
//! 6. For each pattern, for each of its predicates, one double per
//!    outcome in the pattern.

use std::collections::HashMap;

use turbo_jvm::strict_math;

use crate::error::{Error, Result};
use crate::java_data::{JavaData, canonical_modified_utf8, decode_modified_utf8};
use crate::model::{MaxentModel, ModelKind};

/// `ResourceLimits.MAX_ENTRIES`: OpenNLP refuses outcome, pattern and
/// predicate counts above this.
const MAX_ENTRIES: i32 = 10_000_000;

impl MaxentModel {
    /// Reads a model from the bytes of a `.model` entry (or a standalone
    /// binary model file). Bytes after the parameters are ignored, as the
    /// Java reader ignores them.
    ///
    /// Where the Java reader would load a model that fails later, this one
    /// refuses it now: a model with no outcomes, and an outcome pattern
    /// naming an outcome the model does not have. `NaiveBayes` models, and
    /// any other type string, are refused with the type in the error.
    pub fn read(bytes: &[u8]) -> Result<MaxentModel> {
        let mut r = JavaData::new(bytes);
        let type_string = r.read_chunked_utf()?;
        let kind = match &*type_string {
            b"GIS" => ModelKind::Gis,
            b"QN" => ModelKind::Qn,
            b"Perceptron" => ModelKind::Perceptron,
            other => return Err(Error::ModelType(decode_modified_utf8(other).into())),
        };
        if kind == ModelKind::Gis {
            r.read_int()?;
            r.read_double()?;
        }

        let outcome_count = read_count(&mut r, "outcome")?;
        if outcome_count == 0 {
            return Err(Error::Count { what: "outcome", value: 0, offset: r.position() - 4 });
        }
        let mut outcomes: Vec<Box<str>> = Vec::with_capacity(capacity(&r, outcome_count));
        for _ in 0..outcome_count {
            outcomes.push(decode_modified_utf8(&r.read_chunked_utf()?).into());
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

        // Predicates beyond those the patterns describe have no parameters:
        // the Java reader leaves their Context null, which contributes
        // nothing to a sum, the same as an empty range here.
        let parameters: usize = patterns.iter().map(|p| p.predicates as usize * p.outcomes.len()).sum();
        if parameters > r.remaining() / 8 {
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
            // UniformPrior.setLabels: StrictMath.log(1.0 / numOutcomes).
            ModelKind::Gis => strict_math::log(1.0 / f64::from(outcome_count)),
            ModelKind::Qn | ModelKind::Perceptron => 0.0,
        };
        Ok(MaxentModel { kind, outcomes, predicates, pred_start, pred_outcome, pred_param, start })
    }
}

/// Reads a count and checks it as `AbstractModelReader` does.
fn read_count(r: &mut JavaData<'_>, what: &'static str) -> Result<u32> {
    let offset = r.position();
    let value = r.read_int()?;
    if !(0..=MAX_ENTRIES).contains(&value) {
        return Err(Error::Count { what, value, offset });
    }
    Ok(value as u32)
}

/// A capacity to reserve for `count` strings: each takes at least the two
/// bytes of its length, so a count the remaining data cannot hold does
/// not reserve memory for it.
fn capacity(r: &JavaData<'_>, count: u32) -> usize {
    (count as usize).min(r.remaining() / 2)
}

/// One outcome pattern.
struct Pattern {
    /// How many consecutive predicates share the pattern. A negative count
    /// in the file is zero here: the Java loop over it runs no times.
    predicates: u32,
    outcomes: Vec<u32>,
}

impl Pattern {
    /// Parses a pattern string as `getOutcomePatterns` does: tokens
    /// separated by spaces (a `StringTokenizer` on `" "`, so runs of spaces
    /// are one separator), each an `Integer.parseInt`.
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

/// `Integer.parseInt` for ASCII input: an optional sign, then decimal
/// digits, in range for an `i32`.
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

    /// Writes model data as `DataOutputStream` would.
    #[derive(Default)]
    struct Writer(Vec<u8>);

    impl Writer {
        fn utf(&mut self, s: &str) -> &mut Self {
            self.0.extend_from_slice(&(s.len() as u16).to_be_bytes());
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        fn int(&mut self, v: i32) -> &mut Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        fn double(&mut self, v: f64) -> &mut Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
    }

    /// A GIS model with outcomes x, y; pattern "2 0 1" for predicates a, b
    /// and pattern "1 1" for predicate c.
    fn gis() -> Writer {
        let mut w = Writer::default();
        w.utf("GIS").int(1).double(1.0);
        w.int(2).utf("x").utf("y");
        w.int(2).utf("2 0 1").utf("1 1");
        w.int(3).utf("a").utf("b").utf("c");
        w.double(0.5).double(-0.5).double(1.5).double(0.25).double(2.0);
        w
    }

    #[test]
    fn a_gis_model_is_read_into_columns() {
        let m = MaxentModel::read(&gis().0).unwrap();
        assert_eq!(m.kind(), ModelKind::Gis);
        assert_eq!(m.outcomes().collect::<Vec<_>>(), ["x", "y"]);
        assert_eq!(m.pred_start, [0, 2, 4, 5]);
        assert_eq!(m.pred_outcome, [0, 1, 0, 1, 1]);
        assert_eq!(m.pred_param, [0.5, -0.5, 1.5, 0.25, 2.0]);
        assert_eq!(m.predicates[&b"c"[..]], 2);
        assert_eq!(m.start, 0.5f64.ln());
    }

    #[test]
    fn qn_and_perceptron_have_no_gis_header() {
        for (name, kind) in [("QN", ModelKind::Qn), ("Perceptron", ModelKind::Perceptron)] {
            let mut w = Writer::default();
            w.utf(name).int(1).utf("only").int(1).utf("1 0").int(1).utf("p").double(3.0);
            let m = MaxentModel::read(&w.0).unwrap();
            assert_eq!(m.kind(), kind);
            assert_eq!(m.pred_param, [3.0]);
            assert_eq!(m.start, 0.0);
        }
    }

    #[test]
    fn unknown_and_naive_bayes_types_are_named() {
        let mut w = Writer::default();
        w.utf("NaiveBayes");
        assert!(matches!(MaxentModel::read(&w.0), Err(Error::ModelType(t)) if &*t == "NaiveBayes"));
    }

    #[test]
    fn a_repeated_predicate_name_maps_to_its_last_listing() {
        let mut w = Writer::default();
        w.utf("QN").int(1).utf("o").int(1).utf("2 0").int(2).utf("p").utf("p").double(1.0).double(2.0);
        let m = MaxentModel::read(&w.0).unwrap();
        assert_eq!(m.num_predicates(), 1);
        assert_eq!(m.pred_param, [1.0, 2.0]);
        assert_eq!(m.predicates[&b"p"[..]], 1);
    }

    #[test]
    fn predicates_the_patterns_do_not_describe_have_no_parameters() {
        let mut w = Writer::default();
        w.utf("QN").int(2).utf("o").utf("q").int(1).utf("1 0 1").int(2).utf("p").utf("r").double(1.0).double(2.0);
        let m = MaxentModel::read(&w.0).unwrap();
        assert_eq!(m.pred_start, [0, 2, 2]);
    }

    #[test]
    fn malformed_models_are_refused_with_the_reason() {
        let mut w = Writer::default();
        w.utf("QN").int(-1);
        assert!(matches!(MaxentModel::read(&w.0), Err(Error::Count { what: "outcome", value: -1, offset: 4 })));

        let mut w = Writer::default();
        w.utf("QN").int(0);
        assert!(matches!(MaxentModel::read(&w.0), Err(Error::Count { what: "outcome", value: 0, .. })));

        let mut w = Writer::default();
        w.utf("QN").int(1).utf("o").int(1).utf("1 1");
        assert!(matches!(
            MaxentModel::read(&w.0),
            Err(Error::OutcomePattern { pattern: 0, message: "an outcome index is outside the model's outcomes" })
        ));

        let mut w = Writer::default();
        w.utf("QN").int(1).utf("o").int(2).utf("1 0").utf("  ");
        assert!(matches!(
            MaxentModel::read(&w.0),
            Err(Error::OutcomePattern { pattern: 1, message: "the pattern is empty" })
        ));

        let mut w = Writer::default();
        w.utf("QN").int(1).utf("o").int(1).utf("x 0");
        assert!(matches!(MaxentModel::read(&w.0), Err(Error::OutcomePattern { pattern: 0, .. })));

        let mut w = Writer::default();
        w.utf("QN").int(1).utf("o").int(1).utf("3 0").int(2).utf("p").utf("q");
        assert!(matches!(MaxentModel::read(&w.0), Err(Error::PatternPredicates { described: 3, listed: 2 })));

        let mut truncated = gis().0;
        truncated.truncate(truncated.len() - 3);
        assert!(matches!(MaxentModel::read(&truncated), Err(Error::Data { .. })));

        let mut w = Writer::default();
        w.utf("QN").int(1).utf("o").int(20_000_000);
        assert!(matches!(MaxentModel::read(&w.0), Err(Error::Count { what: "outcome pattern", .. })));
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
