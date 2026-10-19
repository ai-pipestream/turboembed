//! A decision point dump as typed values: the header a golden has, then
//! every `eval` call the annotator made of its maxent model, with the
//! contexts it passed and the probabilities it got back.
//! docs/analysis.md, "Decision point dumps", describes the format.
//!
//! The contexts are kept as UTF-16 units: the tokenizer's and the
//! language detector's contexts are cut at UTF-16 units and can hold half
//! of a surrogate pair, which no Rust string can. The outcome labels are
//! `JavaString`s for the same reason: a lemma class edits the units of a
//! word.

use std::path::Path;

use crate::escape::unescape_utf16;
use crate::format::Header;
use crate::read::{GoldenError, Lines, field_error, parse_f64, read_header_records, record, u32_field};
use turbo_maxent::JavaString;

/// The format version the first line names (`dump 1`).
pub const DUMP_VERSION: u32 = 1;

/// A whole dump.
#[derive(Clone, Debug, PartialEq)]
pub struct Dump {
    /// The header records, the same as the golden's at the same path. A
    /// dump never has an error record.
    pub header: Header,
    /// One per `eval` call, in the order the annotator made them.
    pub points: Vec<Point>,
}

/// Where in the input a decision's UTF-16 index lies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unit {
    /// At the start of a code point (`unit 0`).
    Start,
    /// Between the two halves of a surrogate pair (`unit 1`); the
    /// position is the byte offset of that pair's code point.
    Inside,
}

/// One outcome of the model and the probability `eval` gave it.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub label: JavaString,
    pub prob: f64,
}

/// One `eval` call.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    /// The UTF-8 byte offset in the input the decision is about, 0 for a
    /// decision about the whole document.
    pub position: u32,
    pub unit: Unit,
    /// The context strings in the order the model received them, as UTF-16
    /// units.
    pub contexts: Vec<Vec<u16>>,
    /// The feature value of each context, when the annotator called
    /// `eval(String[], float[])`; `None` for `eval(String[])`.
    pub values: Option<Vec<f64>>,
    /// Every outcome of the model, in its outcome index order.
    pub outcomes: Vec<Outcome>,
    /// For a sentence detector point whose best outcome is a split, what
    /// `isAcceptableBreak` returned; `None` everywhere else.
    pub acceptable: Option<bool>,
}

impl Point {
    /// The outcome `getBestOutcome` names: the first of the highest
    /// probability.
    pub fn best(&self) -> &Outcome {
        let mut best = &self.outcomes[0];
        for o in &self.outcomes[1..] {
            if o.prob > best.prob {
                best = o;
            }
        }
        best
    }

    /// The probability of the outcome with this label, if the model has
    /// it.
    pub fn prob(&self, label: &str) -> Option<f64> {
        self.outcomes.iter().find(|o| o.label == label).map(|o| o.prob)
    }

    /// The contexts as Rust strings, or `None` when one holds half of a
    /// surrogate pair.
    pub fn context_strings(&self) -> Option<Vec<String>> {
        self.contexts.iter().map(|c| String::from_utf16(c).ok()).collect()
    }
}

impl Dump {
    /// Reads and parses a dump file.
    pub fn read(path: impl AsRef<Path>) -> Result<Dump, GoldenError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| GoldenError::Io { path: path.into(), error })?;
        Dump::parse_bytes(&bytes).map_err(|e| GoldenError::File { path: path.into(), error: Box::new(e) })
    }

    /// Parses a dump's bytes.
    pub fn parse_bytes(bytes: &[u8]) -> Result<Dump, GoldenError> {
        let text = std::str::from_utf8(bytes).map_err(|e| GoldenError::NotUtf8 {
            line: 1 + bytes[..e.valid_up_to()].iter().filter(|&&b| b == b'\n').count(),
        })?;
        Dump::parse(text)
    }

    /// Parses a dump's text.
    pub fn parse(text: &str) -> Result<Dump, GoldenError> {
        if let Some(at) = text.find('\r') {
            return Err(GoldenError::CarriageReturn { line: 1 + text[..at].matches('\n').count() });
        }
        let Some(body) = text.strip_suffix('\n') else {
            return Err(GoldenError::NoFinalLineEnd { line: 1 + text.matches('\n').count() });
        };
        let mut lines = Lines::new(body);
        let first = lines.expect("\"dump 1\"")?;
        if first != format!("dump {DUMP_VERSION}") {
            return Err(GoldenError::DumpVersion { line: lines.line, found: first.into() });
        }
        let header = read_header_records(&mut lines)?;
        if let Some(e) = &header.error {
            return Err(GoldenError::Record {
                line: lines.line,
                expected: "a point record: a run that threw has no dump".into(),
                found: format!("error\t{}", e.class),
            });
        }
        let mut points = Vec::new();
        while let Some(l) = lines.next() {
            points.push(read_point(&mut lines, l, points.len())?);
        }
        Ok(Dump { header, points })
    }
}

fn read_point(lines: &mut Lines<'_>, head: &str, index: usize) -> Result<Point, GoldenError> {
    let line = lines.line;
    let [i, position, contexts, outcomes, unit, values] =
        record(line, head, "point", ["index", "position", "contexts", "outcomes", "unit", "none|given"])?;
    let i = u32_field(line, "point index", i)?;
    if i as usize != index {
        return Err(field_error(line, "point index", "the points numbered from 0 in order", &i.to_string()));
    }
    let position = u32_field(line, "point position", position)?;
    let n_contexts = u32_field(line, "point contexts", contexts)? as usize;
    let n_outcomes = u32_field(line, "point outcomes", outcomes)? as usize;
    if n_outcomes == 0 {
        return Err(field_error(line, "point outcomes", "at least one outcome", outcomes));
    }
    let unit = match unit {
        "0" => Unit::Start,
        "1" => Unit::Inside,
        u => return Err(field_error(line, "point unit", "0 or 1", u)),
    };
    let given = match values {
        "none" => false,
        "given" => true,
        v => return Err(field_error(line, "point values", "none or given", v)),
    };

    let mut point = Point {
        position,
        unit,
        contexts: Vec::new(),
        values: given.then(Vec::new),
        outcomes: Vec::new(),
        acceptable: None,
    };
    for k in 0..n_contexts {
        let l = lines.expect(&format!("context {k} of point {index}"))?;
        let line = lines.line;
        let text = if given {
            let [text, value] = record(line, l, "context", ["str", "f64"])?;
            let value = parse_f64(value).ok_or_else(|| {
                field_error(line, "context value", "16 lowercase hex digits of bits, a space, a decimal", value)
            })?;
            point.values.as_mut().expect("given values").push(value);
            text
        } else {
            let [text] = record(line, l, "context", ["str"])?;
            text
        };
        let units =
            unescape_utf16(text).map_err(|error| GoldenError::Escape { line, field: "context".into(), error })?;
        point.contexts.push(units);
    }
    for k in 0..n_outcomes {
        let l = lines.expect(&format!("outcome {k} of point {index}"))?;
        let line = lines.line;
        let [label, prob] = record(line, l, "outcome", ["label", "f64"])?;
        let prob = parse_f64(prob).ok_or_else(|| {
            field_error(line, "outcome probability", "16 lowercase hex digits of bits, a space, a decimal", prob)
        })?;
        let label = unescape_utf16(label).map_err(|error| GoldenError::Escape {
            line,
            field: "outcome label".into(),
            error,
        })?;
        point.outcomes.push(Outcome { label: JavaString::from_utf16(&label), prob });
    }
    let mut l = lines.expect(&format!("\"end\t{index}\""))?;
    if l.split('\t').next() == Some("acceptable") {
        let line = lines.line;
        let [a] = record(line, l, "acceptable", ["0|1"])?;
        point.acceptable = Some(match a {
            "0" => false,
            "1" => true,
            a => return Err(field_error(line, "acceptable", "0 or 1", a)),
        });
        l = lines.expect(&format!("\"end\t{index}\""))?;
    }
    let end = format!("end\t{index}");
    if l != end {
        return Err(GoldenError::Record {
            line: lines.line,
            expected: format!("\"end\\t{index}\" after {n_contexts} contexts and {n_outcomes} outcomes"),
            found: l.into(),
        });
    }
    Ok(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "dump 1\ngenerator\tg\t1\nopennlp\tcdf33c653d5d61d5af7d37ffee189a83bd8392ea\t3.0.0\n\
                          jdk\t21\t21+1\tv\nunicode\t15.0\ninput\ten/a\t3\t3\t\
                          0000000000000000000000000000000000000000000000000000000000000000\n\
                          model\tm.bin\t\
                          00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000\n\
                          annotator\tC\tm\n";

    fn parse(points: &str) -> Result<Dump, GoldenError> {
        Dump::parse(&format!("{HEADER}{points}"))
    }

    #[test]
    fn points_with_and_without_values_and_the_acceptable_record() {
        let d = parse(
            "point\t0\t2\t2\t2\t0\tnone\ncontext\tsn\ncontext\tx=\\ud83d\n\
             outcome\tn\t3fd0000000000000 0.25\noutcome\ts\t3fe8000000000000 0.75\nacceptable\t0\nend\t0\n\
             point\t1\t0\t1\t2\t1\tgiven\ncontext\tab\t3ff0000000000000 1.0\n\
             outcome\tT\t3fe0000000000000 0.5\noutcome\tF\t3fe0000000000000 0.5\nend\t1\n",
        )
        .unwrap();
        assert_eq!(d.header.models[0].file, "m.bin");
        assert_eq!(d.points.len(), 2);
        let p = &d.points[0];
        assert_eq!((p.position, p.unit, p.values.as_ref()), (2, Unit::Start, None));
        assert_eq!(p.contexts, vec!["sn".encode_utf16().collect::<Vec<_>>(), vec![0x78, 0x3d, 0xd83d]]);
        assert_eq!(p.context_strings(), None, "half of a pair is no Rust string");
        assert_eq!(p.best().label, "s");
        assert_eq!(p.prob("n"), Some(0.25));
        assert_eq!(p.acceptable, Some(false));
        let q = &d.points[1];
        assert_eq!((q.unit, q.values.as_deref()), (Unit::Inside, Some(&[1.0][..])));
        assert_eq!(q.context_strings(), Some(vec!["ab".to_string()]));
        // A tie goes to the first outcome, as getBestOutcome does.
        assert_eq!(q.best().label, "T");
        assert_eq!(q.acceptable, None);
    }

    #[test]
    fn an_outcome_label_with_half_a_surrogate_pair_is_the_jvms_string() {
        // A lemma class that deletes the two units of an emoji.
        let d = parse(
            "point\t0\t1\t1\t2\t0\tnone\ncontext\tw0=a\n\
             outcome\tO\t3fd0000000000000 0.25\noutcome\tD1\\ude00D0\\ud83d\t3fe8000000000000 0.75\nend\t0\n",
        )
        .unwrap();
        let p = &d.points[0];
        assert_eq!(p.best().label, JavaString::from_utf16(&[0x44, 0x31, 0xDE00, 0x44, 0x30, 0xD83D]));
        assert_eq!(p.best().label.to_str(), None);
        assert_eq!(p.prob("O"), Some(0.25));
    }

    #[test]
    fn a_header_alone_is_a_dump_without_points() {
        assert!(parse("").unwrap().points.is_empty());
    }

    #[test]
    fn malformed_dumps_are_refused_on_their_line() {
        let line = |e: GoldenError| match e {
            GoldenError::Record { line, .. }
            | GoldenError::Field { line, .. }
            | GoldenError::Fields { line, .. }
            | GoldenError::Eof { line, .. }
            | GoldenError::Escape { line, .. }
            | GoldenError::DumpVersion { line, .. } => line,
            e => panic!("{e:?}"),
        };
        let ok = "point\t0\t2\t1\t1\t0\tnone\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nend\t0\n";
        assert!(parse(ok).is_ok());
        // The point record is line 9.
        for (bad, at) in [
            ("point\t1\t2\t1\t1\t0\tnone\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nend\t1\n", 9),
            ("point\t0\t2\t1\t1\t2\tnone\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nend\t0\n", 9),
            ("point\t0\t2\t1\t1\t0\tsome\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nend\t0\n", 9),
            ("point\t0\t2\t1\t0\t0\tnone\ncontext\ta\nend\t0\n", 9),
            ("point\t0\t2\t1\t1\t0\tnone\ncontext\ta\tb\noutcome\tn\t3ff0000000000000 1.0\nend\t0\n", 10),
            ("point\t0\t2\t1\t1\t0\tnone\ncontext\t\\q\noutcome\tn\t3ff0000000000000 1.0\nend\t0\n", 10),
            ("point\t0\t2\t1\t1\t0\tnone\ncontext\ta\noutcome\tn\t1.0\nend\t0\n", 11),
            ("point\t0\t2\t1\t1\t0\tnone\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nacceptable\t2\nend\t0\n", 12),
            ("point\t0\t2\t1\t1\t0\tnone\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nend\t1\n", 12),
            ("point\t0\t2\t2\t1\t0\tnone\ncontext\ta\noutcome\tn\t3ff0000000000000 1.0\nend\t0\n", 11),
            ("point\t0\t2\t1\t1\t0\tnone\ncontext\ta\n", 11),
        ] {
            assert_eq!(line(parse(bad).unwrap_err()), at, "{bad:?}");
        }
        let e = Dump::parse(&HEADER.replacen("dump 1", "golden 1", 1)).unwrap_err();
        assert_eq!(line(e), 1);
        let threw = HEADER.to_string() + "error\tjava.lang.X\tm\n";
        assert!(matches!(Dump::parse(&threw), Err(GoldenError::Record { .. })));
    }
}
