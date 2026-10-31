//! Liblinear text serialization and the retired OpenNLP binary envelope writer.
//!
//! The binary envelope reader returns a typed disabled error before decoding.
//! The independent liblinear text parser remains available until the training
//! stack migration in #1532. Text round trips are checked against native numeric
//! values, without a JVM oracle.

use std::fmt::Write as _;

use super::SolverType;
use super::error::ModelError;
use super::linear::LinearModel;
use super::model::LiblinearModel;
use crate::error::{Error, Result};
use crate::java_string::JavaString;
use crate::text::{parse_double, parse_int};

/// The most bytes `DataOutputStream.writeUTF` writes.
const MAX_UTF_BYTES: usize = 65535;

/// `LiblinearModel.serialize`: the model's bytes. Fails as `writeUTF`
/// does, on an outcome label or a predicate over 65535 bytes of modified
/// UTF-8.
pub fn write(model: &LiblinearModel) -> Result<Vec<u8>> {
    let text = save_model(&model.linear);
    let mut out = Vec::with_capacity(text.len() + 64);
    write_int(&mut out, text.len());
    out.extend_from_slice(text.as_bytes());
    write_int(&mut out, model.outcomes.len());
    for o in &model.outcomes {
        write_utf(&mut out, o, "outcome label")?;
    }
    write_int(&mut out, model.predicates.len());
    for (p, id) in &model.predicates {
        write_utf(&mut out, p, "predicate")?;
        out.extend_from_slice(&id.to_be_bytes());
    }
    Ok(out)
}

fn write_int(out: &mut Vec<u8>, n: usize) {
    let n = i32::try_from(n).expect("a Java array holds at most i32::MAX entries");
    out.extend_from_slice(&n.to_be_bytes());
}

fn write_utf(out: &mut Vec<u8>, s: &JavaString, what: &'static str) -> Result<()> {
    let bytes = s.as_bytes();
    if bytes.len() > MAX_UTF_BYTES {
        return Err(Error::StringTooLong { what, bytes: bytes.len() });
    }
    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// `Linear.saveModel`: the text model.
pub fn save_model(m: &LinearModel) -> String {
    let nr_w = m.nr_w();
    let mut s = String::with_capacity(64 + m.w.len() * 24);
    let _ = write!(s, "solver_type {}\nnr_class {}\nlabel", m.solver.name(), m.label.len());
    for l in &m.label {
        let _ = write!(s, " {l}");
    }
    let _ = write!(s, "\nnr_feature {}\nbias ", m.nr_feature);
    push_g17(&mut s, m.bias);
    s.push_str("\nw\n");
    for row in m.w.chunks(nr_w.max(1)) {
        for &v in row {
            if v == 0.0 {
                s.push('0');
            } else {
                push_g17(&mut s, v);
            }
            s.push(' ');
        }
        s.push('\n');
    }
    s
}

/// Java's `String.format(Locale.ENGLISH, "%.17g", x)` for a finite `x`: the
/// digits `FloatingDecimal` finds for `x` padded with zeros to 17
/// significant digits, in plain notation when the decimal exponent is from
/// -4 to 16 and in scientific notation (`d.dddde+XX`, the exponent with
/// its sign and at least two digits) otherwise.
///
/// `FloatingDecimal`'s digits are the shortest that read back to `x`, as
/// Rust's are, except where `x` lies exactly halfway between two shortest
/// candidates that both read back to `x`: Java then takes the one with the
/// even last digit, and Rust may take the other.
pub(crate) fn push_g17(out: &mut String, x: f64) {
    const PRECISION: usize = 17;
    if x.is_sign_negative() {
        out.push('-');
    }
    if x == 0.0 {
        out.push_str("0.0000000000000000");
        return;
    }
    let (digits, exponent) = java_digits(x.abs());
    let digits = digits.as_bytes();
    let digit = |i: usize| digits.get(i).map_or('0', |&d| d as char);
    if !(-4..PRECISION as i32).contains(&exponent) {
        out.push(digit(0));
        out.push('.');
        for i in 1..PRECISION {
            out.push(digit(i));
        }
        let _ = write!(out, "e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs());
    } else if exponent >= 0 {
        let whole = exponent as usize + 1;
        for i in 0..whole {
            out.push(digit(i));
        }
        if whole < PRECISION {
            out.push('.');
            for i in whole..PRECISION {
                out.push(digit(i));
            }
        }
    } else {
        out.push_str("0.");
        for _ in 0..(-exponent - 1) {
            out.push('0');
        }
        for i in 0..PRECISION {
            out.push(digit(i));
        }
    }
}

/// The significant digits `FloatingDecimal` gives a positive finite `x`,
/// without trailing zeros, and the decimal exponent of the first:
/// `x = d.ddd * 10^exponent`.
fn java_digits(x: f64) -> (String, i32) {
    // FloatingDecimal gives these eight smallest subnormals two digits
    // where one reads back. Every other double whose bits are at most
    // 300000 gets the shortest digits, as compared with JDK 21.
    match x.to_bits() {
        1 => return ("49".into(), -324),
        2 => return ("99".into(), -324),
        bits @ (10 | 12 | 14 | 16 | 18) => return (format!("{}9", bits / 2 - 1), -323),
        20 => return ("99".into(), -323),
        _ => {}
    }
    // Rust's `{:e}` writes the shortest digits that read back: `d.ddde-5`.
    let sci = format!("{x:e}");
    let (mantissa, exponent) = sci.split_once('e').expect("{:e} writes an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} writes a decimal exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let k = digits.len();
    // A tie needs x to be the k + 1 digit decimal halfway between the
    // shortest candidates: correctly rounded to k + 1 digits it ends in 5,
    // and nothing follows in its exact expansion (767 digits hold any
    // double's).
    let near = format!("{x:.k$e}");
    let (near_mantissa, near_exponent) = near.split_once('e').expect("{:e} writes an exponent");
    if !near_mantissa.ends_with('5') || near_exponent.parse::<i32>() != Ok(exponent) {
        return (digits, exponent);
    }
    let exact = format!("{x:.767e}");
    let exact_digits: Vec<u8> =
        exact.split_once('e').expect("{:e} writes an exponent").0.bytes().filter(u8::is_ascii_digit).collect();
    if exact_digits[k] != b'5' || exact_digits[k + 1..].iter().any(|&d| d != b'0') {
        return (digits, exponent);
    }
    let last = digits.as_bytes()[k - 1] - b'0';
    if last.is_multiple_of(2) {
        return (digits, exponent);
    }
    // The other candidate is one unit away on the far side of x.
    let shortest: u64 = digits.parse().expect("at most 17 digits");
    let truncated: u64 = std::str::from_utf8(&exact_digits[..k]).expect("digits").parse().expect("at most 17 digits");
    let other = if truncated == shortest { shortest + 1 } else { shortest - 1 };
    let mut other_digits = other.to_string();
    let mut other_exponent = exponent;
    if other_digits.len() > k {
        other_exponent += 1;
    } else if other_digits.len() < k {
        other_exponent -= 1;
    }
    while other_digits.len() > 1 && other_digits.ends_with('0') {
        other_digits.pop();
    }
    // Below a power of two the doubles are closer together, so the other
    // candidate may not read back to x; it is then no candidate.
    let reads_back = format!("{}.{}e{other_exponent}", &other_digits[..1], &other_digits[1..]).parse::<f64>() == Ok(x);
    if !reads_back {
        return (digits, exponent);
    }
    (other_digits, other_exponent)
}

/// OpenNLP binary-envelope loading is disabled under #1511.
/// No input bytes are decoded, including otherwise valid serialized models.
pub fn read(_bytes: &[u8]) -> std::result::Result<LiblinearModel, ModelError> {
    Err(ModelError::Disabled)
}

fn line_error(line: usize, detail: impl Into<String>) -> ModelError {
    ModelError::Line { line, detail: detail.into() }
}

/// Whether `text` holds a character `Double.parseDouble` would trim:
/// `saveModel` separates its fields with single spaces and nothing else.
fn padded(text: &str) -> bool {
    text.bytes().any(|b| b <= b' ')
}

/// `Linear.atoi` without its leading `+`, which `saveModel` never writes.
fn int(line: usize, field: &str, text: &str) -> std::result::Result<i32, ModelError> {
    if padded(text) {
        return Err(line_error(line, format!("{field}: {text:?} is not an int")));
    }
    parse_int(text).map_err(|_| line_error(line, format!("{field}: {text:?} is not an int")))
}

/// `Linear.atof`: a double that is neither NaN nor infinite.
fn double(line: usize, field: &str, text: &str) -> std::result::Result<f64, ModelError> {
    match parse_double(text) {
        Ok(d) if d.is_finite() && !padded(text) => Ok(d),
        _ => Err(line_error(line, format!("{field}: {text:?} is not a finite double"))),
    }
}

/// `Linear.loadModel`, strictly: the header in `saveModel`'s order, then
/// the weights.
pub fn load_model(text: &str) -> std::result::Result<LinearModel, ModelError> {
    let Some(body) = text.strip_suffix('\n') else {
        return Err(ModelError::Invalid("the text model does not end its last line".into()));
    };
    let mut lines = body.split('\n').enumerate().map(|(i, l)| (i + 1, l));
    let mut next = |keyword: &'static str| -> std::result::Result<(usize, &str), ModelError> {
        let Some((n, line)) = lines.next() else {
            return Err(ModelError::Invalid(format!("the text model ends before {keyword}")));
        };
        if line == keyword {
            return Ok((n, ""));
        }
        match line.strip_prefix(keyword).and_then(|r| r.strip_prefix(' ')) {
            Some(rest) => Ok((n, rest)),
            None => Err(line_error(n, format!("expected {keyword}, found {line:?}"))),
        }
    };

    let (n, name) = next("solver_type")?;
    let solver = SolverType::from_name(name).ok_or_else(|| line_error(n, format!("no solver type {name:?}")))?;
    if !solver.is_logistic_regression() {
        return Err(ModelError::Invalid(format!(
            "a {} model, which the add-on cannot score (only logistic regression models give probabilities)",
            solver.name()
        )));
    }
    let (n, nr_class) = next("nr_class")?;
    let nr_class = int(n, "nr_class", nr_class)?;
    if nr_class < 1 {
        return Err(line_error(n, format!("nr_class is {nr_class}")));
    }
    let (n, labels) = next("label")?;
    let label = labels.split(' ').map(|t| int(n, "label", t)).collect::<std::result::Result<Vec<_>, _>>()?;
    if labels.is_empty() || label.len() != nr_class as usize {
        return Err(line_error(n, format!("{} labels for {nr_class} classes", label.len())));
    }
    let (n, nr_feature) = next("nr_feature")?;
    let nr_feature = int(n, "nr_feature", nr_feature)?;
    let Ok(nr_feature) = usize::try_from(nr_feature) else {
        return Err(line_error(n, format!("nr_feature is {nr_feature}")));
    };
    let (n, bias) = next("bias")?;
    let bias = double(n, "bias", bias)?;
    let (n, rest) = next("w")?;
    if !rest.is_empty() {
        return Err(line_error(n, "text after w"));
    }

    let mut model = LinearModel { solver, label, nr_feature, bias, w: Vec::new() };
    let (w_size, nr_w) = (model.w_size(), model.nr_w());
    let mut rows = 0usize;
    for (n, line) in lines {
        if rows == w_size {
            return Err(line_error(n, "a line after the last feature's weights"));
        }
        let Some(values) = line.strip_suffix(' ') else {
            return Err(line_error(n, "a weight not followed by a space"));
        };
        let before = model.w.len();
        for t in values.split(' ') {
            model.w.push(double(n, "weight", t)?);
        }
        if model.w.len() - before != nr_w {
            return Err(line_error(n, format!("{} weights, {nr_w} expected", model.w.len() - before)));
        }
        rows += 1;
    }
    if rows != w_size {
        return Err(ModelError::Invalid(format!("weights for {rows} features, {w_size} expected")));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g17(x: f64) -> String {
        let mut s = String::new();
        push_g17(&mut s, x);
        s
    }

    #[test]
    fn text_numbers_preserve_finite_native_values() {
        for value in [
            0.0_f64,
            -0.0,
            1.0,
            -1.0,
            0.1,
            1e-5,
            1e17,
            123.456,
            -2.5e-300,
            f64::MAX,
            f64::MIN_POSITIVE,
            5.960464477539063e-8,
        ] {
            assert_eq!(g17(value).parse::<f64>().unwrap().to_bits(), value.to_bits());
        }
        for bits in 1..=300_000 {
            let value = f64::from_bits(bits);
            assert_eq!(g17(value).parse::<f64>().unwrap().to_bits(), bits);
        }
    }
}
