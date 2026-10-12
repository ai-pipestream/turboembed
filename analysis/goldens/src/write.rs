//! The text form of a golden, byte for byte as the generator writes it:
//! UTF-8, LF line ends, one tab between fields, a line end after the last
//! line. An `f64` is its 16 hex digits of bits, a space, and the decimal
//! Java's `Double.toString` gives, which `java_double` reproduces, so a
//! file Rust writes is the file the JVM would.

use std::io::{self, Write};

use crate::escape::escape_into;
use crate::format::{ColumnData, Golden, GoldenAnchor, GoldenLayer, Header, VERSION};
use crate::hex::to_hex;

impl Golden {
    /// Writes the file. Refuses (as `InvalidInput`) what the reader would
    /// refuse or the format cannot carry: an empty header field, aligned
    /// anchor or column name, one holding a tab or a line end, unsorted
    /// config keys, a layer with no columns or with columns of different
    /// lengths, and layers beside an error record.
    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let mut out = String::new();
        self.write_to_string(&mut out)?;
        w.write_all(out.as_bytes())
    }

    /// The file's text.
    pub fn to_text(&self) -> io::Result<String> {
        let mut out = String::new();
        self.write_to_string(&mut out)?;
        Ok(out)
    }

    fn write_to_string(&self, out: &mut String) -> io::Result<()> {
        if let Some(e) = &self.header.error
            && !self.layers.is_empty()
        {
            return Err(invalid(format!("a run that threw {} has layers; such a file has none", e.class)));
        }
        write_header(&self.header, out)?;
        for layer in &self.layers {
            write_layer(layer, out)?;
        }
        Ok(())
    }
}

fn write_header(h: &Header, out: &mut String) -> io::Result<()> {
    out.push_str(&format!("golden {VERSION}\n"));
    record(out, &["generator", &h.generator.name, &h.generator.version.to_string()])?;
    record(out, &["opennlp", &to_hex(&h.opennlp.commit), &h.opennlp.version])?;
    record(out, &["jdk", &h.jdk.version, &h.jdk.vm_version, &h.jdk.vendor])?;
    record(out, &["unicode", &h.unicode.to_string()])?;
    let i = &h.input;
    // Tabs and line ends in the set and id are refused with the record.
    if i.set.is_empty() || i.id.is_empty() {
        return Err(invalid(format!("the input record's set {:?} or id {:?} is empty", i.set, i.id)));
    }
    if i.set.contains('/') {
        return Err(invalid(format!("the input set {:?} holds a '/'", i.set)));
    }
    record(
        out,
        &[
            "input",
            &format!("{}/{}", i.set, i.id),
            &i.utf8_len.to_string(),
            &i.utf16_len.to_string(),
            &to_hex(&i.sha256),
        ],
    )?;
    for m in &h.models {
        record(out, &["model", &m.file, &to_hex(&m.sha512)])?;
    }
    record(out, &["annotator", &h.annotator.class, &h.annotator.method])?;
    if let Some(w) = h.config.windows(2).find(|w| w[0].key >= w[1].key) {
        return Err(invalid(format!(
            "the config key {:?} comes after {:?}: keys are sorted, each once",
            w[1].key, w[0].key
        )));
    }
    // Key and value are str fields, escaped like a row's.
    for c in &h.config {
        out.push_str("config\t");
        escape_into(&c.key, out);
        out.push('\t');
        escape_into(&c.value, out);
        out.push('\n');
    }
    if let Some(e) = &h.error {
        field("the error record's class", &e.class)?;
        out.push_str("error\t");
        out.push_str(&e.class);
        out.push('\t');
        escape_into(&e.message, out);
        out.push('\n');
    }
    Ok(())
}

/// A field written as it is, unescaped: a header field, an aligned
/// anchor's layer id, a column name. The reader refuses one that is
/// empty, and one holding a tab or a line end would break its record, so
/// the writer refuses both.
fn field(what: &str, f: &str) -> io::Result<()> {
    if f.is_empty() {
        return Err(invalid(format!("{what} is empty")));
    }
    if f.contains(['\t', '\n', '\r']) {
        return Err(invalid(format!("{what} {f:?} holds a tab or a line end")));
    }
    Ok(())
}

/// One header record, every field checked by `field`.
fn record(out: &mut String, fields: &[&str]) -> io::Result<()> {
    for (i, f) in fields.iter().enumerate() {
        field(&format!("the {} record's field", fields[0]), f)?;
        if i > 0 {
            out.push('\t');
        }
        out.push_str(f);
    }
    out.push('\n');
    Ok(())
}

fn write_layer(l: &GoldenLayer, out: &mut String) -> io::Result<()> {
    if l.id.contains(['\t', '\n', '\r']) {
        return Err(invalid(format!("the layer id {:?} holds a tab or a line end", l.id)));
    }
    if l.columns.is_empty() {
        return Err(invalid(format!("the layer {} has no columns", l.id)));
    }
    let rows = l.len();
    if let Some(c) = l.columns.iter().find(|c| c.data.len() != rows) {
        return Err(invalid(format!(
            "the layer {} has {rows} rows in its first column and {} in {}",
            l.id,
            c.data.len(),
            c.name
        )));
    }
    out.push_str("layer\t");
    out.push_str(&l.id);
    out.push('\t');
    if let GoldenAnchor::Aligned(of) = &l.anchor {
        field(&format!("the aligned anchor of {}", l.id), of)?;
    }
    out.push_str(&l.anchor.to_string());
    out.push('\t');
    out.push_str(&rows.to_string());
    for c in &l.columns {
        field("the column name", &c.name)?;
        if c.name.contains(':') {
            return Err(invalid(format!("the column name {:?} holds a ':'", c.name)));
        }
        out.push('\t');
        out.push_str(&c.name);
        out.push(':');
        out.push_str(c.data.ty().name());
    }
    out.push('\n');
    for r in 0..rows {
        for (i, c) in l.columns.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            match &c.data {
                ColumnData::U32(v) => out.push_str(&v[r].to_string()),
                ColumnData::F64(v) => push_f64(out, v[r]),
                ColumnData::Str(v) => escape_into(v.get(r), out),
            }
        }
        out.push('\n');
    }
    out.push_str("end\t");
    out.push_str(&l.id);
    out.push('\n');
    Ok(())
}

/// An `f64` field: the raw bits as 16 lowercase hex digits, a space, and
/// Java's decimal.
pub fn push_f64(out: &mut String, x: f64) {
    out.push_str(&to_hex(&x.to_bits().to_be_bytes()));
    out.push(' ');
    out.push_str(&java_double(x));
}

fn invalid(detail: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}

/// `Double.toString(x)` as JDK 19 and later define it (the generator runs
/// on JDK 21).
///
/// Java picks the decimal with the fewest digits that rounds back to `x`,
/// and of those the one nearest `x`, the one with an even last digit on a
/// tie. When one digit would do, it picks the nearest decimal of one or
/// two digits, so `Double.MIN_VALUE` is `4.9E-324`, not `5.0E-324`. Rust's
/// shortest formatting gives the fewest digits, but breaks a tie
/// differently (`2.1674000000000003E15` where Java has `...02E15`), so
/// only its length is used.
///
/// It lays the digits out plainly when `10^-3 <= |d| < 10^7` (`0.001`,
/// `1234567.0`), and as `d.dddE<exp>` otherwise (`1.0E7`, `9.99E-4`),
/// with at least one digit after the point either way. Zero is `0.0` or
/// `-0.0`, and the others are `NaN`, `Infinity` and `-Infinity`.
pub fn java_double(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let mut out = String::new();
    if x.is_sign_negative() {
        out.push('-');
    }
    if x == 0.0 {
        out.push_str("0.0");
        return out;
    }
    let a = x.abs();
    let (shortest, shortest_exp) = scientific(&format!("{a:e}"));
    let (digits, exp) = nearest(a, shortest.len().max(2)).unwrap_or((shortest, shortest_exp));
    let d = digits.as_bytes();
    if (-3..7).contains(&exp) {
        if exp >= 0 {
            let int = exp as usize + 1;
            for i in 0..int {
                out.push(*d.get(i).unwrap_or(&b'0') as char);
            }
            out.push('.');
            if d.len() > int { out.push_str(&digits[int..]) } else { out.push('0') }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        }
    } else {
        out.push(d[0] as char);
        out.push('.');
        if d.len() > 1 {
            out.push_str(&digits[1..])
        } else {
            out.push('0')
        }
        out.push('E');
        out.push_str(&exp.to_string());
    }
    out
}

/// Splits Rust's `{:e}` form (`1.2345e-7`) into its significant digits,
/// without trailing zeros (`12345`), and its decimal exponent (`-7`).
fn scientific(s: &str) -> (String, i32) {
    let (mantissa, exp) = s.split_once('e').expect("an exponent in {:e}");
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    (digits, exp.parse().expect("a decimal exponent"))
}

/// Of the decimals with `n` significant digits that round back to `a`,
/// the one nearest `a`, ties to an even last digit; `None` when no decimal
/// of `n` digits rounds back. Trailing zeros are dropped from the digits.
///
/// The nearest `n`-digit decimal to `a` is `a` rounded to `n` digits, which
/// Rust computes exactly, ties to even, as Java breaks them. When it rounds
/// back to `a` it is the answer. When it does not, the decimals that do
/// round back form an interval around `a` that stops short of it, so they
/// all lie on the other side of `a`, and the nearest of them is the next
/// `n`-digit decimal on that side.
fn nearest(a: f64, n: usize) -> Option<(String, i32)> {
    let (digits, exp) = scientific(&format!("{a:.*e}", n - 1));
    if parse(&digits, exp) == a {
        return Some((digits, exp));
    }
    // The n digits as an integer from 10^(n-1) to 10^n - 1, at exponent
    // exp - (n - 1). n is at most 17, the most a double ever needs.
    let low = 10u64.pow(n as u32 - 1);
    let mut m: u64 = digits.parse::<u64>().expect("digits") * 10u64.pow((n - digits.len()) as u32);
    let mut e = exp;
    if parse(&digits, exp) < a {
        m += 1;
        if m == low * 10 {
            m = low;
            e += 1;
        }
    } else {
        m -= 1;
        if m < low {
            m = low * 10 - 1;
            e -= 1;
        }
    }
    let mut digits = m.to_string();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    (parse(&digits, e) == a).then_some((digits, e))
}

/// The double nearest `0.d1d2... * 10^(exp + 1)`, that is `d1.d2... * 10^exp`.
fn parse(digits: &str, exp: i32) -> f64 {
    format!("{}.{}e{exp}", &digits[..1], if digits.len() > 1 { &digits[1..] } else { "0" }).parse().expect("a decimal")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each pair is a value and what `Double.toString` printed for it on
    /// OpenJDK 21.
    #[test]
    fn matches_java() {
        let cases: &[(u64, &str)] = &[
            (0x3fe0000000000000, "0.5"),
            (0x3ff0000000000000, "1.0"),
            (0x416312d000000000, "1.0E7"),
            (0x416312cfe0000000, "9999999.0"),
            (0x3f50624dd2f1a9fc, "0.001"),
            (0x3f505e1c15097c81, "9.99E-4"),
            (0x8000000000000000, "-0.0"),
            (0x0000000000000000, "0.0"),
            (0x0000000000000001, "4.9E-324"),
            (0x0000000000000002, "9.9E-324"),
            (0x0000000000000003, "1.5E-323"),
            (0x7fefffffffffffff, "1.7976931348623157E308"),
            (0x3fd3333333333334, "0.30000000000000004"),
            (0x44c52d02c7e14af6, "2.0E23"),
            (0x444b1ae4d6e2ef50, "1.0E21"),
            (0x40fe240c9fbe76c9, "123456.789"),
            (0x3ee4f8b588e368f1, "1.0E-5"),
            (0x7ff8000000000000, "NaN"),
            (0x7ff0000000000000, "Infinity"),
            (0xfff0000000000000, "-Infinity"),
            (0xbfe0000000000000, "-0.5"),
            // Ties between two shortest decimals, which Java breaks to even.
            (0x431eccf48c584001, "2.1674000000000002E15"),
            (0x420a70ef1f0d9000, "1.4195483617695312E10"),
            (0xc27be6bc3d487280, "-1.9173634100551562E12"),
        ];
        for &(bits, java) in cases {
            assert_eq!(java_double(f64::from_bits(bits)), java, "{bits:016x}");
        }
    }

    #[test]
    fn an_f64_field() {
        let mut s = String::new();
        push_f64(&mut s, 0.5);
        assert_eq!(s, "3fe0000000000000 0.5");
    }
}
