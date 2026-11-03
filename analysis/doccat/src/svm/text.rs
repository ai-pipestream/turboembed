//! Line reading and number formatting shared by the two SVM model files.

/// The lines of `text` numbered from 1, without their `\n` or `\r\n`. A
/// final line end does not start another line.
pub(crate) fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let body = text.strip_suffix('\n').unwrap_or(text);
    let empty = text.is_empty();
    body.split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(move |_| !empty)
}

/// Formats `x` in `Double.toString` layout: plain decimal when the magnitude
/// is in `[1e-3, 1e7)`, scientific notation with `E` otherwise.
pub(crate) fn double_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (digits, exponent) =
        shortest(format!("{:e}", x), |s| s.parse::<f64>().ok() == Some(x), || format!("{:.800e}", x));
    layout(x.is_sign_negative(), &digits, exponent, (1e-3..1e7).contains(&x.abs()))
}

fn shortest(short: String, rounds_back: impl Fn(&str) -> bool, exact: impl Fn() -> String) -> (Vec<u8>, i32) {
    let (digits, exponent) = split_e(&short);
    let n = digits.len().max(2);
    let (all, mut e) = split_e(&exact());
    let mut kept: Vec<u8> = all[..n.min(all.len())].to_vec();
    kept.resize(n, 0);
    let rest = &all[n.min(all.len())..];
    let up = match rest.first() {
        None => false,
        Some(&d) if d > 5 => true,
        Some(&d) if d < 5 => false,
        Some(_) => rest[1..].iter().any(|&d| d != 0) || kept[n - 1] % 2 == 1,
    };
    if up {
        let mut i = n;
        loop {
            if i == 0 {
                kept.insert(0, 1);
                kept.pop();
                e += 1;
                break;
            }
            i -= 1;
            if kept[i] == 9 {
                kept[i] = 0;
            } else {
                kept[i] += 1;
                break;
            }
        }
    }
    while kept.len() > 1 && kept.last() == Some(&0) {
        kept.pop();
    }
    let mut text = String::new();
    if short.starts_with('-') {
        text.push('-');
    }
    text.push((b'0' + kept[0]) as char);
    text.push('.');
    if kept.len() > 1 {
        kept[1..].iter().for_each(|&d| text.push((b'0' + d) as char));
    } else {
        text.push('0');
    }
    text.push_str(&format!("e{e}"));
    if !rounds_back(&text) {
        return (digits, exponent);
    }
    (kept, e)
}

fn split_e(text: &str) -> (Vec<u8>, i32) {
    let text = text.strip_prefix('-').unwrap_or(text);
    let (mantissa, exponent) = text.split_once('e').expect("{:e} output has an exponent");
    let digits = mantissa.bytes().filter(u8::is_ascii_digit).map(|b| b - b'0').collect();
    (digits, exponent.parse().expect("{:e} output has an integer exponent"))
}

fn layout(negative: bool, digits: &[u8], exponent: i32, plain: bool) -> String {
    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }
    let digit = |d: u8| (b'0' + d) as char;
    if plain {
        if exponent >= 0 {
            let int_len = exponent as usize + 1;
            for i in 0..int_len {
                out.push(digit(digits.get(i).copied().unwrap_or(0)));
            }
            out.push('.');
            if digits.len() > int_len {
                digits[int_len..].iter().for_each(|&d| out.push(digit(d)));
            } else {
                out.push('0');
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            digits.iter().for_each(|&d| out.push(digit(d)));
        }
    } else {
        out.push(digit(digits[0]));
        out.push('.');
        if digits.len() > 1 {
            digits[1..].iter().for_each(|&d| out.push(digit(d)));
        } else {
            out.push('0');
        }
        out.push('E');
        out.push_str(&exponent.to_string());
    }
    out
}

/// Parses an `f64` in `Double.parseDouble` decimal grammar.
pub(crate) fn parse_double(s: &str) -> Result<f64, ()> {
    let trimmed = s.trim_matches(|c: char| c <= ' ');
    if trimmed.is_empty() {
        return Err(());
    }
    let (negative, rest) = match trimmed.as_bytes()[0] {
        b'-' => (true, &trimmed[1..]),
        b'+' => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    let value: f64 = match rest {
        "NaN" => f64::NAN,
        "Infinity" => f64::INFINITY,
        _ => {
            let bytes = rest.as_bytes();
            let body_end = match bytes.last() {
                Some(b'f' | b'F' | b'd' | b'D') => bytes.len() - 1,
                _ => bytes.len(),
            };
            let body = &rest[..body_end];
            let b = body.as_bytes();
            let mut i = 0;
            let mut mantissa_digits = 0;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
                mantissa_digits += 1;
            }
            if i < b.len() && b[i] == b'.' {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                    mantissa_digits += 1;
                }
            }
            if mantissa_digits == 0 {
                return Err(());
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                let start = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i == start {
                    return Err(());
                }
            }
            if i != b.len() {
                return Err(());
            }
            body.parse().map_err(|_| ())?
        }
    };
    Ok(if negative { -value } else { value })
}

/// Parses an `i32` from an optional sign and ASCII decimal digits.
pub(crate) fn parse_int(s: &str) -> Result<i32, ()> {
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        Some(_) => (false, s),
        None => return Err(()),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    let mut value: i64 = 0;
    for b in digits.bytes() {
        value = value * 10 + i64::from(b - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return Err(());
        }
    }
    i32::try_from(if negative { -value } else { value }).map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_numbered_without_their_ends() {
        assert_eq!(lines("a\nb\r\nc\n").collect::<Vec<_>>(), [(1, "a"), (2, "b"), (3, "c")]);
        assert_eq!(lines("a\n\nb").collect::<Vec<_>>(), [(1, "a"), (2, ""), (3, "b")]);
        assert_eq!(lines("").count(), 0);
        assert_eq!(lines("\n").collect::<Vec<_>>(), [(1, "")]);
    }
}
