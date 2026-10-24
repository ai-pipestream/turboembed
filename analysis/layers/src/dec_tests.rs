use super::*;

fn d(s: &str) -> Dec {
    let neg = s.starts_with('-');
    let s = s.trim_start_matches('-');
    let (i, f) = s.split_once('.').unwrap_or((s, ""));
    let v = Dec::from_digits(i.as_bytes(), f.as_bytes());
    if neg { v.negated() } else { v }
}

#[test]
fn values_are_normalized() {
    assert_eq!(d("5.00"), d("5"));
    assert_eq!(d("4.800").to_string(), "4.8");
    assert_eq!(d("0000"), Dec::ZERO);
    assert_eq!(d("1000"), Dec::new(1, -3).unwrap());
    assert_eq!(d(".23456000").to_string(), "0.23456");
    assert_eq!(d("0.08").to_string(), "0.08");
    assert!(d("5.00").is_integer());
    assert!(!d("4.8").is_integer());
}

#[test]
fn long_numerals_round_past_the_kept_digits() {
    let forty = "1".repeat(40);
    let v = d(&forty);
    assert_eq!(v.to_string(), format!("{}0000", "1".repeat(36)));
    let v = d(&format!("0.{}", "6".repeat(40)));
    assert_eq!(v.to_string(), format!("0.{}7", "6".repeat(35)));
}

#[test]
fn arithmetic_is_exact_in_range() {
    assert_eq!(d("0.1").checked_add(d("0.2")).unwrap(), d("0.3"));
    assert_eq!(d("200").checked_add(d("0.03")).unwrap().to_string(), "200.03");
    assert_eq!(d("1.5").checked_mul(d("1000000")).unwrap(), d("1500000"));
    assert_eq!(d("-1").checked_mul(d("4")).unwrap(), d("-4"));
    assert_eq!(d("2").checked_sub(d("5")).unwrap(), d("-3"));
    assert_eq!(d("1.1").pow(23).unwrap().to_string(), "8.95430243255237372246531");
    assert_eq!(d("2.5").pow(-1).unwrap(), d("0.4"));
    assert_eq!(d("1.3").shift(6).unwrap(), d("1300000"));
    assert_eq!(d("6.23").shift(-4).unwrap().to_string(), "0.000623");
}

#[test]
fn quotients_keep_the_precision() {
    let third = Dec::ONE.checked_div(d("3")).unwrap();
    assert_eq!(third.to_string(), format!("0.{}", "3".repeat(34)));
    assert_eq!(d("2").checked_div(d("3")).unwrap().to_string(), format!("0.{}7", "6".repeat(33)));
    assert_eq!(d("1").checked_div(d("8")).unwrap(), d("0.125"));
    assert_eq!(d("-4").checked_div(d("5")).unwrap(), d("-0.8"));
    assert_eq!(Dec::ONE.checked_div(Dec::ZERO), None);
    assert_eq!(Dec::ZERO.checked_div(d("7")), Some(Dec::ZERO));
}

#[test]
fn large_operands_are_rounded_not_overflowed() {
    let big = d(&"9".repeat(36));
    let sum = big.checked_add(d("0.000000001")).unwrap();
    assert_eq!(sum, big);
    let p = big.checked_mul(big).unwrap();
    assert!(p.to_f64() > 9.99e71);
    let tiny = d("1").shift(-200).unwrap();
    assert_eq!(d("1").checked_add(tiny).unwrap(), d("1"));
    assert!(d("10").pow(5000).is_none());
}

#[test]
fn ordering_compares_values() {
    let mut v = [d("2"), d("-3.5"), d("0.25"), d("100"), d("-0.1"), Dec::ZERO, d("1")];
    v.sort();
    let s: Vec<String> = v.iter().map(Dec::to_string).collect();
    assert_eq!(s, ["-3.5", "-0.1", "0", "0.25", "1", "2", "100"]);
    assert!(d("999") < d("1000"));
    assert!(d("0.999") < d("1"));
    assert!(d("-1000") < d("-999"));
}

#[test]
fn integers_convert() {
    assert_eq!(d("21000000000322").to_i64(), Some(21_000_000_000_322));
    assert_eq!(d("1.5").to_i64(), None);
    assert_eq!(Dec::new(1, -30).unwrap().to_i64(), None);
    assert_eq!(Dec::ZERO.to_i64(), Some(0));
}

#[test]
fn coefficients_expand_negative_scales() {
    let mut s = String::new();
    assert_eq!(Dec::new(12, -3).unwrap().write_coefficient(&mut s), 0);
    assert_eq!(s, "12000");
    s.clear();
    assert_eq!(d("-0.05").write_coefficient(&mut s), 2);
    assert_eq!(s, "5");
    assert_eq!(d("-0.05").to_string(), "-0.05");
    assert_eq!(d("-0.05").to_f64(), -0.05);
}

#[test]
fn divisors_over_nineteen_digits_and_wide_quotients_keep_full_precision() {
    assert_eq!(d("2.5").pow(-14).unwrap(), d("0.4").pow(14).unwrap());
    let d36 = d(&"3".repeat(36));
    assert_eq!(d36.checked_div(d36).unwrap(), Dec::ONE);

    let eight36 = d(&"8".repeat(36));
    let q = eight36.checked_div(d("3")).unwrap();
    assert_eq!(digits(q.mantissa()), PRECISION);
    assert_eq!(q.to_string(), format!("{}300", "296296296296296296296296296296296"));

    let m = pow10(34) * 2 + 9;
    let a = Dec::new(m, 0).unwrap();
    let half = a.checked_div(d("2")).unwrap();
    assert_eq!(half, Dec::new(1, -34).unwrap());
}

#[test]
fn shift_handles_extreme_exponents_without_overflow() {
    assert_eq!(Dec::ONE.shift(i32::MIN), None);
    assert_eq!(Dec::ONE.shift(i32::MAX), None);
    assert_eq!(Dec::ZERO.shift(i32::MIN), Some(Dec::ZERO));
}
