use super::*;

#[test]
fn plain_and_scientific_doubles() {
    assert_eq!(double_to_string(1.0), "1.0");
    assert_eq!(double_to_string(-1.5), "-1.5");
    assert_eq!(double_to_string(0.001), "0.001");
    assert_eq!(double_to_string(0.0009), "9.0E-4");
    assert_eq!(double_to_string(1234567.0), "1234567.0");
    assert_eq!(double_to_string(1.0e7), "1.0E7");
    assert_eq!(double_to_string(123456.789), "123456.789");
    assert_eq!(double_to_string(1.0e23), "1.0E23");
    assert_eq!(double_to_string(1.0e-10), "1.0E-10");
    assert_eq!(double_to_string(12345.0e-20), "1.2345E-16");
    assert_eq!(double_to_string(100.0), "100.0");
    assert_eq!(double_to_string(0.1 + 0.2), "0.30000000000000004");
    assert_eq!(double_to_string(f64::MAX), "1.7976931348623157E308");
    assert_eq!(double_to_string(f64::MIN_POSITIVE), "2.2250738585072014E-308");
}

#[test]
fn a_one_digit_shortest_gives_way_to_a_closer_two_digit_decimal() {
    // Double.MIN_VALUE is about 4.94e-324: 5E-324 is the shortest,
    // 4.9E-324 is closer and also rounds to it.
    assert_eq!(double_to_string(f64::from_bits(1)), "4.9E-324");
    assert_eq!(double_to_string(2.0 * f64::from_bits(1)), "9.9E-324");
    assert_eq!(float_to_string(f32::from_bits(1)), "1.4E-45");
    // 2.0E-3 stays: it is exact.
    assert_eq!(double_to_string(0.002), "0.002");
    assert_eq!(double_to_string(2.0e-3), "0.002");
}

#[test]
fn special_values() {
    assert_eq!(double_to_string(f64::NAN), "NaN");
    assert_eq!(double_to_string(f64::INFINITY), "Infinity");
    assert_eq!(double_to_string(f64::NEG_INFINITY), "-Infinity");
    assert_eq!(double_to_string(0.0), "0.0");
    assert_eq!(double_to_string(-0.0), "-0.0");
    assert_eq!(float_to_string(f32::NAN), "NaN");
    assert_eq!(float_to_string(-0.0), "-0.0");
}

#[test]
fn floats() {
    assert_eq!(float_to_string(1.0), "1.0");
    assert_eq!(float_to_string(0.1), "0.1");
    assert_eq!(float_to_string(1.0e7), "1.0E7");
    assert_eq!(float_to_string(3.4028235e38), "3.4028235E38");
    assert_eq!(float_to_string(16777216.0), "1.6777216E7");
    assert_eq!(float_to_string(0.33333334), "0.33333334");
}

#[test]
fn parse_double_follows_javas_grammar() {
    assert_eq!(parse_double("1.5"), Ok(1.5));
    assert_eq!(parse_double("  1.5  "), Ok(1.5));
    assert_eq!(parse_double("1.5f"), Ok(1.5));
    assert_eq!(parse_double("1.5D"), Ok(1.5));
    assert_eq!(parse_double("-.5"), Ok(-0.5));
    assert_eq!(parse_double("5."), Ok(5.0));
    assert_eq!(parse_double("+1e3"), Ok(1000.0));
    assert_eq!(parse_double("1E-3"), Ok(0.001));
    assert_eq!(parse_double("Infinity"), Ok(f64::INFINITY));
    assert_eq!(parse_double("-Infinity"), Ok(f64::NEG_INFINITY));
    assert!(parse_double("NaN").unwrap().is_nan());
    assert_eq!(parse_double("1e400"), Ok(f64::INFINITY));
    assert_eq!(parse_double("1e-400"), Ok(0.0));
    for bad in ["", " ", "inf", "nan", "1e", "1e+", ".", "1.5x", "1,5", "--1", "f", "1.5 f"] {
        assert_eq!(parse_double(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
    }
    assert_eq!(parse_float("0.1"), Ok(0.1f32));
    assert_eq!(parse_float("3.4028236e38"), Ok(f32::INFINITY));
    assert_eq!(parse_float("1.00000001"), Ok(1.0f32));
}

#[test]
fn hexadecimal_literals_follow_javas_grammar() {
    assert_eq!(parse_double("0x1p3"), Ok(8.0));
    assert_eq!(parse_double("0X1.8P1"), Ok(3.0));
    assert_eq!(parse_double("-0x.8p1"), Ok(-1.0));
    assert_eq!(parse_double("+0x1.p0"), Ok(1.0));
    assert_eq!(parse_double(" 0x0001.0000p-0 "), Ok(1.0));
    assert_eq!(parse_double("0x.0001p16"), Ok(1.0));
    assert_eq!(parse_double("0xABCDEFp0"), Ok(11259375.0));
    assert_eq!(parse_double("0xabcdefp0f"), Ok(11259375.0));
    assert_eq!(parse_double("0x1p3D"), Ok(8.0));
    assert_eq!(parse_double("0x1p00003"), Ok(8.0));
    assert_eq!(parse_float("0x1p3"), Ok(8.0f32));
    // Too many digits for the type: rounded to the nearest, ties to even.
    assert_eq!(parse_double("0x1.00000000000008p0"), Ok(1.0));
    assert_eq!(parse_double("0x1.00000000000008000000000000000001p0"), Ok(1.0 + f64::EPSILON));
    assert_eq!(parse_double("0x1.00000000000018p0"), Ok(1.0 + 2.0 * f64::EPSILON));
    assert_eq!(parse_double("0x1.1234567890abcdef1234567890p0"), Ok(f64::from_bits(0x3ff1234567890abd)));
    // The float is rounded from the digits, not from the double: the
    // double of the second is above 1 + 2^-24 and would round up.
    assert_eq!(parse_float("0x1.000001p0"), Ok(1.0f32));
    assert_eq!(parse_double("0x1.000001p0"), Ok(1.0 + 2f64.powi(-24)));
    assert_eq!(parse_float("0x1.000003p0"), Ok(1.0f32 + 2.0 * f32::EPSILON));
    assert_eq!(parse_float("0x1.0000010000001p0"), Ok(1.0f32 + f32::EPSILON));
    // The ends of the ranges: the largest values, the smallest subnormals,
    // and the ties and carries at both.
    assert_eq!(parse_double("0x1.fffffffffffffp1023"), Ok(f64::MAX));
    assert_eq!(parse_double("0x1.fffffffffffff8p1023"), Ok(f64::INFINITY));
    assert_eq!(parse_double("0x1p1024"), Ok(f64::INFINITY));
    assert_eq!(parse_double("-0x1p1024"), Ok(f64::NEG_INFINITY));
    assert_eq!(parse_double("0x1p-1022"), Ok(f64::MIN_POSITIVE));
    assert_eq!(parse_double("0x1.ffffffffffffep-1023"), Ok(f64::MIN_POSITIVE - f64::from_bits(1)));
    // Half an ulp under the smallest normal: a tie, to the even mantissa.
    assert_eq!(parse_double("0x1.fffffffffffffp-1023"), Ok(f64::MIN_POSITIVE));
    assert_eq!(parse_double("0x1.fffffffffffff8p-1023"), Ok(f64::MIN_POSITIVE));
    assert_eq!(parse_double("0x1p-1074"), Ok(f64::from_bits(1)));
    assert_eq!(parse_double("0x1p-1075"), Ok(0.0));
    assert_eq!(parse_double("0x1.8p-1075"), Ok(f64::from_bits(1)));
    assert_eq!(parse_double("0x1.0000000000000000000000000001p-1075"), Ok(f64::from_bits(1)));
    assert_eq!(parse_double("0x1p-1076"), Ok(0.0));
    assert!(parse_double("-0x1p-1076").unwrap().is_sign_negative());
    assert_eq!(parse_float("0x1.fffffep127"), Ok(f32::MAX));
    assert_eq!(parse_float("0x1.ffffffp127"), Ok(f32::INFINITY));
    assert_eq!(parse_double("0x1.ffffffp127"), Ok(3.4028235677973366e38));
    assert_eq!(parse_float("0x1p-126"), Ok(f32::MIN_POSITIVE));
    assert_eq!(parse_float("0x1.ffffffp-127"), Ok(f32::MIN_POSITIVE));
    assert_eq!(parse_float("0x1p-149"), Ok(f32::from_bits(1)));
    assert_eq!(parse_float("0x1p-150"), Ok(0.0f32));
    assert_eq!(parse_float("0x1.8p-150"), Ok(f32::from_bits(1)));
    assert_eq!(parse_float("0x1p-151"), Ok(0.0f32));
    // A zero significand is zero whatever the exponent, and an exponent
    // over an int is infinity or zero by its sign.
    assert_eq!(parse_double("0x0p99999999999"), Ok(0.0));
    assert!(parse_double("-0x0.000p-99999999999").unwrap().is_sign_negative());
    assert_eq!(parse_double("0x1p2147483647"), Ok(f64::INFINITY));
    assert_eq!(parse_double("0x1p2147483648"), Ok(f64::INFINITY));
    assert_eq!(parse_double("0x1p-2147483648"), Ok(0.0));
    assert_eq!(parse_double("0x1p-2147483647"), Ok(0.0));
    assert_eq!(parse_float("0x1p99999999999"), Ok(f32::INFINITY));
    for bad in [
        "0x",
        "0x1",
        "0x1.8",
        "0xp1",
        "0x.p1",
        "0x1p",
        "0x1p+",
        "0x1p3.5",
        "0x1e5",
        "0x1p0x3",
        "0x+1p3",
        "0x1p3x",
        "0x1p3 f",
        "0x1pg",
        "0x1p\u{663}",
        "0xg1p0",
        "0x1,8p1",
    ] {
        assert_eq!(parse_double(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
        assert_eq!(parse_float(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
    }
}

#[test]
fn parse_int_and_boolean() {
    assert_eq!(parse_int("5"), Ok(5));
    assert_eq!(parse_int("-5"), Ok(-5));
    assert_eq!(parse_int("+5"), Ok(5));
    assert_eq!(parse_int("2147483647"), Ok(i32::MAX));
    assert_eq!(parse_int("-2147483648"), Ok(i32::MIN));
    for bad in ["", "-", "2147483648", " 5", "5 ", "1.0", "0x10"] {
        assert_eq!(parse_int(bad), Err(ParseError::NotANumber(bad.into())), "{bad:?}");
    }
    assert!(parse_boolean("true"));
    assert!(parse_boolean("TRUE"));
    assert!(parse_boolean("tRuE"));
    assert!(!parse_boolean("yes"));
    assert!(!parse_boolean(" true"));
    assert!(!parse_boolean(""));
}
