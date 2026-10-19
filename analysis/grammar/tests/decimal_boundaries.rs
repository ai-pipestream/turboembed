use turbo_grammar::Dec;

#[test]
fn mantissa_rounding_rejects_scale_underflow() {
    for mantissa in [i128::MIN, i128::MAX] {
        assert!(Dec::new(mantissa, i32::MIN).is_none());
    }
}

#[test]
fn trailing_zero_normalization_rejects_scale_underflow() {
    for mantissa in [-10, 10] {
        assert!(Dec::new(mantissa, i32::MIN).is_none());
    }
}

#[test]
fn extreme_decimal_operations_are_checked() {
    let large = Dec::new(1, i16::MIN.into()).unwrap();
    let tiny = Dec::new(1, i16::MAX.into()).unwrap();
    assert!(large.checked_mul(Dec::from_i64(10)).is_none());
    assert!(tiny.checked_div(Dec::from_i64(10)).is_none());
    assert!(large.checked_div(tiny).is_none());
    assert!(tiny.checked_div(large).is_none());
    assert_eq!(large.checked_div(large), Some(Dec::ONE));
    assert_eq!(large.checked_mul(Dec::ZERO), Some(Dec::ZERO));
    assert_eq!(large.checked_div(Dec::ZERO), None);
    assert_eq!(Dec::ONE.pow(i32::MIN), None);
}
