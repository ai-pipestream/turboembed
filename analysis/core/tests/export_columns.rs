//! The column helpers the layer codecs share, against bytes written
//! out by hand: a column is its length as a little-endian `u32`, then
//! its values, and a variant is one byte, its position in its table. A
//! writer and a reader changed together still agree with each other,
//! so only these bytes hold the format still. The refusals are pinned
//! by their words, which are all a caller is told.

use turbo_analysis::export::{
    ExportError, Reader, Writer, get_f64s, get_u32s, get_variant, get_variants, put_f64s, put_u32s, put_variant,
    put_variants,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Shade {
    Red,
    Green,
    Blue,
}

const SHADES: [Shade; 3] = [Shade::Red, Shade::Green, Shade::Blue];

fn refusal<T: std::fmt::Debug>(result: Result<T, ExportError>) -> String {
    match result {
        Err(ExportError::Malformed(detail)) => detail,
        other => panic!("not a malformed refusal: {other:?}"),
    }
}

#[test]
fn a_u32_column_is_its_length_then_each_value_low_byte_first() {
    let mut w = Writer::new();
    put_u32s(&mut w, &[1, 0x0102_0304, u32::MAX]).unwrap();
    let bytes = w.finish();
    assert_eq!(bytes, [3, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 255, 255, 255, 255]);
    let mut r = Reader::new(&bytes);
    assert_eq!(get_u32s(&mut r).unwrap(), [1, 0x0102_0304, u32::MAX]);
    assert!(r.is_done());

    let mut w = Writer::new();
    put_u32s(&mut w, &[]).unwrap();
    assert_eq!(w.finish(), [0, 0, 0, 0]);
}

#[test]
fn an_f64_column_is_its_length_then_the_bits_of_each_value() {
    let mut w = Writer::new();
    put_f64s(&mut w, &[1.0, -2.5]).unwrap();
    let bytes = w.finish();
    // 1.0 is 0x3ff0000000000000 and -2.5 is 0xc004000000000000.
    assert_eq!(bytes, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xf0, 0x3f, 0, 0, 0, 0, 0, 0, 0x04, 0xc0]);
    let mut r = Reader::new(&bytes);
    assert_eq!(get_f64s(&mut r).unwrap(), [1.0, -2.5]);
    assert!(r.is_done());
}

#[test]
fn a_variant_is_one_byte_its_place_in_the_table() {
    let mut w = Writer::new();
    put_variant(&mut w, &SHADES, Shade::Blue, "shade").unwrap();
    put_variants(&mut w, &SHADES, &[Shade::Green, Shade::Red, Shade::Blue], "shade").unwrap();
    let bytes = w.finish();
    assert_eq!(bytes, [2, 3, 0, 0, 0, 1, 0, 2]);
    let mut r = Reader::new(&bytes);
    assert_eq!(get_variant(&mut r, &SHADES, "shade").unwrap(), Shade::Blue);
    assert_eq!(get_variants(&mut r, &SHADES, "shade").unwrap(), [Shade::Green, Shade::Red, Shade::Blue]);
    assert!(r.is_done());
}

#[test]
fn a_variant_outside_its_table_is_refused_by_name() {
    // Read: the position is past the table, alone and in a column.
    assert_eq!(refusal(get_variant(&mut Reader::new(&[3]), &SHADES, "shade")), "a shade discriminant of 3");
    assert_eq!(
        refusal(get_variants(&mut Reader::new(&[2, 0, 0, 0, 1, 7]), &SHADES, "shade")),
        "a shade discriminant of 7"
    );

    // Written: the table does not hold the value, or is longer than a
    // byte can index. Nothing is written for the value refused.
    let short = [Shade::Red, Shade::Green];
    let mut w = Writer::new();
    assert_eq!(refusal(put_variant(&mut w, &short, Shade::Blue, "shade")), "a shade outside its table");
    assert_eq!(refusal(put_variants(&mut w, &short, &[Shade::Red, Shade::Blue], "shade")), "a shade outside its table");
    assert_eq!(w.finish(), [2, 0, 0, 0, 0]);

    let long: Vec<u16> = (0..=256).collect();
    let mut w = Writer::new();
    put_variant(&mut w, &long, 255, "code").unwrap();
    assert_eq!(refusal(put_variant(&mut w, &long, 256, "code")), "a code table longer than a u8 can index");
    assert_eq!(w.finish(), [255]);
}
