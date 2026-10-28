use super::*;

const TABLE: &str = include_str!("../../../testdata/quantize/sentence-transformers-145ef47d.txt");

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

fn floats(hex: &str) -> Vec<f32> {
    bytes(hex).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[test]
fn codes_match_sentence_transformers_byte_for_byte() {
    let mut seen = 0;
    for line in TABLE.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        let dim: usize = f[0].parse().unwrap();
        let (x, ranges) = (floats(f[1]), floats(f[2]));
        assert_eq!((x.len(), ranges.len()), (dim, 2 * dim), "{line}");
        check_int8_ranges(&ranges).unwrap();
        let mut int8 = vec![0u8; dim];
        int8_row(&x, &ranges, &mut int8);
        assert_eq!(int8, bytes(f[3]), "int8, dim {dim}: {line}");
        let mut bits = vec![0u8; dim.div_ceil(8)];
        ubinary_row(&x, &mut bits);
        assert_eq!(bits, bytes(f[4]), "ubinary, dim {dim}: {line}");
        seen += 1;
    }
    assert_eq!(seen, 30, "five widths of six rows each");
}

#[test]
fn int8_buckets_by_hand() {
    // min 0, max 255: step 1, so a value's code is its floor less 128.
    let ranges = [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0, 255.0];
    let mut out = [0u8; 5];
    int8_row(&[-3.0, 0.0, 0.999, 254.5, 1e9], &ranges, &mut out);
    assert_eq!(out.map(|b| b as i8), [-128, -128, -128, 126, 127]);
    // A dimension whose minimum is its maximum has step 1, not a division by 0.
    int8_row(&[2.0, 2.5, 1.0, 9.0, 0.0], &[2.0; 10], &mut out);
    assert_eq!(out.map(|b| b as i8), [-128, -128, -128, -121, -128]);
}

#[test]
fn ubinary_bits_by_hand() {
    let mut out = [0u8; 2];
    // The first value is the top bit; 0, -0 and NaN are 0 bits.
    let x = [1.0, -1.0, 0.0, -0.0, f32::NAN, 1e-30, f32::INFINITY, -2.0, 3.0];
    ubinary_row(&x, &mut out);
    assert_eq!(out, [0b1000_0110, 0b1000_0000]);
}

#[test]
fn row_bytes_and_rows() {
    assert_eq!([1, 7, 8, 9, 64].map(|d| Quantization::UBinary.row_bytes(d)), [1, 1, 1, 2, 8], "(dim + 7) / 8 bytes");
    assert_eq!((Quantization::Int8.row_bytes(9), Quantization::F32.row_bytes(9)), (9, 36));
    let x = [1.0f32, -1.0, -1.0, 1.0];
    let mut out = [0u8; 2];
    rows(Quantization::UBinary, &x, 2, &[], &mut out);
    assert_eq!(out, [0x80, 0x40]);
}

#[test]
fn ranges_are_refused_by_reason() {
    assert!(check_int8_ranges(&[0.0, 1.0, 2.0]).unwrap_err().contains("not 2 per dimension"));
    assert!(check_int8_ranges(&[f32::NAN, 1.0]).unwrap_err().contains("not both finite"));
    assert!(check_int8_ranges(&[0.0, f32::INFINITY]).unwrap_err().contains("not both finite"));
    assert!(check_int8_ranges(&[2.0, 1.0]).unwrap_err().contains("minimum 2 is over its maximum 1"));
    check_int8_ranges(&[1.0, 1.0]).unwrap();
}
