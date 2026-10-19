//! Raw Deflate controls from Python zlib 1.3.2; no runtime dependency.
use super::*;

#[test]
fn zlib_fixed_and_dynamic_streams_match() {
    let mut unit = b"abracadabra ".to_vec();
    unit.extend(0u8..32);
    let expected = unit.repeat(300);
    let encoded_0 = &[
        75, 76, 42, 74, 76, 78, 76, 73, 4, 82, 10, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124,
        252, 2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 137, 163, 106, 71, 213, 142, 170,
        29, 85, 59, 170, 118, 84, 237, 168, 218, 81, 181, 163, 106, 71, 213, 142, 170, 29, 85, 59, 170, 118, 84, 237,
        168, 218, 81, 181, 163, 106, 71, 213, 142, 170, 29, 85, 59, 170, 118, 84, 237, 168, 218, 81, 181, 163, 106, 71,
        213, 142, 170, 29, 85, 59, 170, 118, 84, 237, 168, 218, 81, 181, 163, 106, 71, 213, 142, 170, 29, 85, 59, 170,
        118, 84, 237, 168, 218, 81, 181, 163, 106, 71, 213, 142, 170, 29, 85, 59, 170, 118, 84, 237, 168, 218, 81, 181,
        163, 106, 71, 213, 142, 112, 181, 0,
    ];
    assert_eq!(inflate(encoded_0, expected.len()).unwrap(), expected);
    for end in 0..encoded_0.len() {
        assert!(inflate(&encoded_0[..end], expected.len()).is_err());
    }
    let encoded_1 = &[
        237, 202, 89, 2, 66, 0, 20, 0, 64, 164, 141, 68, 43, 69, 203, 213, 30, 78, 208, 253, 63, 234, 30, 205, 215,
        252, 76, 140, 159, 152, 98, 142, 31, 239, 36, 205, 22, 249, 114, 181, 222, 108, 139, 114, 87, 237, 235, 230,
        112, 60, 157, 47, 215, 182, 187, 221, 251, 225, 241, 124, 133, 235, 186, 174, 235, 186, 174, 235, 186, 174,
        235, 186, 174, 235, 186, 174, 235, 186, 174, 235, 186, 174, 235, 186, 174, 235, 186, 174, 235, 186, 174, 235,
        186, 174, 235, 186, 174, 235, 254, 249, 253, 2,
    ];
    assert_eq!(inflate(encoded_1, expected.len()).unwrap(), expected);
    for end in 0..encoded_1.len() {
        assert!(inflate(&encoded_1[..end], expected.len()).is_err());
    }
}

#[test]
fn small_arbitrary_streams_do_not_panic() {
    let mut seed = 7u64;
    for n in 0..4096 {
        let mut bytes = vec![0u8; n % 64];
        for byte in &mut bytes {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (seed >> 32) as u8;
        }
        let _ = inflate(&bytes, n % 128);
    }
}
