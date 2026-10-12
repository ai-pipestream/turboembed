//! CRC-32 as zip uses it (the IEEE 802.3 polynomial, reflected, with the
//! initial value and final value all ones), computed eight bytes at a
//! time with the slicing-by-8 tables.

const POLYNOMIAL: u32 = 0xEDB8_8320;

const TABLES: [[u32; 256]; 8] = tables();

const fn tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLYNOMIAL } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut k = 1;
        while k < 8 {
            let prev = t[k - 1][i];
            t[k][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            k += 1;
        }
        i += 1;
    }
    t
}

/// The CRC-32 of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    let (words, rest) = data.as_chunks::<8>();
    for w in words {
        let lo = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) ^ crc;
        let hi = u32::from_le_bytes([w[4], w[5], w[6], w[7]]);
        crc = TABLES[7][(lo & 0xFF) as usize]
            ^ TABLES[6][((lo >> 8) & 0xFF) as usize]
            ^ TABLES[5][((lo >> 16) & 0xFF) as usize]
            ^ TABLES[4][(lo >> 24) as usize]
            ^ TABLES[3][(hi & 0xFF) as usize]
            ^ TABLES[2][((hi >> 8) & 0xFF) as usize]
            ^ TABLES[1][((hi >> 16) & 0xFF) as usize]
            ^ TABLES[0][(hi >> 24) as usize];
    }
    for &b in rest {
        crc = (crc >> 8) ^ TABLES[0][((crc ^ u32::from(b)) & 0xFF) as usize];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::crc32;

    #[test]
    fn known_values() {
        // The check value of CRC-32/ISO-HDLC, and values java.util.zip.CRC32
        // gives for the same bytes.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
    }

    #[test]
    fn every_length_and_alignment_matches_the_bytewise_form() {
        let data: Vec<u8> = (0..300u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for start in 0..9 {
            for end in start..data.len() {
                let slice = &data[start..end];
                let mut c = !0u32;
                for &b in slice {
                    c ^= u32::from(b);
                    for _ in 0..8 {
                        c = if c & 1 != 0 { (c >> 1) ^ super::POLYNOMIAL } else { c >> 1 };
                    }
                }
                assert_eq!(crc32(slice), !c);
            }
        }
    }
}
