//! CRC-32C (Castagnoli), what `java.util.zip.CRC32C` computes. OpenNLP
//! hashes the training events with it (`ChecksumEventStream`) and writes
//! the value into the model manifest as `Training-Eventhash`, and
//! `TwoPassDataIndexer` checks its event spill file with it.

/// The reflected form of the Castagnoli polynomial 0x1EDC6F41.
const POLYNOMIAL: u32 = 0x82F6_3B78;

const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ POLYNOMIAL } else { crc >> 1 };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// A running CRC-32C, fed in pieces as `Checksum.update` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crc32c {
    state: u32,
}

impl Default for Crc32c {
    fn default() -> Crc32c {
        Crc32c::new()
    }
}

impl Crc32c {
    pub fn new() -> Crc32c {
        Crc32c { state: 0xFFFF_FFFF }
    }

    /// Feeds more bytes.
    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.state = TABLE[((self.state ^ u32::from(b)) & 0xFF) as usize] ^ (self.state >> 8);
        }
    }

    /// The checksum of everything fed so far, as `CRC32C.getValue` returns
    /// it (the JVM widens it to a `long` without sign).
    pub fn value(&self) -> u32 {
        !self.state
    }
}

/// The CRC-32C of one byte string.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = Crc32c::new();
    crc.update(data);
    crc.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // The check value every CRC-32C implementation agrees on.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        // 32 zero bytes and 32 0xFF bytes, from RFC 3720 appendix B.4 (which
        // lists the bytes in transmission order, least significant first).
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
    }

    #[test]
    fn pieces_hash_like_the_whole() {
        let mut crc = Crc32c::new();
        crc.update(b"1234");
        crc.update(b"");
        crc.update(b"56789");
        assert_eq!(crc.value(), crc32c(b"123456789"));
    }
}
