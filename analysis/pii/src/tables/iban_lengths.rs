//! IBAN lengths by country from the ISO 13616 registry, OpenNLP's
//! `IbanLengths` (its country and length projection has the SHA-256
//! `0eb634a364ca9ba6b2f5541d3fcd18ff63300e5a027c634f0b8db83af5e994be`).

const REGISTRY: [([u8; 2], u8); 89] = [
    (*b"AD", 24),
    (*b"AE", 23),
    (*b"AL", 28),
    (*b"AT", 20),
    (*b"AZ", 28),
    (*b"BA", 20),
    (*b"BE", 16),
    (*b"BG", 22),
    (*b"BH", 22),
    (*b"BI", 27),
    (*b"BR", 29),
    (*b"BY", 28),
    (*b"CH", 21),
    (*b"CR", 22),
    (*b"CY", 28),
    (*b"CZ", 24),
    (*b"DE", 22),
    (*b"DJ", 27),
    (*b"DK", 18),
    (*b"DO", 28),
    (*b"EE", 20),
    (*b"EG", 29),
    (*b"ES", 24),
    (*b"FI", 18),
    (*b"FK", 18),
    (*b"FO", 18),
    (*b"FR", 27),
    (*b"GB", 22),
    (*b"GE", 22),
    (*b"GI", 23),
    (*b"GL", 18),
    (*b"GR", 27),
    (*b"GT", 28),
    (*b"HN", 28),
    (*b"HR", 21),
    (*b"HU", 28),
    (*b"IE", 22),
    (*b"IL", 23),
    (*b"IQ", 23),
    (*b"IS", 26),
    (*b"IT", 27),
    (*b"JO", 30),
    (*b"KW", 30),
    (*b"KZ", 20),
    (*b"LB", 28),
    (*b"LC", 32),
    (*b"LI", 21),
    (*b"LT", 20),
    (*b"LU", 20),
    (*b"LV", 21),
    (*b"LY", 25),
    (*b"MC", 27),
    (*b"MD", 24),
    (*b"ME", 22),
    (*b"MK", 19),
    (*b"MN", 20),
    (*b"MR", 27),
    (*b"MT", 31),
    (*b"MU", 30),
    (*b"NI", 28),
    (*b"NL", 18),
    (*b"NO", 15),
    (*b"OM", 23),
    (*b"PK", 24),
    (*b"PL", 28),
    (*b"PS", 29),
    (*b"PT", 25),
    (*b"QA", 29),
    (*b"RO", 24),
    (*b"RS", 22),
    (*b"RU", 33),
    (*b"SA", 24),
    (*b"SC", 31),
    (*b"SD", 18),
    (*b"SE", 24),
    (*b"SI", 19),
    (*b"SK", 24),
    (*b"SM", 27),
    (*b"SO", 23),
    (*b"ST", 25),
    (*b"SV", 28),
    (*b"TL", 23),
    (*b"TN", 24),
    (*b"TR", 26),
    (*b"UA", 29),
    (*b"VA", 22),
    (*b"VG", 24),
    (*b"XK", 20),
    (*b"YE", 30),
];

/// The length of every pair of capitals, 0 for an unregistered country.
const LENGTHS: [u8; 26 * 26] = {
    let mut l = [0u8; 26 * 26];
    let mut i = 0;
    while i < REGISTRY.len() {
        let (code, length) = REGISTRY[i];
        l[(code[0] - b'A') as usize * 26 + (code[1] - b'A') as usize] = length;
        i += 1;
    }
    l
};

/// The registered IBAN length of the country with the two capitals, or 0.
pub(crate) fn registered_length(first: u8, second: u8) -> usize {
    LENGTHS[usize::from(first - b'A') * 26 + usize::from(second - b'A')].into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lengths() {
        assert!(REGISTRY.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(registered_length(b'D', b'E'), 22);
        assert_eq!(registered_length(b'N', b'O'), 15);
        assert_eq!(registered_length(b'R', b'U'), 33);
        assert_eq!(registered_length(b'U', b'S'), 0);
        assert_eq!(registered_length(b'Z', b'Z'), 0);
    }
}
