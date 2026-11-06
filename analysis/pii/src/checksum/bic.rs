//! ISO 9362 Business Identifier Code (BIC / SWIFT) structural and ISO 3166-1
//! alpha-2 country validation.

/// Assigned ISO 3166-1 alpha-2 country codes (plus `XK`), packed as one
/// 26-bit row per first letter `A..=Z`.
const fn country_table() -> [u32; 26] {
    const CODES: &[&[u8; 2]] = &[
        b"AD", b"AE", b"AF", b"AG", b"AI", b"AL", b"AM", b"AO", b"AQ", b"AR", b"AS", b"AT", b"AU", b"AW", b"AX", b"AZ",
        b"BA", b"BB", b"BD", b"BE", b"BF", b"BG", b"BH", b"BI", b"BJ", b"BL", b"BM", b"BN", b"BO", b"BQ", b"BR", b"BS",
        b"BT", b"BV", b"BW", b"BY", b"BZ", b"CA", b"CC", b"CD", b"CF", b"CG", b"CH", b"CI", b"CK", b"CL", b"CM", b"CN",
        b"CO", b"CR", b"CU", b"CV", b"CW", b"CX", b"CY", b"CZ", b"DE", b"DJ", b"DK", b"DM", b"DO", b"DZ", b"EC", b"EE",
        b"EG", b"EH", b"ER", b"ES", b"ET", b"FI", b"FJ", b"FK", b"FM", b"FO", b"FR", b"GA", b"GB", b"GD", b"GE", b"GF",
        b"GG", b"GH", b"GI", b"GL", b"GM", b"GN", b"GP", b"GQ", b"GR", b"GS", b"GT", b"GU", b"GW", b"GY", b"HK", b"HM",
        b"HN", b"HR", b"HT", b"HU", b"ID", b"IE", b"IL", b"IM", b"IN", b"IO", b"IQ", b"IR", b"IS", b"IT", b"JE", b"JM",
        b"JO", b"JP", b"KE", b"KG", b"KH", b"KI", b"KM", b"KN", b"KP", b"KR", b"KW", b"KY", b"KZ", b"LA", b"LB", b"LC",
        b"LI", b"LK", b"LR", b"LS", b"LT", b"LU", b"LV", b"LY", b"MA", b"MC", b"MD", b"ME", b"MF", b"MG", b"MH", b"MK",
        b"ML", b"MM", b"MN", b"MO", b"MP", b"MQ", b"MR", b"MS", b"MT", b"MU", b"MV", b"MW", b"MX", b"MY", b"MZ", b"NA",
        b"NC", b"NE", b"NF", b"NG", b"NI", b"NL", b"NO", b"NP", b"NR", b"NU", b"NZ", b"OM", b"PA", b"PE", b"PF", b"PG",
        b"PH", b"PK", b"PL", b"PM", b"PN", b"PR", b"PS", b"PT", b"PW", b"PY", b"QA", b"RE", b"RO", b"RS", b"RU", b"RW",
        b"SA", b"SB", b"SC", b"SD", b"SE", b"SG", b"SH", b"SI", b"SJ", b"SK", b"SL", b"SM", b"SN", b"SO", b"SR", b"SS",
        b"ST", b"SV", b"SX", b"SY", b"SZ", b"TC", b"TD", b"TF", b"TG", b"TH", b"TJ", b"TK", b"TL", b"TM", b"TN", b"TO",
        b"TR", b"TT", b"TV", b"TW", b"TZ", b"UA", b"UG", b"UM", b"US", b"UY", b"UZ", b"VA", b"VC", b"VE", b"VG", b"VI",
        b"VN", b"VU", b"WF", b"WS", b"XK", b"YE", b"YT", b"ZA", b"ZM", b"ZW",
    ];
    let mut rows = [0u32; 26];
    let mut i = 0;
    while i < CODES.len() {
        let r = (CODES[i][0] - b'A') as usize;
        let c = (CODES[i][1] - b'A') as u32;
        rows[r] |= 1u32 << c;
        i += 1;
    }
    rows
}

const COUNTRIES: [u32; 26] = country_table();

/// Returns whether `(c0, c1)` is an assigned ISO 3166-1 alpha-2 country code.
#[inline]
pub const fn is_iso_country(c0: u8, c1: u8) -> bool {
    if !c0.is_ascii_uppercase() || !c1.is_ascii_uppercase() {
        return false;
    }
    (COUNTRIES[(c0 - b'A') as usize] & (1u32 << (c1 - b'A'))) != 0
}

/// Checks an 8- or 11-byte uppercase ISO 9362 Business Identifier Code (BIC):
/// 4 uppercase institution letters, 2 ISO 3166-1 alpha-2 country letters,
/// 2 alphanumeric location characters, and an optional 3-character
/// alphanumeric branch code.
pub fn check(bytes: &[u8]) -> bool {
    if bytes.len() != 8 && bytes.len() != 11 {
        return false;
    }
    if !bytes[..4].iter().all(|b| b.is_ascii_uppercase()) {
        return false;
    }
    if !is_iso_country(bytes[4], bytes[5]) {
        return false;
    }
    bytes[6..].iter().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}
