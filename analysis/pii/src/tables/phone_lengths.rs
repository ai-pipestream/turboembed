//! National number lengths by E.164 calling code, OpenNLP's
//! `PhoneNumberLengths`: per assigned code, the union of the national
//! lengths any territory under it uses, from libphonenumber's
//! `PhoneNumberMetadata.xml` (Apache License 2.0) at revision
//! `4ad67e90c65e` of 2026-08-28.

/// Calling code and length mask pairs, by code: bit `n` of a mask allows
/// a national number of `n` digits.
const CODE_AND_MASK: [(u16, u32); 215] = [
    (1, 0x480),
    (7, 0x4400),
    (20, 0x700),
    (27, 0x7E0),
    (30, 0x1C00),
    (31, 0xFE0),
    (32, 0x300),
    (33, 0x200),
    (34, 0x200),
    (36, 0x300),
    (39, 0x1FC0),
    (40, 0x240),
    (41, 0x1200),
    (43, 0x3FF0),
    (44, 0x680),
    (45, 0x100),
    (46, 0x17C0),
    (47, 0x120),
    (48, 0x7C0),
    (49, 0xFFF0),
    (51, 0x300),
    (52, 0x400),
    (53, 0x5C0),
    (54, 0xC00),
    (55, 0xF00),
    (56, 0xE00),
    (57, 0xD00),
    (58, 0x400),
    (60, 0x700),
    (61, 0x17E0),
    (62, 0x3FF80),
    (63, 0x3F40),
    (64, 0x7E0),
    (65, 0xD00),
    (66, 0x2700),
    (81, 0x3FF00),
    (82, 0x7F60),
    (84, 0x780),
    (86, 0x1F80),
    (90, 0x3480),
    (91, 0x3F00),
    (92, 0x1F00),
    (93, 0x200),
    (94, 0x200),
    (95, 0x7C0),
    (98, 0x4F0),
    (211, 0x200),
    (212, 0x200),
    (213, 0x300),
    (216, 0x100),
    (218, 0x200),
    (220, 0x280),
    (221, 0x200),
    (222, 0x100),
    (223, 0x100),
    (224, 0x300),
    (225, 0x400),
    (226, 0x100),
    (227, 0x100),
    (228, 0x100),
    (229, 0x500),
    (230, 0x580),
    (231, 0x380),
    (232, 0x100),
    (233, 0x300),
    (234, 0x7C00),
    (235, 0x100),
    (236, 0x100),
    (237, 0x300),
    (238, 0x80),
    (239, 0x80),
    (240, 0x200),
    (241, 0x180),
    (242, 0x200),
    (243, 0x780),
    (244, 0x200),
    (245, 0x280),
    (246, 0x80),
    (247, 0x60),
    (248, 0x80),
    (249, 0x200),
    (250, 0x300),
    (251, 0x200),
    (252, 0x3C0),
    (253, 0x100),
    (254, 0x780),
    (255, 0x200),
    (256, 0x200),
    (257, 0x100),
    (258, 0x300),
    (260, 0x200),
    (261, 0x200),
    (262, 0x200),
    (263, 0x680),
    (264, 0x300),
    (265, 0x280),
    (266, 0x100),
    (267, 0x580),
    (268, 0x300),
    (269, 0x80),
    (290, 0x30),
    (291, 0x80),
    (297, 0x80),
    (298, 0x40),
    (299, 0x40),
    (350, 0x100),
    (351, 0x200),
    (352, 0xFF0),
    (353, 0x780),
    (354, 0x280),
    (355, 0x3C0),
    (356, 0x100),
    (357, 0x100),
    (358, 0x1FE0),
    (359, 0x13C0),
    (370, 0x100),
    (371, 0x100),
    (372, 0x580),
    (373, 0x100),
    (374, 0x100),
    (375, 0xFC0),
    (376, 0x340),
    (377, 0x300),
    (378, 0x500),
    (380, 0x600),
    (381, 0x1FC0),
    (382, 0x300),
    (383, 0x1F00),
    (385, 0x380),
    (386, 0x1E0),
    (387, 0x300),
    (389, 0x100),
    (420, 0x1E00),
    (421, 0x2C0),
    (423, 0x280),
    (500, 0x20),
    (501, 0x880),
    (502, 0x900),
    (503, 0x980),
    (504, 0x900),
    (505, 0x100),
    (506, 0x500),
    (507, 0xD80),
    (508, 0x240),
    (509, 0x100),
    (590, 0x200),
    (591, 0x300),
    (592, 0x80),
    (593, 0xF00),
    (594, 0x200),
    (595, 0xFC0),
    (596, 0x200),
    (597, 0xC0),
    (598, 0x3FF0),
    (599, 0x180),
    (670, 0x180),
    (672, 0x40),
    (673, 0x80),
    (674, 0x80),
    (675, 0x180),
    (676, 0xA0),
    (677, 0xA0),
    (678, 0xA0),
    (679, 0x880),
    (680, 0x80),
    (681, 0x240),
    (682, 0x20),
    (683, 0x90),
    (685, 0x4E0),
    (686, 0x120),
    (687, 0x40),
    (688, 0xE0),
    (689, 0x340),
    (690, 0xF0),
    (691, 0x80),
    (692, 0x80),
    (800, 0x100),
    (808, 0x100),
    (850, 0x500),
    (852, 0xBE0),
    (853, 0x180),
    (855, 0x700),
    (856, 0x700),
    (870, 0x1200),
    (878, 0x1000),
    (880, 0x7C0),
    (881, 0x600),
    (882, 0x1F80),
    (883, 0x1F00),
    (886, 0xF80),
    (888, 0x800),
    (960, 0x480),
    (961, 0x180),
    (962, 0x300),
    (963, 0x300),
    (964, 0x700),
    (965, 0x180),
    (966, 0x600),
    (967, 0x380),
    (968, 0x380),
    (970, 0x700),
    (971, 0x1FE0),
    (972, 0x1F80),
    (973, 0x100),
    (974, 0xB80),
    (975, 0x180),
    (976, 0x700),
    (977, 0xD00),
    (979, 0x200),
    (992, 0x200),
    (993, 0x100),
    (994, 0x200),
    (995, 0x200),
    (996, 0x600),
    (998, 0x200),
];

/// The mask of every code below 1000, 0 for an unassigned one.
const MASKS: [u32; 1000] = {
    let mut m = [0u32; 1000];
    let mut i = 0;
    while i < CODE_AND_MASK.len() {
        m[CODE_AND_MASK[i].0 as usize] = CODE_AND_MASK[i].1;
        i += 1;
    }
    m
};

/// `PhoneNumberLengths.plausibleInternational`: whether the leading one
/// to three of the ASCII `digits` are an assigned calling code with the
/// rest a national length used under it. A code never starts with 0.
pub(crate) fn plausible_international(digits: &[u8]) -> bool {
    if digits.first().is_none_or(|&d| d == b'0') {
        return false;
    }
    let mut code = 0usize;
    for (i, &d) in digits.iter().take(3).enumerate() {
        code = code * 10 + usize::from(d - b'0');
        let national = digits.len() - i - 1;
        if national > 0 && national <= 31 && MASKS[code] & (1 << national) != 0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_lengths() {
        assert!(CODE_AND_MASK.windows(2).all(|w| w[0].0 < w[1].0));
        // +1 takes 7 or 10 national digits, +44 7, 9 or 10.
        assert!(plausible_international(b"15555550100"));
        assert!(plausible_international(b"442079460958"));
        assert!(!plausible_international(b"1555555010"));
        assert!(!plausible_international(b"0442079460958"));
        assert!(!plausible_international(b""));
        assert!(!plausible_international(b"1"));
        // 999 is unassigned.
        assert!(!plausible_international(b"9991234567"));
    }

    #[test]
    fn a_zero_prefix_never_reuses_an_assigned_code() {
        let mut accepted = 0;
        for code in 1..1000 {
            let prefix = code.to_string();
            for length in 1..=15 - prefix.len() {
                let digits = format!("{prefix}{}", "2".repeat(length));
                if plausible_international(digits.as_bytes()) {
                    accepted += 1;
                    for zeros in ["0", "00"] {
                        assert!(!plausible_international(format!("{zeros}{digits}").as_bytes()), "{digits}");
                    }
                }
            }
        }
        assert!(accepted > 200);
    }

    #[test]
    fn national_lengths_follow_the_metadata() {
        for (code, lengths) in [("220", &[7, 9][..]), ("263", &[7, 9, 10][..])] {
            for length in 1..=12 {
                let digits = format!("{code}{}", "2".repeat(length));
                assert_eq!(plausible_international(digits.as_bytes()), lengths.contains(&length), "{digits}");
            }
        }
        for digits in ["390212345678", "441020304050", "10000000000"] {
            assert!(plausible_international(digits.as_bytes()));
        }
    }
}
